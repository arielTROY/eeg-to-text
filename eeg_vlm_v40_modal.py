import modal
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


app = modal.App("eeg-vlm-v40")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v40", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)


def _download_models():
    from transformers import AutoModelForCausalLM, AutoTokenizer, VideoMAEModel

    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install(
        [
            "torch>=2.4.0",
            "transformers>=4.45.0",
            "peft>=0.12",
            "numpy",
            "scipy",
            "jiwer",
            "einops",
            "h5py",
            "mne",
            "pandas",
            "sentence-transformers",
            "accelerate",
            "osfclient",
            "qwen-vl-utils",
            "editdistance",
            "sentencepiece",
            "pronouncing",
        ]
    )
    .run_function(_download_models)
)


# V4.0:
#   [MAE] Masked EEG-video pretraining before supervised CTC.
#   [CTC] Dual heads: character CTC remains the text-facing path, phoneme CTC is
#         an auxiliary phonological objective for Phase 1 only.
#   [P2] Phase 2 text prompting path stays char-based.


CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz0123456789.,!?\'\" ()"
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(CHAR_VOCAB)}
CHAR_VOCAB_SIZE = len(CHAR_VOCAB)

ARPABET_PHONES = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
PHONE_BOUNDARY = "|"
PHONE_FALLBACKS = [f"#{c.upper()}" for c in "abcdefghijklmnopqrstuvwxyz"]
PHONE_VOCAB = ["_"] + ARPABET_PHONES + [PHONE_BOUNDARY] + PHONE_FALLBACKS
PHONE_TO_ID = {p: i for i, p in enumerate(PHONE_VOCAB)}
ID_TO_PHONE = {i: p for i, p in enumerate(PHONE_VOCAB)}
PHONE_VOCAB_SIZE = len(PHONE_VOCAB)

VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

_PHONE_CACHE = {}
_PRONOUNCING = None


def get_pronouncing():
    global _PRONOUNCING
    if _PRONOUNCING is None:
        try:
            import pronouncing as _p

            _PRONOUNCING = _p
        except Exception:
            _PRONOUNCING = False
    return _PRONOUNCING or None


def text_to_char_ids(text):
    return [CHAR_TO_ID[c] for c in text.lower() if c in CHAR_TO_ID]


def _collapse_ctc_ids(seq):
    out = []
    prev = -1
    for tok_id in seq:
        if tok_id != prev and tok_id != 0:
            out.append(tok_id)
        prev = tok_id
    return out


def decode_char_ids(ids):
    return "".join(ID_TO_CHAR.get(i, "") for i in ids)


def decode_phone_ids(ids):
    return " ".join(ID_TO_PHONE.get(i, "") for i in ids if i in ID_TO_PHONE)


def greedy_ctc_decode(logits, decode_fn):
    ids = logits.argmax(dim=-1)
    outputs = []
    for b in range(ids.shape[1]):
        collapsed = _collapse_ctc_ids(ids[:, b].tolist())
        outputs.append(decode_fn(collapsed))
    return outputs


def greedy_char_ctc_decode(logits):
    return greedy_ctc_decode(logits, decode_char_ids)


def greedy_phone_ctc_decode(logits):
    return greedy_ctc_decode(logits, decode_phone_ids)


def beam_ctc_decode(logits, beam_width=5):
    log_probs = F.log_softmax(logits, dim=-1)
    t_steps, batch_size, vocab_size = log_probs.shape
    results = []
    for b in range(batch_size):
        beams = [([], -1, 0.0)]
        for t in range(t_steps):
            new_beams = {}
            frame_lp = log_probs[t, b]
            topk_vals, topk_ids = frame_lp.topk(min(beam_width * 2, vocab_size))
            for seq, last_tok, score in beams:
                for val, tok_id in zip(topk_vals.tolist(), topk_ids.tolist()):
                    new_score = score + val
                    if tok_id == 0:
                        key = tuple(seq)
                        prev = new_beams.get(key)
                        if prev is None or prev[2] < new_score:
                            new_beams[key] = (seq, 0, new_score)
                    elif tok_id == last_tok:
                        key = tuple(seq)
                        prev = new_beams.get(key)
                        if prev is None or prev[2] < new_score:
                            new_beams[key] = (seq, tok_id, new_score)
                    else:
                        new_seq = seq + [tok_id]
                        key = tuple(new_seq)
                        prev = new_beams.get(key)
                        if prev is None or prev[2] < new_score:
                            new_beams[key] = (new_seq, tok_id, new_score)
            beams = sorted(new_beams.values(), key=lambda x: -x[2])[:beam_width]
        best_seq = beams[0][0] if beams else []
        results.append(decode_char_ids(best_seq))
    return results


def _normalize_phone_token(token):
    return re.sub(r"\d", "", token)


def word_to_phone_tokens(word):
    clean = re.sub(r"[^a-z]", "", word.lower())
    if not clean:
        return []
    cached = _PHONE_CACHE.get(clean)
    if cached is not None:
        return list(cached)

    phones = []
    pronouncing = get_pronouncing()
    if pronouncing is not None:
        try:
            candidates = pronouncing.phones_for_word(clean)
        except Exception:
            candidates = []
        if candidates:
            phones = [
                _normalize_phone_token(tok)
                for tok in candidates[0].split()
                if _normalize_phone_token(tok) in PHONE_TO_ID
            ]

    if not phones:
        phones = [f"#{c.upper()}" for c in clean if f"#{c.upper()}" in PHONE_TO_ID]

    _PHONE_CACHE[clean] = tuple(phones)
    return phones


def text_to_phone_ids(text):
    words = re.findall(r"[a-zA-Z']+", text)
    ids = []
    for word in words:
        phones = word_to_phone_tokens(word)
        if not phones:
            continue
        if ids:
            ids.append(PHONE_TO_ID[PHONE_BOUNDARY])
        ids.extend(PHONE_TO_ID[p] for p in phones)
    return ids


def compute_cer(pred, ref):
    import editdistance

    if not ref:
        return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, ref) / len(ref), 2.0)


def compute_wer(pred, ref):
    import editdistance

    pred_w = pred.strip().split()
    ref_w = ref.strip().split()
    if not ref_w:
        return 0.0 if not pred_w else 1.0
    return min(editdistance.eval(pred_w, ref_w) / len(ref_w), 2.0)


def compute_per(pred, ref):
    import editdistance

    pred_p = pred.strip().split()
    ref_p = ref.strip().split()
    if not ref_p:
        return 0.0 if not pred_p else 1.0
    return min(editdistance.eval(pred_p, ref_p) / len(ref_p), 2.0)


def augment_eeg(eeg_np, strength="moderate"):
    eeg = eeg_np.copy()
    t_steps, channels = eeg.shape
    if strength == "moderate":
        jitter, amp_lo, amp_hi, noise_std, ch_drop_p, mask_spans = (
            64,
            0.85,
            1.15,
            0.05,
            0.10,
            2,
        )
    else:
        jitter, amp_lo, amp_hi, noise_std, ch_drop_p, mask_spans = (
            16,
            0.95,
            1.05,
            0.02,
            0.05,
            1,
        )

    shift = np.random.randint(-jitter, jitter + 1)
    if shift != 0:
        eeg = np.roll(eeg, shift, axis=0)
        if shift > 0:
            eeg[:shift] = 0
        else:
            eeg[shift:] = 0

    scales = np.random.uniform(amp_lo, amp_hi, size=(1, channels)).astype(np.float32)
    eeg = eeg * scales
    eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * noise_std

    n_drop = int(channels * ch_drop_p)
    if n_drop > 0:
        drop_idx = np.random.choice(channels, n_drop, replace=False)
        eeg[:, drop_idx] = 0

    for _ in range(mask_spans):
        span_len = np.random.randint(16, 33)
        start = np.random.randint(0, max(1, t_steps - span_len))
        eeg[start : start + span_len] = 0

    return eeg


class MemoryBank:
    def __init__(self, size=512, dim=896):
        self.size = size
        self.bank = torch.zeros(size, dim)
        self.ptr = 0
        self.n = 0

    def enqueue(self, emb):
        emb = emb.detach().cpu().float()
        batch_size = emb.shape[0]
        idx = torch.arange(self.ptr, self.ptr + batch_size) % self.size
        self.bank[idx] = emb
        self.ptr = (self.ptr + batch_size) % self.size
        self.n = min(self.n + batch_size, self.size)

    def get(self, device):
        if self.n < 2:
            return None
        return self.bank[: self.n].to(device)


class MultiScaleRasterizer(nn.Module):
    def __init__(self, size=64, n_electrodes=64):
        super().__init__()
        angles = torch.linspace(0, 2 * np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(
            torch.linspace(-1, 1, size),
            torch.linspace(-1, 1, size),
            indexing="ij",
        )
        px = gx.flatten().unsqueeze(1)
        py = gy.flatten().unsqueeze(1)
        ex = pos_2d[:, 0].unsqueeze(0)
        ey = pos_2d[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex) ** 2 + (py - ey) ** 2)
        w = 1.0 / (dist + 1e-4) ** 2.0
        r = torch.sqrt(px**2 + py**2)
        w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)

    def forward(self, x):
        batch_size, t_steps, _ = x.shape
        x_split = x.reshape(batch_size * t_steps, 4, 64)
        proj = x_split @ self.W.T.to(device=x.device, dtype=x.dtype)
        return proj.reshape(batch_size, t_steps, 4, 64, 64)


class ChannelAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        batch_size, t_steps, channels, height, width = imgs.shape
        out = self.conv(imgs.view(batch_size * t_steps, channels, height, width))
        return out.view(batch_size, t_steps, 3, height, width)


class CrossAttentionBridge(nn.Module):
    def __init__(self, in_ch=768, out_ch=1024, n_latents=128):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, in_ch))
        self.mha = nn.MultiheadAttention(in_ch, num_heads=16, batch_first=True)
        self.ln_k = nn.LayerNorm(in_ch)
        self.ln_q = nn.LayerNorm(in_ch)
        self.proj = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        batch_size = x.shape[0]
        q = self.latents.unsqueeze(0).expand(batch_size, -1, -1)
        x, _ = self.mha(self.ln_q(q), self.ln_k(x), x)
        return self.proj(x)


class EEG_VLM_V40(nn.Module):
    def __init__(self, q_name):
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, VideoMAEConfig, VideoMAEModel

        self.num_frames = 1024
        self.image_size = 64
        self.patch_size = 16
        self.tubelet_size = 4
        self.patch_dim = 3 * self.tubelet_size * self.patch_size * self.patch_size

        v_cfg = VideoMAEConfig(
            num_channels=3,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_frames=self.num_frames,
            tubelet_size=self.tubelet_size,
            hidden_size=768,
        )
        self.rasterizer = MultiScaleRasterizer()
        self.ch_adapt = ChannelAdapter()
        self.video_enc = VideoMAEModel(v_cfg)
        self.bridge = CrossAttentionBridge(768, 1024, n_latents=128)
        self.char_ctc_head = nn.Linear(768, CHAR_VOCAB_SIZE)
        self.phone_ctc_head = nn.Linear(768, PHONE_VOCAB_SIZE)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        self.mae_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, self.patch_dim),
        )

        self.llm = AutoModelForCausalLM.from_pretrained(
            q_name, torch_dtype=torch.bfloat16
        )
        self.llm = get_peft_model(
            self.llm,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=128,
                lora_alpha=256,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            ),
        )
        self.llm_proj = nn.Linear(1024, 896)

    def make_video(self, traj):
        imgs = self.rasterizer(traj)
        imgs = self.ch_adapt(imgs)
        return imgs

    def patchify_video(self, imgs):
        batch_size, t_steps, channels, height, width = imgs.shape
        nt = t_steps // self.tubelet_size
        nh = height // self.patch_size
        nw = width // self.patch_size
        x = imgs.reshape(
            batch_size,
            nt,
            self.tubelet_size,
            channels,
            nh,
            self.patch_size,
            nw,
            self.patch_size,
        )
        x = x.permute(0, 1, 4, 6, 2, 3, 5, 7).contiguous()
        return x.view(batch_size, nt * nh * nw, self.patch_dim)

    def unpatchify_video(self, patches):
        batch_size, _, _ = patches.shape
        nt = self.num_frames // self.tubelet_size
        nh = self.image_size // self.patch_size
        nw = self.image_size // self.patch_size
        x = patches.view(
            batch_size,
            nt,
            nh,
            nw,
            self.tubelet_size,
            3,
            self.patch_size,
            self.patch_size,
        )
        x = x.permute(0, 1, 4, 5, 2, 6, 3, 7).contiguous()
        return x.view(
            batch_size,
            self.num_frames,
            3,
            self.image_size,
            self.image_size,
        )

    def forward_mae(self, traj, mask_ratio=0.9):
        imgs = self.make_video(traj)
        target_patches = self.patchify_video(imgs.float())
        mask = torch.rand(
            target_patches.shape[:2], device=traj.device
        ) < mask_ratio
        if not mask.any():
            mask[:, 0] = True
        masked_patches = target_patches.clone()
        masked_patches[mask] = 0.0
        masked_imgs = self.unpatchify_video(masked_patches).to(dtype=imgs.dtype)
        enc_tokens = self.video_enc(pixel_values=masked_imgs).last_hidden_state
        pred_patches = self.mae_head(enc_tokens.float())
        loss = F.mse_loss(pred_patches[mask], target_patches[mask])
        return loss

    def encode(self, traj):
        imgs = self.make_video(traj)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(dim=2)

    def predict_char_texts(self, traj, use_beam=False):
        v_seq = self.encode(traj)
        char_logits = self.char_ctc_head(v_seq).transpose(0, 1)
        if use_beam:
            return beam_ctc_decode(char_logits, beam_width=5)
        return greedy_char_ctc_decode(char_logits)

    def forward(
        self,
        traj,
        char_ids,
        char_lens,
        phone_ids,
        phone_lens,
        input_ids=None,
        labels=None,
        attn_mask=None,
        neg_bank=None,
        ctc_prompt_ids=None,
        ctc_prompt_mask=None,
        prefix_drop_rate=0.0,
    ):
        batch_size = traj.shape[0]
        device = traj.device

        v_seq = self.encode(traj)

        char_logits = self.char_ctc_head(v_seq).transpose(0, 1)
        phone_logits = self.phone_ctc_head(v_seq).transpose(0, 1)
        input_lens = torch.full((batch_size,), 256, dtype=torch.long, device=device)

        loss_char_ctc = self.ctc_loss_fn(
            F.log_softmax(char_logits, dim=-1), char_ids, input_lens, char_lens
        )
        loss_phone_ctc = self.ctc_loss_fn(
            F.log_softmax(phone_logits, dim=-1), phone_ids, input_lens, phone_lens
        )

        probs = F.softmax(char_logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        ent = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        loss_div = -ent.mean()

        if input_ids is None:
            zero = torch.tensor(0.0, device=device)
            return {
                "loss_lm": zero,
                "loss_char_ctc": loss_char_ctc,
                "loss_phone_ctc": loss_phone_ctc,
                "loss_lock": zero,
                "loss_div": loss_div,
                "tok_mean": None,
            }

        prefix = self.llm_proj(self.bridge(v_seq))
        if prefix_drop_rate > 0 and self.training:
            mask = (
                torch.rand(batch_size, 1, 1, device=device) > prefix_drop_rate
            ).to(prefix.dtype)
            prefix = prefix * mask

        embed_fn = self.llm.get_input_embeddings()
        tok_embs = embed_fn(input_ids)

        prefix_mean = F.normalize(prefix.mean(dim=1), dim=-1)
        tok_mean = F.normalize(tok_embs.mean(dim=1), dim=-1)
        if neg_bank is not None and neg_bank.shape[0] > 1:
            all_text = torch.cat([tok_mean, neg_bank.to(device)], dim=0)
            sims = prefix_mean @ all_text.T / 0.07
            tgt_nce = torch.arange(batch_size, device=device)
        else:
            sims = prefix_mean @ tok_mean.T / 0.07
            tgt_nce = torch.arange(batch_size, device=device)
        loss_lock = F.cross_entropy(sims, tgt_nce)

        ctc_embs = embed_fn(ctc_prompt_ids)
        combined = torch.cat([prefix, ctc_embs, tok_embs], dim=1)

        prefix_len = prefix.shape[1]
        prompt_len = ctc_embs.shape[1]
        masked_labels = labels.clone()
        if attn_mask is not None:
            masked_labels[attn_mask == 0] = -100

        combined_labels = torch.cat(
            [
                torch.full(
                    (batch_size, prefix_len + prompt_len),
                    -100,
                    device=device,
                    dtype=labels.dtype,
                ),
                masked_labels,
            ],
            dim=1,
        )
        combined_attn = torch.cat(
            [
                torch.ones(batch_size, prefix_len, device=device, dtype=torch.long),
                ctc_prompt_mask,
                attn_mask if attn_mask is not None else torch.ones_like(input_ids),
            ],
            dim=1,
        )

        lm_out = self.llm(
            inputs_embeds=combined, labels=combined_labels, attention_mask=combined_attn
        )
        return {
            "loss_lm": lm_out.loss,
            "loss_char_ctc": loss_char_ctc,
            "loss_phone_ctc": loss_phone_ctc,
            "loss_lock": loss_lock,
            "loss_div": loss_div,
            "tok_mean": tok_embs.mean(dim=1).detach(),
        }

    @torch.no_grad()
    def generate(self, traj, tokenizer, max_tokens=64, use_beam=True):
        traj = traj.to(next(self.parameters()).dtype)
        v_seq = self.encode(traj)

        char_logits = self.char_ctc_head(v_seq).transpose(0, 1)
        phone_logits = self.phone_ctc_head(v_seq).transpose(0, 1)
        char_texts = (
            beam_ctc_decode(char_logits, beam_width=5)
            if use_beam
            else greedy_char_ctc_decode(char_logits)
        )
        phone_texts = greedy_phone_ctc_decode(phone_logits)

        prefix = self.llm_proj(self.bridge(v_seq))
        embed_fn = self.llm.get_input_embeddings()

        llm_texts = []
        for b in range(traj.shape[0]):
            hint = char_texts[b]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Recover the exact original sentence from noisy EEG-derived "
                        "text. Be conservative: copy words from the hint when "
                        "possible. Do not add unsupported facts. Return only the "
                        "sentence."
                    ),
                },
                {"role": "user", "content": f'Noisy EEG reading: "{hint}"'},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer.encode(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            ).to(traj.device)
            prompt_embs = embed_fn(prompt_ids)
            combined = torch.cat([prefix[b : b + 1], prompt_embs], dim=1)
            out_ids = self.llm.generate(
                inputs_embeds=combined,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            llm_texts.append(tokenizer.decode(out_ids[0], skip_special_tokens=True))

        return llm_texts, char_texts, phone_texts


def load_pretrained_videomae_encoder(video_enc):
    from transformers import VideoMAEModel as _VMAE

    print("Loading pretrained VideoMAE-base encoder weights...")
    src = _VMAE.from_pretrained("MCG-NJU/videomae-base")
    src_sd = src.state_dict()
    dst_sd = video_enc.state_dict()
    n_copied = 0
    n_skipped = 0
    for key, value in src_sd.items():
        if key in dst_sd and dst_sd[key].shape == value.shape:
            dst_sd[key] = value.clone()
            n_copied += 1
        else:
            n_skipped += 1
    video_enc.load_state_dict(dst_sd, strict=False)
    print(
        f"  ✓ Transferred {n_copied} tensors, skipped {n_skipped} "
        "(patch emb + pos emb expected)"
    )
    del src
    torch.cuda.empty_cache()


def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with open(path, "rb") as f:
            return b"MATLAB" in f.read(128)
    except Exception:
        return False


def download_zuco(base_path, vol):
    zuco_v1 = base_path / "ZuCo_v1"
    zuco_v2 = base_path / "ZuCo_v2"
    for directory in [zuco_v1, zuco_v2]:
        directory.mkdir(parents=True, exist_ok=True)

    def fetch(project_id, target_dir, names):
        res = __import__("subprocess").run(
            ["osf", "-p", project_id, "list"], capture_output=True, text=True
        )
        paths = res.stdout.splitlines()
        for name in names:
            local = target_dir / name
            if verify_mat_file(local):
                continue
            remote = next((p for p in paths if p.strip().endswith(name)), None)
            if remote:
                print(f"  Fetching {name}...")
                __import__("subprocess").run(
                    ["osf", "-p", project_id, "fetch", remote.strip(), str(local)],
                    timeout=900,
                )

    fetch(
        "q3zws",
        zuco_v1,
        ["resultsZAB_SR.mat", "resultsZDM_SR.mat", "resultsZAB_NR.mat", "resultsZDM_NR.mat"],
    )
    fetch("2urht", zuco_v2, ["resultsYAC_NR.mat", "resultsYAG_NR.mat", "resultsYAK_NR.mat"])
    vol.commit()


def load_mat_any(path):
    import h5py
    import scipy.io as sio

    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        sentences = data["sentenceData"]
        if not isinstance(sentences, (np.ndarray, list)):
            sentences = [sentences]
        for sent in sentences:
            if (
                hasattr(sent, "rawData")
                and hasattr(sent, "content")
                and isinstance(sent.rawData, np.ndarray)
            ):
                yield sent.rawData, str(sent.content)
    except Exception:
        try:
            with h5py.File(path, "r") as f:
                if "sentenceData" in f:
                    for key in f["sentenceData"].keys():
                        sent = f["sentenceData"][key]
                        if "rawData" in sent and "content" in sent:
                            yield np.array(sent["rawData"]).T, "hdf5_content_placeholder"
        except Exception:
            pass


def normalize_eeg(eeg, target_ch=64, target_t=1024):
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 2:
        return None
    ch, t_steps = eeg.shape
    if ch > t_steps:
        eeg = eeg.T
        ch, t_steps = eeg.shape
    if ch > target_ch:
        eeg = eeg[:target_ch, :]
    elif ch < target_ch:
        eeg = np.pad(eeg, ((0, target_ch - ch), (0, 0)))
    if t_steps > target_t:
        eeg = eeg[:, :target_t]
    elif t_steps < target_t:
        eeg = np.pad(eeg, ((0, 0), (0, target_t - t_steps)))

    mu = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True).clip(min=1e-6)
    eeg = (eeg - mu) / std

    from scipy.signal import butter, filtfilt

    def band_env(lo, hi, data):
        try:
            b, a = butter(4, [lo / 64.0, hi / 64.0], btype="bandpass")
            env = np.abs(filtfilt(b, a, data, axis=1))
            kernel = np.ones(5) / 5
            smoothed = np.zeros_like(env)
            for ch_idx in range(env.shape[0]):
                smoothed[ch_idx] = np.convolve(env[ch_idx], kernel, mode="same")
            return smoothed
        except Exception:
            return np.zeros_like(data)

    alpha = band_env(8, 13, eeg)
    beta = band_env(13, 30, eeg)
    gamma = band_env(30, 100, eeg)
    combined = np.concatenate([eeg.T, alpha.T, beta.T, gamma.T], axis=1)
    return combined.astype(np.float32)


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, augment="none"):
        self.samples = []
        self.augment = augment
        base = Path(base_path)
        skipped = 0
        for zuco_path in [base / "ZuCo_v1", base / "ZuCo_v2"]:
            for mat in zuco_path.rglob("*.mat"):
                for raw, text in load_mat_any(mat):
                    if not text or "placeholder" in text.lower():
                        skipped += 1
                        continue
                    normed = normalize_eeg(raw)
                    if normed is None:
                        continue
                    char_ids = text_to_char_ids(text)
                    phone_ids = text_to_phone_ids(text)
                    if not char_ids or not phone_ids:
                        skipped += 1
                        continue
                    self.samples.append(
                        {
                            "eeg": normed,
                            "text": text,
                            "char_ids": char_ids,
                            "phone_ids": phone_ids,
                        }
                    )
        print(
            f"[Dataset] {len(self.samples)} samples "
            f"(skipped {skipped}, augment={augment})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        eeg = sample["eeg"]
        if self.augment != "none":
            eeg = augment_eeg(eeg, strength=self.augment)
        return eeg, sample["text"], sample["char_ids"], sample["phone_ids"]


def collate_fn(batch, tokenizer):
    eegs, texts, char_ids_list, phone_ids_list = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs))

    char_lens = torch.tensor([len(c) for c in char_ids_list], dtype=torch.long)
    max_chars = max(int(char_lens.max().item()), 1)
    char_pad = torch.zeros(len(char_ids_list), max_chars, dtype=torch.long)
    for i, char_ids in enumerate(char_ids_list):
        if char_ids:
            char_pad[i, : len(char_ids)] = torch.tensor(char_ids, dtype=torch.long)

    phone_lens = torch.tensor([len(p) for p in phone_ids_list], dtype=torch.long)
    max_phones = max(int(phone_lens.max().item()), 1)
    phone_pad = torch.zeros(len(phone_ids_list), max_phones, dtype=torch.long)
    for i, phone_ids in enumerate(phone_ids_list):
        if phone_ids:
            phone_pad[i, : len(phone_ids)] = torch.tensor(phone_ids, dtype=torch.long)

    enc = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    labels = enc.input_ids.clone()
    return (
        eegs,
        enc.input_ids,
        labels,
        enc.attention_mask,
        char_pad,
        char_lens,
        phone_pad,
        phone_lens,
    )


def build_ctc_hint_batch(model, traj, tokenizer, device):
    with torch.no_grad():
        ctc_texts = model.predict_char_texts(traj, use_beam=False)
    prompts = []
    for hint in ctc_texts:
        messages = [
            {
                "role": "system",
                "content": (
                    "Recover the exact original sentence from noisy EEG-derived "
                    "text. Be conservative: copy words from the hint when possible. "
                    "Do not add unsupported facts. Return only the sentence."
                ),
            },
            {"role": "user", "content": f'Noisy EEG reading: "{hint}"'},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    enc = tokenizer(
        prompts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def build_teacher_hint_batch(texts, tokenizer, device, corruption_rate=0.3):
    prompts = []
    for text in texts:
        words = text.split()
        corrupted = []
        for word in words:
            r = np.random.random()
            if r < corruption_rate:
                pass
            elif r < corruption_rate * 1.3:
                corrupted.append("...")
            else:
                corrupted.append(word)
        hint = " ".join(corrupted) if corrupted else "..."
        messages = [
            {
                "role": "system",
                "content": (
                    "Recover the exact original sentence from noisy EEG-derived "
                    "text. Be conservative: copy words from the hint when possible. "
                    "Do not add unsupported facts. Return only the sentence."
                ),
            },
            {"role": "user", "content": f'Noisy EEG reading: "{hint}"'},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
    enc = tokenizer(
        prompts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device)


ACCUM_STEPS = 8


def _train_mae(model, epochs, train_loader, val_loader, ckpt_prefix, mask_ratio=0.9):
    device = next(model.parameters()).device

    for param in model.parameters():
        param.requires_grad = False
    for module in [model.ch_adapt, model.video_enc, model.mae_head]:
        for param in module.parameters():
            param.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"── EEG-MAE Pretraining ──")
    print(f"  Trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M")
    optimizer = torch.optim.AdamW(trainable, lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        tot_loss = 0.0

        for step, batch in enumerate(train_loader):
            eeg = batch[0].to(device).to(torch.bfloat16)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model.forward_mae(eeg, mask_ratio=mask_ratio) / ACCUM_STEPS
            loss.backward()
            tot_loss += loss.item() * ACCUM_STEPS

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if len(train_loader) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch[0].to(device).to(torch.bfloat16)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    val_losses.append(model.forward_mae(eeg, mask_ratio=mask_ratio).item())
                if len(val_losses) >= 8:
                    break

        train_mean = tot_loss / max(len(train_loader), 1)
        val_mean = float(np.mean(val_losses)) if val_losses else train_mean
        lr = optimizer.param_groups[0]["lr"]
        print(f"MAE Epoch {epoch:3d} | Train:{train_mean:.4f} Val:{val_mean:.4f} | LR:{lr:.2e}")

        torch.save(model.state_dict(), f"{ckpt_prefix}_mae_ep{epoch}.pt")
        if val_mean < best_val:
            best_val = val_mean
            torch.save(model.state_dict(), f"{ckpt_prefix}_mae_best.pt")
            print(f"  ✓ New best MAE {best_val:.4f}")
        ckpt_vol.commit()


def _train_phase(model, phase, epochs, train_loader, val_loader, tokenizer, ckpt_prefix, mem_bank=None):
    device = next(model.parameters()).device

    if phase == 1:
        print("── Phase 1: Char CTC + Phoneme CTC ──")
        for name, param in model.named_parameters():
            param.requires_grad = ("llm" not in name and "llm_proj" not in name)
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Phase 1 trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M")
        optimizer = torch.optim.AdamW(trainable, lr=5e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            steps_per_epoch=len(train_loader),
            epochs=max(epochs, 1),
            pct_start=0.1,
        )
    else:
        print("── Phase 2: Char-Hint Denoising ──")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.bridge.parameters():
            param.requires_grad = True
        for param in model.llm_proj.parameters():
            param.requires_grad = True
        for name, param in model.llm.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Phase 2 trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f}M (encoder frozen)")
        optimizer = torch.optim.AdamW(trainable, lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_char_ctc = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        tot_lm = 0.0
        tot_char_ctc = 0.0
        tot_phone_ctc = 0.0
        tot_lock = 0.0
        tot_div = 0.0

        if phase == 2:
            teacher_ratio = max(0.0, 0.8 - 0.8 * (epoch - 1) / max(epochs * 0.4, 1))
            prefix_drop_rate = min(0.5, 0.1 + 0.4 * (epoch - 1) / max(epochs * 0.5, 1))
            if epoch <= 3 or epoch % 5 == 0:
                print(
                    f"  Curriculum: teacher={teacher_ratio:.2f}, "
                    f"prefix_drop={prefix_drop_rate:.2f}"
                )

        for step, batch in enumerate(train_loader):
            (
                eeg,
                input_ids,
                labels,
                attn_mask,
                char_ids,
                char_lens,
                phone_ids,
                phone_lens,
            ) = batch
            eeg = eeg.to(device).to(torch.bfloat16)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attn_mask = attn_mask.to(device)
            char_ids = char_ids.to(device)
            char_lens = char_lens.to(device)
            phone_ids = phone_ids.to(device)
            phone_lens = phone_lens.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if phase == 1:
                    out = model(eeg, char_ids, char_lens, phone_ids, phone_lens)
                    loss = (
                        10.0 * out["loss_char_ctc"]
                        + 4.0 * out["loss_phone_ctc"]
                        + 0.5 * out["loss_div"]
                    ) / ACCUM_STEPS
                else:
                    use_teacher = np.random.random() < teacher_ratio
                    if use_teacher:
                        batch_texts = [
                            tokenizer.decode(input_ids[b].tolist(), skip_special_tokens=True)
                            for b in range(input_ids.shape[0])
                        ]
                        ctc_prompt_ids, ctc_prompt_mask = build_teacher_hint_batch(
                            batch_texts, tokenizer, device, corruption_rate=0.3
                        )
                    else:
                        ctc_prompt_ids, ctc_prompt_mask = build_ctc_hint_batch(
                            model, eeg, tokenizer, device
                        )

                    neg_bank = mem_bank.get(device) if mem_bank else None
                    out = model(
                        eeg,
                        char_ids,
                        char_lens,
                        phone_ids,
                        phone_lens,
                        input_ids=input_ids,
                        labels=labels,
                        attn_mask=attn_mask,
                        neg_bank=neg_bank,
                        ctc_prompt_ids=ctc_prompt_ids,
                        ctc_prompt_mask=ctc_prompt_mask,
                        prefix_drop_rate=prefix_drop_rate,
                    )
                    current_penalty = 1.0 + 14.0 * min(epoch / 10, 1.0)
                    loss = (
                        out["loss_lm"]
                        + out["loss_char_ctc"]
                        + current_penalty * out["loss_lock"]
                        + 2.0 * out["loss_div"]
                    ) / ACCUM_STEPS

            loss.backward()
            if out["tok_mean"] is not None and mem_bank is not None:
                mem_bank.enqueue(out["tok_mean"])

            tot_lm += out["loss_lm"].item()
            tot_char_ctc += out["loss_char_ctc"].item()
            tot_phone_ctc += out["loss_phone_ctc"].item()
            tot_lock += out["loss_lock"].item()
            tot_div += out["loss_div"].item()

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if phase == 1:
                    scheduler.step()
                optimizer.zero_grad()

        if len(train_loader) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if phase == 1:
                scheduler.step()
            optimizer.zero_grad()

        if phase == 2:
            scheduler.step()

        n_batches = max(len(train_loader), 1)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | LM:{tot_lm / n_batches:.3f} "
            f"CCTC:{tot_char_ctc / n_batches:.3f} "
            f"PCTC:{tot_phone_ctc / n_batches:.3f} "
            f"Lock:{tot_lock / n_batches:.4f} Div:{tot_div / n_batches:.4f} "
            f"| LR:{lr:.2e}"
        )

        if phase == 1 and (tot_char_ctc / n_batches) < best_char_ctc:
            best_char_ctc = tot_char_ctc / n_batches
            torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_best.pt")
            print(f"  ✓ New best char CTC {best_char_ctc:.3f}")

        if epoch % 3 == 0:
            model.eval()
            cer_scores = []
            wer_scores = []
            per_scores = []
            vowel_correct = vowel_total = cons_correct = cons_total = 0
            print(f"── Val Epoch {epoch} ──")
            with torch.no_grad():
                shown = 0
                for batch in val_loader:
                    (
                        eeg_v,
                        input_ids_v,
                        labels_v,
                        attn_v,
                        char_ids_v,
                        char_lens_v,
                        phone_ids_v,
                        phone_lens_v,
                    ) = batch
                    llm_preds, char_preds, phone_preds = model.generate(
                        eeg_v.to(device), tokenizer, use_beam=(phase == 2)
                    )
                    for b in range(eeg_v.shape[0]):
                        ref = tokenizer.decode(
                            input_ids_v[b].tolist(), skip_special_tokens=True
                        )
                        ref_lower = ref.lower()
                        phone_ref = decode_phone_ids(
                            phone_ids_v[b][: phone_lens_v[b]].tolist()
                        )
                        cer = compute_cer(char_preds[b], ref_lower)
                        wer = compute_wer(char_preds[b], ref_lower)
                        per = compute_per(phone_preds[b], phone_ref)
                        cer_scores.append(cer)
                        wer_scores.append(wer)
                        per_scores.append(per)
                        pred_chars = set(char_preds[b])
                        ref_chars = set(ref_lower)
                        for char in ref_chars:
                            if char in VOWELS:
                                vowel_total += 1
                                if char in pred_chars:
                                    vowel_correct += 1
                            elif char in CONSONANTS:
                                cons_total += 1
                                if char in pred_chars:
                                    cons_correct += 1
                        if len(ref.split()) > 3 or shown < 2:
                            print(f"  REF: '{ref[:100]}'")
                            print(
                                f"  CTC: '{char_preds[b][:100]}'  CER={cer:.2f} WER={wer:.2f}"
                            )
                            print(f"  PHN: '{phone_preds[b][:100]}'  PER={per:.2f}")
                            print(f"  GEN: '{llm_preds[b][:100]}'")
                            shown += 1
                        if shown >= 4:
                            break
                    if shown >= 4:
                        break

            v_acc = vowel_correct / max(vowel_total, 1)
            c_acc = cons_correct / max(cons_total, 1)
            if cer_scores:
                print(
                    f"  Mean CER: {np.mean(cer_scores):.3f}  "
                    f"Mean WER: {np.mean(wer_scores):.3f}  "
                    f"Mean PER: {np.mean(per_scores):.3f}"
                )
                print(
                    f"  Vowel recall: {v_acc:.2f} ({vowel_correct}/{vowel_total})  "
                    f"Consonant recall: {c_acc:.2f} ({cons_correct}/{cons_total})"
                )
            torch.save(model.state_dict(), f"{ckpt_prefix}_phase{phase}_ep{epoch}.pt")
            ckpt_vol.commit()

        if phase == 2:
            avg_lm = tot_lm / n_batches
            if not hasattr(_train_phase, "_best_lm") or avg_lm < _train_phase._best_lm:
                _train_phase._best_lm = avg_lm
                torch.save(model.state_dict(), f"{ckpt_prefix}_phase2_best.pt")
                print(f"  ✓ New best LM {avg_lm:.3f}")


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={"/data": data_vol, "/persist": ckpt_vol},
    retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=10.0),
)
def run_pipeline(
    epochs_mae: int = 12,
    epochs_p1: int = 40,
    epochs_p2: int = 40,
    mae_mask_ratio: float = 0.9,
):
    from transformers import AutoTokenizer

    q_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(q_name)

    download_zuco(Path("/data/EEG_Text"), data_vol)

    ds_train = EEGDataset("/data/EEG_Text", augment="none")
    ds_val = EEGDataset("/data/EEG_Text", augment="none")
    indices = list(range(len(ds_train)))
    np.random.seed(42)
    np.random.shuffle(indices)
    n_train = int(0.9 * len(ds_train))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_ds = torch.utils.data.Subset(ds_train, train_idx)
    val_ds = torch.utils.data.Subset(ds_val, val_idx)

    def make_loader(dataset, shuffle):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=4,
            shuffle=shuffle,
            collate_fn=lambda b: collate_fn(b, tokenizer),
        )

    train_loader = make_loader(train_ds, True)
    val_loader = make_loader(val_ds, False)

    model = EEG_VLM_V40(q_name).to(torch.bfloat16).cuda()

    import glob as _glob

    mae_ckpts = sorted(_glob.glob("/persist/v40_mae_ep*.pt"))
    p1_ckpts = sorted(_glob.glob("/persist/v40_phase1_ep*.pt"))
    p2_ckpts = sorted(_glob.glob("/persist/v40_phase2_ep*.pt"))
    best_mae = "/persist/v40_mae_best.pt"
    best_p1 = "/persist/v40_phase1_best.pt"
    final = "/persist/v40_final.pt"

    skip_mae = epochs_mae <= 0
    skip_p1 = epochs_p1 <= 0
    skip_p2 = epochs_p2 <= 0
    epochs_mae_remaining = epochs_mae
    epochs_p1_remaining = epochs_p1
    epochs_p2_remaining = epochs_p2

    if Path(final).exists():
        print("✓ V4.0 already complete.")
        return

    if p2_ckpts:
        last_p2 = p2_ckpts[-1]
        last_epoch = int(last_p2.split("_ep")[-1].replace(".pt", ""))
        model.load_state_dict(torch.load(last_p2, map_location="cuda"), strict=False)
        print(f"✓ Resuming Phase 2 from epoch {last_epoch}")
        skip_mae = True
        skip_p1 = True
        epochs_p2_remaining = max(0, epochs_p2 - last_epoch)
        if epochs_p2_remaining == 0:
            skip_p2 = True
    elif p1_ckpts:
        last_p1 = p1_ckpts[-1]
        last_epoch = int(last_p1.split("_ep")[-1].replace(".pt", ""))
        if Path(best_p1).exists() and last_epoch >= max(epochs_p1 - 3, 1):
            model.load_state_dict(torch.load(best_p1, map_location="cuda"), strict=False)
            print("✓ Phase 1 complete. Starting Phase 2 from best checkpoint.")
            skip_mae = True
            skip_p1 = True
        else:
            model.load_state_dict(torch.load(last_p1, map_location="cuda"), strict=False)
            print(f"✓ Resuming Phase 1 from epoch {last_epoch}")
            skip_mae = True
            epochs_p1_remaining = max(0, epochs_p1 - last_epoch)
            if epochs_p1_remaining == 0:
                skip_p1 = True
    elif mae_ckpts:
        last_mae = mae_ckpts[-1]
        last_epoch = int(last_mae.split("_ep")[-1].replace(".pt", ""))
        if Path(best_mae).exists() and last_epoch >= max(epochs_mae, 1):
            model.load_state_dict(torch.load(best_mae, map_location="cuda"), strict=False)
            print("✓ EEG-MAE complete. Starting Phase 1 from best MAE checkpoint.")
            skip_mae = True
        else:
            model.load_state_dict(torch.load(last_mae, map_location="cuda"), strict=False)
            print(f"✓ Resuming EEG-MAE from epoch {last_epoch}")
            epochs_mae_remaining = max(0, epochs_mae - last_epoch)
            if epochs_mae_remaining == 0:
                skip_mae = True
    elif Path(best_p1).exists():
        model.load_state_dict(torch.load(best_p1, map_location="cuda"), strict=False)
        print("✓ Found Phase 1 best checkpoint. Skipping MAE and Phase 1.")
        skip_mae = True
        skip_p1 = True
    elif Path(best_mae).exists():
        model.load_state_dict(torch.load(best_mae, map_location="cuda"), strict=False)
        print("✓ Found EEG-MAE best checkpoint. Starting Phase 1 from it.")
        skip_mae = True
    else:
        load_pretrained_videomae_encoder(model.video_enc)

    print(f"\n{'=' * 64}")
    print(" V4.0: EEG-MAE + Dual-Head Char/Phoneme CTC + Char-Hint Phase 2")
    print(
        f" EEG-MAE={epochs_mae_remaining}  "
        f"Phase1={epochs_p1_remaining}  "
        f"Phase2={epochs_p2_remaining}"
    )
    print(f"{'=' * 64}\n")

    if not skip_mae and epochs_mae_remaining > 0:
        _train_mae(
            model,
            epochs=epochs_mae_remaining,
            train_loader=train_loader,
            val_loader=val_loader,
            ckpt_prefix="/persist/v40",
            mask_ratio=mae_mask_ratio,
        )
        if Path(best_mae).exists():
            model.load_state_dict(torch.load(best_mae, map_location="cuda"), strict=False)
            print("\n✓ Loaded best EEG-MAE checkpoint for Phase 1\n")

    if not skip_p1 and epochs_p1_remaining > 0:
        _train_phase(
            model,
            phase=1,
            epochs=epochs_p1_remaining,
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            ckpt_prefix="/persist/v40",
        )
        if Path(best_p1).exists():
            model.load_state_dict(torch.load(best_p1, map_location="cuda"), strict=False)
            print("\n✓ Loaded best Phase 1 checkpoint for Phase 2\n")

    if not skip_p2 and epochs_p2_remaining > 0:
        ds_train_p2 = EEGDataset("/data/EEG_Text", augment="light")
        train_ds_p2 = torch.utils.data.Subset(ds_train_p2, train_idx)
        train_loader_p2 = make_loader(train_ds_p2, True)
        mem_bank = MemoryBank(size=512, dim=896)
        _train_phase(
            model,
            phase=2,
            epochs=epochs_p2_remaining,
            train_loader=train_loader_p2,
            val_loader=val_loader,
            tokenizer=tokenizer,
            ckpt_prefix="/persist/v40",
            mem_bank=mem_bank,
        )

    torch.save(model.state_dict(), "/persist/v40_final.pt")
    ckpt_vol.commit()
    print("\n✓ V4.0 complete. Saved /persist/v40_final.pt")


@app.local_entrypoint()
def main(
    mode: str = "pipeline",
    epochs_mae: int = 12,
    epochs_p1: int = 40,
    epochs_p2: int = 40,
    mae_mask_ratio: float = 0.9,
):
    if mode == "pipeline":
        print(
            f"Launching V4.0 pipeline: MAE={epochs_mae}ep, "
            f"Phase1={epochs_p1}ep, Phase2={epochs_p2}ep"
        )
        run_pipeline.remote(
            epochs_mae=epochs_mae,
            epochs_p1=epochs_p1,
            epochs_p2=epochs_p2,
            mae_mask_ratio=mae_mask_ratio,
        )
    elif mode == "mae":
        print(f"Launching V4.0 EEG-MAE only: {epochs_mae}ep")
        run_pipeline.remote(
            epochs_mae=epochs_mae,
            epochs_p1=0,
            epochs_p2=0,
            mae_mask_ratio=mae_mask_ratio,
        )
    elif mode == "p1":
        print(f"Launching V4.0 MAE+Phase1 only: MAE={epochs_mae}ep, Phase1={epochs_p1}ep")
        run_pipeline.remote(
            epochs_mae=epochs_mae,
            epochs_p1=epochs_p1,
            epochs_p2=0,
            mae_mask_ratio=mae_mask_ratio,
        )
    else:
        print("Unknown mode. Use: pipeline / mae / p1")
