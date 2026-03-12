
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

app = modal.App("eeg-vlm-v35")

ckpt_vol    = modal.Volume.from_name("bt-checkpoints-v35", create_if_missing=True)
v20_vol     = modal.Volume.from_name("bt-checkpoints-v20", create_if_missing=False)  # Phase 1 source
data_vol    = modal.Volume.from_name("mindvoice-data", create_if_missing=True)

def _download_models():
    from transformers import VideoMAEModel, AutoModelForCausalLM, AutoTokenizer
    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0", "peft>=0.12",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "sentence-transformers", "accelerate", "osfclient", "qwen-vl-utils",
        "editdistance", "sentencepiece",
    ])
    .run_function(_download_models)
)

# ═══════════════════════════════════════════════════════════════════════════════
# V3.5: V3.1b with consonant boost REMOVED from Phase 1
#
# Root cause of V3.1b failure:
#   The consonant-weighted auxiliary loss (3x focal CE on consonants) competed
#   with CTC alignment, making the optimization landscape harder. Phase 1 CTC
#   plateaued at 3.078 vs V2.0's 1.525. The LLM in Phase 2 received hints
#   that were mostly whitespace — too noisy to use — causing mode collapse to
#   a single generic film review template.
#
# Fix: Remove consonant boost entirely. Plain CTC + diversity loss only in
# Phase 1, identical to V2.0 which proved 1.5 CTC is achievable.
#
# All V3.1b improvements kept:
#   [B1] Gamma band (30-100Hz) restored — theta was too smooth for VideoMAE features
#   [B2] Per-channel temporal smoothing (no cross-channel bleed)
#   [B3] LM padding tokens masked as -100
#   [P2] CTC-guided denoising prompts in Phase 2
#   [P2] Teacher curriculum (0.8→0 over first 40% of Phase 2)
#   [P2] Prefix dropout ramp (0.1→0.5 over first 50% of Phase 2)
#   [P2] Chat-template prompts for Qwen2.5-Instruct
#   [P2] Beam CTC decode at inference (beam_width=5)
#   [P2] Data augmentation (moderate in Phase 1, light in Phase 2)
#   [P2] Proper attention mask in LM forward
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Vocabulary ───────────────────────────────────────────────────────────────

CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz0123456789.,!?'\" ()"
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR  = {i: c for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE  = len(CHAR_VOCAB)

VOWELS     = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

def text_to_char_ids(text):
    return [CHAR_TO_ID[c] for c in text.lower() if c in CHAR_TO_ID]

def greedy_ctc_decode(logits):
    """logits: (T, B, Vocab) → list[str]"""
    ids = logits.argmax(dim=-1)
    results = []
    for b in range(ids.shape[1]):
        seq = ids[:, b].tolist()
        out = ""; prev = -1
        for i in seq:
            if i != prev and i != 0:
                out += ID_TO_CHAR.get(i, "")
            prev = i
        results.append(out)
    return results

def beam_ctc_decode(logits, beam_width=5):
    """logits: (T, B, Vocab) → list[str] using beam search."""
    log_probs = F.log_softmax(logits, dim=-1)
    T, B, V = log_probs.shape
    results = []
    for b in range(B):
        beams = [([], -1, 0.0)]
        for t in range(T):
            new_beams = {}
            frame_lp = log_probs[t, b]
            topk_vals, topk_ids = frame_lp.topk(min(beam_width * 2, V))
            for seq, last_tok, score in beams:
                for val, tok_id in zip(topk_vals.tolist(), topk_ids.tolist()):
                    new_score = score + val
                    if tok_id == 0:
                        key = tuple(seq)
                        if key not in new_beams or new_beams[key][2] < new_score:
                            new_beams[key] = (seq, 0, new_score)
                    elif tok_id == last_tok:
                        key = tuple(seq)
                        if key not in new_beams or new_beams[key][2] < new_score:
                            new_beams[key] = (seq, tok_id, new_score)
                    else:
                        new_seq = seq + [tok_id]
                        key = tuple(new_seq)
                        if key not in new_beams or new_beams[key][2] < new_score:
                            new_beams[key] = (new_seq, tok_id, new_score)
            beams = sorted(new_beams.values(), key=lambda x: -x[2])[:beam_width]
        best_seq = beams[0][0] if beams else []
        results.append(''.join(ID_TO_CHAR.get(i, '') for i in best_seq))
    return results


def compute_cer(pred, ref):
    import editdistance
    if not ref: return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, ref) / len(ref), 2.0)

def compute_wer(pred, ref):
    import editdistance
    pred_w = pred.strip().split()
    ref_w  = ref.strip().split()
    if not ref_w: return 0.0 if not pred_w else 1.0
    return min(editdistance.eval(pred_w, ref_w) / len(ref_w), 2.0)


# ─── EEG Augmentation ─────────────────────────────────────────────────────────

def augment_eeg(eeg_np, strength='moderate'):
    eeg = eeg_np.copy()
    T, C = eeg.shape
    if strength == 'moderate':
        jitter, amp_lo, amp_hi, noise_std, ch_drop_p, mask_spans = (64, 0.85, 1.15, 0.05, 0.10, 2)
    else:
        jitter, amp_lo, amp_hi, noise_std, ch_drop_p, mask_spans = (16, 0.95, 1.05, 0.02, 0.05, 1)

    shift = np.random.randint(-jitter, jitter + 1)
    if shift != 0:
        eeg = np.roll(eeg, shift, axis=0)
        if shift > 0: eeg[:shift] = 0
        else: eeg[shift:] = 0

    scales = np.random.uniform(amp_lo, amp_hi, size=(1, C)).astype(np.float32)
    eeg = eeg * scales
    eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * noise_std

    n_drop = int(C * ch_drop_p)
    if n_drop > 0:
        drop_idx = np.random.choice(C, n_drop, replace=False)
        eeg[:, drop_idx] = 0

    for _ in range(mask_spans):
        span_len = np.random.randint(16, 33)
        start = np.random.randint(0, max(1, T - span_len))
        eeg[start:start+span_len] = 0

    return eeg


# ─── Memory Bank ──────────────────────────────────────────────────────────────

class MemoryBank:
    def __init__(self, size=512, dim=896):
        self.size = size
        self.bank = torch.zeros(size, dim)
        self.ptr  = 0
        self.n    = 0

    def enqueue(self, emb):
        emb = emb.detach().cpu().float()
        B = emb.shape[0]
        idx = torch.arange(self.ptr, self.ptr + B) % self.size
        self.bank[idx] = emb
        self.ptr = (self.ptr + B) % self.size
        self.n = min(self.n + B, self.size)

    def get(self, device):
        if self.n < 2: return None
        return self.bank[:self.n].to(device)


# ─── Modules ──────────────────────────────────────────────────────────────────

class MultiScaleRasterizer(nn.Module):
    def __init__(self, size=64, n_electrodes=64):
        super().__init__()
        angles = torch.linspace(0, 2 * np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
        px = gx.flatten().unsqueeze(1); py = gy.flatten().unsqueeze(1)
        ex = pos_2d[:, 0].unsqueeze(0); ey = pos_2d[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex)**2 + (py - ey)**2)
        w = 1.0 / (dist + 1e-4) ** 2.0
        r = torch.sqrt(px**2 + py**2)
        w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)

    def forward(self, x):
        B, T, Ch = x.shape
        x_split = x.reshape(B * T, 4, 64)
        proj = x_split @ self.W.T.to(device=x.device, dtype=x.dtype)
        return proj.reshape(B, T, 4, 64, 64)


class ChannelAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        out = self.conv(imgs.view(B * T, C, H, W))
        return out.view(B, T, 3, H, W)


class CrossAttentionBridge(nn.Module):
    def __init__(self, in_ch=768, out_ch=1024, n_latents=128):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, in_ch))
        self.mha     = nn.MultiheadAttention(in_ch, num_heads=16, batch_first=True)
        self.ln_k    = nn.LayerNorm(in_ch)
        self.ln_q    = nn.LayerNorm(in_ch)
        self.proj    = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        B = x.shape[0]
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        x, _ = self.mha(self.ln_q(q), self.ln_k(x), x)
        return self.proj(x)


# ─── Model ────────────────────────────────────────────────────────────────────

class EEG_VLM_V35(nn.Module):
    def __init__(self, q_name):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType

        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=1024, tubelet_size=4, hidden_size=768)
        self.rasterizer  = MultiScaleRasterizer()
        self.ch_adapt    = ChannelAdapter()
        self.video_enc   = VideoMAEModel(v_cfg)
        self.bridge      = CrossAttentionBridge(768, 1024, n_latents=128)
        self.ctc_head    = nn.Linear(768, VOCAB_SIZE)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

        self.llm = AutoModelForCausalLM.from_pretrained(q_name, torch_dtype=torch.bfloat16)
        self.llm = get_peft_model(self.llm, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=256,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]))
        self.llm_proj = nn.Linear(1024, 896)

    def encode(self, traj):
        """traj: (B, 1024, 256) → v_seq: (B, 256, 768)"""
        imgs  = self.rasterizer(traj)                # (B, 1024, 4, 64, 64)
        imgs  = self.ch_adapt(imgs)                  # (B, 1024, 3, 64, 64)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(dim=2)  # (B, 256, 768)

    def forward(self, traj, char_ids, char_lens,
                input_ids=None, labels=None, attn_mask=None,
                neg_bank=None, ctc_prompt_ids=None, ctc_prompt_mask=None,
                prefix_drop_rate=0.0):
        """
        Returns: (loss_lm, loss_ctc, loss_lock, loss_div, tok_mean_or_None)
        Phase 1: call with input_ids=None → skips LLM, returns zeros for lm/lock
        Phase 2: pass all args
        """
        B      = traj.shape[0]
        device = traj.device

        v_seq = self.encode(traj)  # (B, 256, 768)

        # ── CTC on 256 real VideoMAE steps ────────────────────────────────────
        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)  # (256, B, V)
        input_lens = torch.full((B,), 256, dtype=torch.long, device=device)
        loss_ctc   = self.ctc_loss_fn(
            F.log_softmax(ctc_logits, dim=-1), char_ids, input_lens, char_lens)

        # ── Diversity loss (blank-collapse prevention) ─────────────────────────
        probs      = F.softmax(ctc_logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        ent        = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        loss_div   = -ent.mean()

        # ── Phase 1: skip LLM entirely ─────────────────────────────────────────
        if input_ids is None:
            zero = torch.tensor(0.0, device=device)
            return zero, loss_ctc, zero, loss_div, None

        # ── Phase 2: LLM path ──────────────────────────────────────────────────
        prefix = self.llm_proj(self.bridge(v_seq))  # (B, 128, 896)

        # Prefix dropout: forces LLM to use CTC hint when prefix is zeroed
        if prefix_drop_rate > 0 and self.training:
            mask   = (torch.rand(B, 1, 1, device=device) > prefix_drop_rate).to(prefix.dtype)
            prefix = prefix * mask

        embed_fn = self.llm.get_input_embeddings()
        tok_embs = embed_fn(input_ids)

        # InfoNCE with memory bank
        prefix_mean = F.normalize(prefix.mean(dim=1), dim=-1)
        tok_mean    = F.normalize(tok_embs.mean(dim=1), dim=-1)
        if neg_bank is not None and neg_bank.shape[0] > 1:
            all_text  = torch.cat([tok_mean, neg_bank.to(device)], dim=0)
            sims      = prefix_mean @ all_text.T / 0.07
            tgt_nce   = torch.arange(B, device=device)
        else:
            sims    = prefix_mean @ tok_mean.T / 0.07
            tgt_nce = torch.arange(B, device=device)
        loss_lock = F.cross_entropy(sims, tgt_nce)

        # CTC-guided denoising: prefix + ctc_hint + target_text
        ctc_embs   = embed_fn(ctc_prompt_ids)
        combined   = torch.cat([prefix, ctc_embs, tok_embs], dim=1)

        prefix_len = prefix.shape[1]
        prompt_len = ctc_embs.shape[1]

        masked_labels = labels.clone()
        if attn_mask is not None:
            masked_labels[attn_mask == 0] = -100

        combined_labels = torch.cat([
            torch.full((B, prefix_len + prompt_len), -100, device=device, dtype=labels.dtype),
            masked_labels], dim=1)
        combined_attn = torch.cat([
            torch.ones(B, prefix_len, device=device, dtype=torch.long),
            ctc_prompt_mask,
            attn_mask if attn_mask is not None else torch.ones_like(input_ids)], dim=1)

        lm_out = self.llm(inputs_embeds=combined, labels=combined_labels,
                          attention_mask=combined_attn)

        tok_mean_raw = tok_embs.mean(dim=1).detach()
        return lm_out.loss, loss_ctc, loss_lock, loss_div, tok_mean_raw

    @torch.no_grad()
    def generate(self, traj, tokenizer, max_tokens=64, use_beam=True):
        traj    = traj.to(next(self.parameters()).dtype)
        v_seq   = self.encode(traj)

        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)
        ctc_texts  = beam_ctc_decode(ctc_logits, beam_width=5) if use_beam else greedy_ctc_decode(ctc_logits)

        prefix   = self.llm_proj(self.bridge(v_seq))
        embed_fn = self.llm.get_input_embeddings()

        llm_texts = []
        for b in range(traj.shape[0]):
            hint = ctc_texts[b]
            messages = [
                {"role": "system", "content": "Recover the exact original sentence from noisy EEG-derived text. Be conservative: copy words from the hint when possible. Do not add unsupported facts. Return only the sentence."},
                {"role": "user", "content": f'Noisy EEG reading: "{hint}"'},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompt_ids  = tokenizer.encode(prompt_text, add_special_tokens=False,
                                           return_tensors="pt").to(traj.device)
            prompt_embs = embed_fn(prompt_ids)
            combined    = torch.cat([prefix[b:b+1], prompt_embs], dim=1)
            out_ids     = self.llm.generate(
                inputs_embeds=combined, max_new_tokens=max_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id)
            llm_texts.append(tokenizer.decode(out_ids[0], skip_special_tokens=True))

        return llm_texts, ctc_texts


# ─── Pretrained weight transfer ───────────────────────────────────────────────

def load_pretrained_videomae_encoder(video_enc):
    from transformers import VideoMAEModel as _VMAE
    print("Loading pretrained VideoMAE-base encoder weights...")
    src    = _VMAE.from_pretrained("MCG-NJU/videomae-base")
    src_sd = src.state_dict()
    dst_sd = video_enc.state_dict()
    n_copied = n_skipped = 0
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone(); n_copied += 1
        else:
            n_skipped += 1
    video_enc.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Transferred {n_copied} tensors, skipped {n_skipped} "
          f"(patch emb + pos emb — re-init from scratch)")
    del src; torch.cuda.empty_cache()


# ─── Data loading ─────────────────────────────────────────────────────────────

def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000: return False
    try:
        with open(path, 'rb') as f: return b'MATLAB' in f.read(128)
    except: return False

def download_zuco(base_path, vol):
    zuco_v1 = base_path / "ZuCo_v1"
    zuco_v2 = base_path / "ZuCo_v2"
    for d in [zuco_v1, zuco_v2]:
        d.mkdir(parents=True, exist_ok=True)

    def fetch(project_id, target_dir, names):
        res   = __import__('subprocess').run(
            ["osf", "-p", project_id, "list"], capture_output=True, text=True)
        paths = res.stdout.splitlines()
        for name in names:
            local = target_dir / name
            if not verify_mat_file(local):
                remote = next((p for p in paths if p.strip().endswith(name)), None)
                if remote:
                    print(f"  Fetching {name}...")
                    __import__('subprocess').run(
                        ["osf", "-p", project_id, "fetch", remote.strip(), str(local)],
                        timeout=900)

    fetch("q3zws", zuco_v1, ["resultsZAB_SR.mat","resultsZDM_SR.mat",
                               "resultsZAB_NR.mat","resultsZDM_NR.mat"])
    fetch("2urht", zuco_v2, ["resultsYAC_NR.mat","resultsYAG_NR.mat","resultsYAK_NR.mat"])
    vol.commit()

def load_mat_any(path):
    import scipy.io as sio, h5py
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        sentences = data['sentenceData']
        if not isinstance(sentences, (np.ndarray, list)):
            sentences = [sentences]
        for s in sentences:
            if hasattr(s,'rawData') and hasattr(s,'content') and isinstance(s.rawData, np.ndarray):
                yield s.rawData, str(s.content)
    except:
        try:
            with h5py.File(path, 'r') as f:
                if 'sentenceData' in f:
                    for key in f['sentenceData'].keys():
                        s = f['sentenceData'][key]
                        if 'rawData' in s and 'content' in s:
                            yield np.array(s['rawData']).T, "hdf5_content_placeholder"
        except: pass

def normalize_eeg(eeg, target_ch=64, target_t=1024):
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 2: return None
    ch, t = eeg.shape
    if ch > t: eeg = eeg.T; ch, t = eeg.shape
    if ch > target_ch:   eeg = eeg[:target_ch, :]
    elif ch < target_ch: eeg = np.pad(eeg, ((0, target_ch - ch), (0, 0)))
    if t > target_t:   eeg = eeg[:, :target_t]
    elif t < target_t: eeg = np.pad(eeg, ((0, 0), (0, target_t - t)))

    mu  = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True).clip(min=1e-6)
    eeg = (eeg - mu) / std

    from scipy.signal import butter, filtfilt

    def band_env(lo, hi, data):
        try:
            b, a = butter(4, [lo/64.0, hi/64.0], btype='bandpass')
            env  = np.abs(filtfilt(b, a, data, axis=1))
            # V2.0 smoothing — matches what the Phase 1 encoder was trained on
            return np.convolve(env.flatten(), np.ones(5)/5, mode='same').reshape(env.shape)
        except:
            return np.zeros_like(data)

    alpha = band_env(8,  13,  eeg)
    beta  = band_env(13, 30,  eeg)
    gamma = band_env(30, 100, eeg)   # reverted: gamma gives VideoMAE richer spatial texture

    combined = np.concatenate([eeg.T, alpha.T, beta.T, gamma.T], axis=1)  # (1024, 256)
    return combined.astype(np.float32)


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, augment='none'):
        self.samples = []
        self.augment = augment
        p = Path(base_path)
        skipped = 0
        for zp in [p/"ZuCo_v1", p/"ZuCo_v2"]:
            for mat in zp.rglob("*.mat"):
                for raw, text in load_mat_any(mat):
                    if not text or "placeholder" in text.lower():
                        skipped += 1; continue
                    normed = normalize_eeg(raw)
                    if normed is not None:
                        self.samples.append({'eeg': normed, 'text': text})
        print(f"[Dataset] {len(self.samples)} samples (skipped {skipped}, augment={augment})")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        eeg = s['eeg']
        if self.augment != 'none':
            eeg = augment_eeg(eeg, strength=self.augment)
        char_ids = text_to_char_ids(s['text'])
        return eeg, s['text'], char_ids


def collate_fn(batch, tokenizer):
    eegs, texts, char_ids_list = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs))
    char_lens = torch.tensor([len(c) for c in char_ids_list], dtype=torch.long)
    max_chars = max(char_lens) if max(char_lens) > 0 else 1
    char_pad  = torch.zeros(len(char_ids_list), max_chars, dtype=torch.long)
    for i, c in enumerate(char_ids_list):
        if len(c) > 0:
            char_pad[i, :len(c)] = torch.tensor(c)
    enc    = tokenizer(list(texts), padding=True, truncation=True,
                       max_length=256, return_tensors="pt")
    labels = enc.input_ids.clone()
    return eegs, enc.input_ids, labels, enc.attention_mask, char_pad, char_lens


# ─── Hint builders ────────────────────────────────────────────────────────────

def build_ctc_hint_batch(model, traj, tokenizer, device):
    with torch.no_grad():
        v_seq      = model.encode(traj)
        ctc_logits = model.ctc_head(v_seq).transpose(0, 1)
        ctc_texts  = greedy_ctc_decode(ctc_logits)
    prompts = []
    for hint in ctc_texts:
        messages = [
            {"role": "system", "content": "Recover the exact original sentence from noisy EEG-derived text. Be conservative: copy words from the hint when possible. Do not add unsupported facts. Return only the sentence."},
            {"role": "user", "content": f'Noisy EEG reading: "{hint}"'},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def build_teacher_hint_batch(texts, tokenizer, device, corruption_rate=0.3):
    prompts = []
    for text in texts:
        words     = text.split()
        corrupted = []
        for w in words:
            r = np.random.random()
            if r < corruption_rate:                pass
            elif r < corruption_rate * 1.3:        corrupted.append('...')
            else:                                  corrupted.append(w)
        hint = ' '.join(corrupted) if corrupted else '...'
        messages = [
            {"role": "system", "content": "Recover the exact original sentence from noisy EEG-derived text. Be conservative: copy words from the hint when possible. Do not add unsupported facts. Return only the sentence."},
            {"role": "user", "content": f'Noisy EEG reading: "{hint}"'},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)


# ─── Training loop ────────────────────────────────────────────────────────────

ACCUM_STEPS = 8   # effective batch = 4 × 8 = 32

def _train_phase(model, phase, epochs, train_loader, val_loader,
                 tokenizer, ckpt_prefix, mem_bank=None):
    device = next(model.parameters()).device

    if phase == 1:
        print("── Phase 1: CTC Only (no consonant boost) ──")
        for name, p in model.named_parameters():
            p.requires_grad = ("llm" not in name and "llm_proj" not in name)
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Phase 1 trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
        optimizer = torch.optim.AdamW(trainable, lr=5e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=5e-4,
            steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.1)
    else:
        print("── Phase 2: CTC-Guided Denoising + Curriculum ──")
        for p in model.parameters(): p.requires_grad = False
        for p in model.bridge.parameters():      p.requires_grad = True
        for p in model.llm_proj.parameters():    p.requires_grad = True
        for n, p in model.llm.named_parameters():
            if "lora_" in n: p.requires_grad = True
        trainable_p2 = [p for p in model.parameters() if p.requires_grad]
        print(f"  Phase 2 trainable params: {sum(p.numel() for p in trainable_p2)/1e6:.1f}M "
              f"(encoder frozen)")
        optimizer = torch.optim.AdamW(trainable_p2, lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_ctc = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        tot_lm = tot_ctc = tot_lock = tot_div = 0.0
        optimizer.zero_grad()

        if phase == 2:
            teacher_ratio    = max(0.0, 0.8 - 0.8 * (epoch - 1) / max(epochs * 0.4, 1))
            prefix_drop_rate = min(0.5, 0.1 + 0.4 * (epoch - 1) / max(epochs * 0.5, 1))
            if epoch <= 3 or epoch % 5 == 0:
                print(f"  Curriculum: teacher={teacher_ratio:.2f}, "
                      f"prefix_drop={prefix_drop_rate:.2f}")

        for step, batch in enumerate(train_loader):
            eeg, input_ids, labels, attn_mask, c_ids, c_lens = batch
            eeg, input_ids, labels, attn_mask, c_ids, c_lens = [
                x.to(device) for x in (eeg, input_ids, labels, attn_mask, c_ids, c_lens)]
            eeg = eeg.to(torch.bfloat16)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if phase == 1:
                    # Pure CTC + diversity — NO consonant boost
                    _, loss_ctc, _, loss_div, _ = model(eeg, c_ids, c_lens)
                    loss     = (10.0 * loss_ctc + 0.5 * loss_div) / ACCUM_STEPS
                    loss_lm  = loss_lock = torch.tensor(0.0)
                    tok_mean = None
                else:
                    use_teacher = (np.random.random() < teacher_ratio)
                    if use_teacher:
                        batch_texts = [tokenizer.decode(input_ids[b], skip_special_tokens=True)
                                       for b in range(input_ids.shape[0])]
                        ctc_prompt_ids, ctc_prompt_mask = build_teacher_hint_batch(
                            batch_texts, tokenizer, device, corruption_rate=0.3)
                    else:
                        ctc_prompt_ids, ctc_prompt_mask = build_ctc_hint_batch(
                            model, eeg, tokenizer, device)

                    neg_bank = mem_bank.get(device) if mem_bank else None
                    loss_lm, loss_ctc, loss_lock, loss_div, tok_mean = model(
                        eeg, c_ids, c_lens, input_ids, labels,
                        attn_mask=attn_mask, neg_bank=neg_bank,
                        ctc_prompt_ids=ctc_prompt_ids,
                        ctc_prompt_mask=ctc_prompt_mask,
                        prefix_drop_rate=prefix_drop_rate)
                    current_penalty = 1.0 + 14.0 * min(epoch / 10, 1.0)
                    loss = (loss_lm + 1.0 * loss_ctc
                            + current_penalty * loss_lock
                            + 2.0 * loss_div) / ACCUM_STEPS

            loss.backward()
            if tok_mean is not None and mem_bank is not None:
                mem_bank.enqueue(tok_mean)

            tot_lm   += loss_lm.item()
            tot_ctc  += loss_ctc.item()
            tot_lock += loss_lock.item()
            tot_div  += loss_div.item()

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if phase == 1: scheduler.step()
                optimizer.zero_grad()

        if len(train_loader) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if phase == 1: scheduler.step()
            optimizer.zero_grad()

        if phase == 2: scheduler.step()

        n  = len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | LM:{tot_lm/n:.3f} CTC:{tot_ctc/n:.3f} "
              f"Lock:{tot_lock/n:.4f} Div:{tot_div/n:.4f} | LR:{lr:.2e}")

        if phase == 1 and (tot_ctc / n) < best_ctc:
            best_ctc = tot_ctc / n
            torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_best.pt")
            print(f"  ✓ New best CTC {best_ctc:.3f}")

        # Validation every 3 epochs
        if epoch % 3 == 0:
            model.eval()
            cer_scores, wer_scores = [], []
            vowel_correct = vowel_total = cons_correct = cons_total = 0
            print(f"── Val Epoch {epoch} ──")
            with torch.no_grad():
                shown = 0
                for batch in val_loader:
                    eeg_v, input_ids_v, labels_v, attn_v, c_ids_v, c_lens_v = batch
                    llm_preds, ctc_preds = model.generate(
                        eeg_v.to(device), tokenizer, use_beam=(phase == 2))
                    for b in range(eeg_v.shape[0]):
                        ref       = tokenizer.decode(input_ids_v[b], skip_special_tokens=True)
                        ref_lower = ref.lower()
                        cer = compute_cer(ctc_preds[b], ref_lower)
                        wer = compute_wer(ctc_preds[b], ref_lower)
                        cer_scores.append(cer)
                        wer_scores.append(wer)
                        pred_chars = set(ctc_preds[b])
                        ref_chars  = set(ref_lower)
                        for c in ref_chars:
                            if c in VOWELS:
                                vowel_total += 1
                                if c in pred_chars: vowel_correct += 1
                            elif c in CONSONANTS:
                                cons_total += 1
                                if c in pred_chars: cons_correct += 1
                        if len(ref.split()) > 3 or shown < 2:
                            print(f"  REF: '{ref[:100]}'")
                            print(f"  CTC: '{ctc_preds[b][:100]}'  CER={cer:.2f} WER={wer:.2f}")
                            print(f"  GEN: '{llm_preds[b][:100]}'")
                            shown += 1
                        if shown >= 4: break
                    if shown >= 4: break
            v_acc = vowel_correct / max(vowel_total, 1)
            c_acc = cons_correct  / max(cons_total,  1)
            if cer_scores:
                print(f"  Mean CER: {np.mean(cer_scores):.3f}  Mean WER: {np.mean(wer_scores):.3f}")
                print(f"  Vowel recall: {v_acc:.2f} ({vowel_correct}/{vowel_total})  "
                      f"Consonant recall: {c_acc:.2f} ({cons_correct}/{cons_total})")
            torch.save(model.state_dict(), f"{ckpt_prefix}_phase{phase}_ep{epoch}.pt")
            ckpt_vol.commit()

        if phase == 2:
            avg_lm = tot_lm / n
            if not hasattr(_train_phase, '_best_lm') or avg_lm < _train_phase._best_lm:
                _train_phase._best_lm = avg_lm
                torch.save(model.state_dict(), f"{ckpt_prefix}_phase2_best.pt")
                print(f"  ✓ New best LM {avg_lm:.3f}")


# ─── Modal functions ──────────────────────────────────────────────────────────

@app.function(image=image, gpu="H100", timeout=86400,
              volumes={"/data": data_vol, "/persist": ckpt_vol, "/v20": v20_vol},
              retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=10.0))
def run_pipeline(epochs_p2: int = 40):
    """Phase 2 only — loads V2.0 Phase 1 best checkpoint (CTC=1.525)."""
    from transformers import AutoTokenizer
    q_name    = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(q_name)

    download_zuco(Path("/data/EEG_Text"), data_vol)

    ds_val = EEGDataset("/data/EEG_Text", augment='none')
    indices = list(range(len(ds_val)))
    np.random.seed(42); np.random.shuffle(indices)
    n_train   = int(0.9 * len(ds_val))
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]
    val_ds    = torch.utils.data.Subset(ds_val, val_idx)

    mkloader = lambda d, shuf: torch.utils.data.DataLoader(
        d, batch_size=4, shuffle=shuf, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = mkloader(val_ds, False)

    model = EEG_VLM_V35(q_name).to(torch.bfloat16).cuda()

    # ── Resume: Phase 2 checkpoint takes priority ─────────────────────────────
    import glob as _glob
    p2_ckpts = sorted(_glob.glob("/persist/v35_phase2_ep*.pt"))
    final    = "/persist/v35_final.pt"

    if Path(final).exists():
        print("✓ V3.5 already complete.")
        return

    skip_p2 = False
    epochs_p2_remaining = epochs_p2

    if p2_ckpts:
        last_p2       = p2_ckpts[-1]
        last_p2_epoch = int(last_p2.split("_ep")[-1].replace(".pt", ""))
        model.load_state_dict(torch.load(last_p2, map_location="cuda"))
        print(f"✓ Resuming Phase 2 from epoch {last_p2_epoch}")
        epochs_p2_remaining = epochs_p2 - last_p2_epoch
        if epochs_p2_remaining <= 0: skip_p2 = True
    else:
        # Load V2.0 Phase 1 best checkpoint (CTC=1.525) — skip retraining Phase 1
        v20_ckpt = "/v20/v20_phase1_best.pt"
        model.load_state_dict(torch.load(v20_ckpt, map_location="cuda"), strict=False)
        print(f"✓ Loaded V2.0 Phase 1 best checkpoint from {v20_ckpt} (CTC=1.525)")

    print(f"\n{'='*60}")
    print(f" V3.5: V2.0 Phase 1 (CTC=1.525) + V3.1b Phase 2 improvements")
    print(f" Phase 2: {epochs_p2_remaining} epochs remaining")
    print(f"{'='*60}\n")

    if not skip_p2:
        ds_p2       = EEGDataset("/data/EEG_Text", augment='light')
        train_ds_p2 = torch.utils.data.Subset(ds_p2, train_idx)
        train_loader_p2 = mkloader(train_ds_p2, True)

        mem_bank = MemoryBank(size=512, dim=896)
        _train_phase(model, phase=2, epochs=epochs_p2_remaining,
                     train_loader=train_loader_p2, val_loader=val_loader,
                     tokenizer=tokenizer, ckpt_prefix="/persist/v35",
                     mem_bank=mem_bank)

    torch.save(model.state_dict(), "/persist/v35_final.pt")
    ckpt_vol.commit()
    print("\n✓ V3.5 complete. Saved /persist/v35_final.pt")


@app.local_entrypoint()
def main(epochs_p2: int = 40):
    if True:
        print(f"Launching V3.5 Phase 2 only: {epochs_p2}ep (using V2.0 Phase 1 CTC=1.525)")
        run_pipeline.remote(epochs_p2=epochs_p2)
    else:
        print(f"Unknown mode '{mode}'. Use: pipeline")
