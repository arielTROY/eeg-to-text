
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

app = modal.App("eeg-vlm-v62")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v62", create_if_missing=True)
v20_vol  = modal.Volume.from_name("bt-checkpoints-v20", create_if_missing=False)
v50_vol  = modal.Volume.from_name("bt-checkpoints-v50", create_if_missing=False)
v61_vol  = modal.Volume.from_name("bt-checkpoints-v61", create_if_missing=False)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)

def _download_models():
    from transformers import VideoMAEModel, AutoModelForCausalLM, AutoTokenizer
    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0", "peft>=0.12",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "sentence-transformers", "accelerate", "osfclient", "qwen-vl-utils",
        "editdistance", "sentencepiece", "openneuro-py",
    ])
    .run_function(_download_models)
)

# ═══════════════════════════════════════════════════════════════════════════════
# V6.2: Replace the weak V6.0 Phase 2 path with a stronger denoiser branch:
# a non-instruct base LM plus a text-only corrupted-hint warmup before EEG
# conditioning.
#
# What V4.2 proved:
#   [1] V20 warm-start works immediately
#   [2] InnerSpeech lexical support helps restore the good Phase 1 regime
#   [3] Validation still stalls on consonants, not vowels
#
# Big upgrade:
#   [P2-1] Use the non-instruct Qwen2.5 base model instead of the instruct model
#          to reduce generic assistant-style completions.
#   [P2-2] Keep the structured multi-beam hint path from V6.0.
#   [P2-3] Add a text-only denoiser warmup on synthetic CTC-like corruptions
#          before any EEG-conditioned Phase 2 optimization.
#   [P2-4] Load the proven V5.0 Phase 1 backbone after warmup and continue with
#          EEG-conditioned Phase 2.
#
# All useful improvements kept:
#   [B1] V20 warm-start from /v20/v20_phase1_best.pt
#   [B2] V20 preprocessing core (gamma + cross-channel smoothing)
#   [B3] Character-only CTC in Phase 1
#   [B4] LM padding tokens masked as -100
#   [P2] Teacher curriculum (0.8→0 over first 40% of Phase 2)
#   [P2] Prefix dropout ramp (0.1→0.5 over first 50% of Phase 2)
#   [P2] Beam CTC decode at inference
#   [P2] Light EEG augmentation in Phase 2 only
#   [P2] Proper attention mask in LM forward
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Vocabulary ───────────────────────────────────────────────────────────────

CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz0123456789.,!?'\" ()"
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR  = {i: c for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE  = len(CHAR_VOCAB)

VOWELS     = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

NUM_EEG_BANDS    = 6
PHASE1_MIX_EPOCHS = 8

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
    return [beams[0][0] if beams else "" for beams in beam_ctc_decode_nbest(logits, beam_width=beam_width, nbest=1)]


def beam_ctc_decode_nbest(logits, beam_width=6, nbest=3):
    """logits: (T, B, Vocab) → list[list[(text, norm_score)]] using beam search."""
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
            beams = sorted(new_beams.values(), key=lambda x: -x[2])[:max(beam_width, nbest)]
        beams = beams[:nbest]
        if not beams:
            results.append([("", 1.0)])
            continue
        scores = torch.tensor([beam[2] for beam in beams], dtype=torch.float32)
        norm_scores = torch.softmax(scores - scores.max(), dim=0).tolist()
        decoded = []
        for (seq, _, _), score in zip(beams, norm_scores):
            text = ''.join(ID_TO_CHAR.get(i, '') for i in seq)
            decoded.append((text, score))
        results.append(decoded)
    return results


def ctc_like_corrupt_text(text, severity=1.0):
    consonant_choices = tuple(sorted(CONSONANTS))
    out = []
    prev_char = ""
    for ch in text.lower():
        if ch not in CHAR_TO_ID:
            continue
        r = np.random.random()
        if ch in VOWELS:
            if r < 0.10 * severity:
                continue
            out.append(ch)
            if r > 0.96:
                out.append(ch)
        elif ch in CONSONANTS:
            if r < 0.42 * severity:
                continue
            if r < 0.58 * severity:
                sub = consonant_choices[np.random.randint(len(consonant_choices))]
                out.append(sub)
            else:
                out.append(ch)
            if r > 0.96:
                out.append(out[-1])
        elif ch == " ":
            if r < 0.18 * severity:
                continue
            out.append(" ")
            if r > 0.93:
                out.append(" ")
        elif ch in ".,!?":
            if r < 0.55:
                out.append(".")
        else:
            if r < 0.25 * severity:
                continue
            out.append(ch)

        if out and out[-1] not in " ." and prev_char != "." and np.random.random() < 0.025 * severity:
            out.append(".")
        if out:
            prev_char = out[-1]

    hint = ''.join(out)
    hint = ' '.join(hint.split())
    if not hint:
        hint = "..."
    if hint[-1] not in ".!?":
        hint = f"{hint}."
    return hint


def format_structured_prompt(beams):
    lines = [
        "TASK: recover the exact original sentence from noisy EEG candidates.",
        "RULES: prefer copied fragments, stay conservative, do not add facts.",
        "CANDIDATES:",
    ]
    for i, (hint, score) in enumerate(beams, start=1):
        hint = hint.strip() or "..."
        lines.append(f"beam{i} score={score:.2f}: {hint}")
    lines.append("ANSWER:")
    return "\n".join(lines)


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
    def __init__(self, size=64, n_electrodes=64, n_bands=NUM_EEG_BANDS):
        super().__init__()
        self.n_bands = n_bands
        self.n_electrodes = n_electrodes
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
        x_split = x.reshape(B * T, self.n_bands, self.n_electrodes)
        proj = x_split @ self.W.T.to(device=x.device, dtype=x.dtype)
        return proj.reshape(B, T, self.n_bands, 64, 64)


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels=NUM_EEG_BANDS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
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

class EEG_VLM_V62(nn.Module):
    def __init__(self, q_name):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType

        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=1024, tubelet_size=4, hidden_size=768)
        self.rasterizer  = MultiScaleRasterizer(n_bands=NUM_EEG_BANDS)
        self.ch_adapt    = ChannelAdapter(NUM_EEG_BANDS)
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
        """traj: (B, 1024, 384) → v_seq: (B, 256, 768)"""
        imgs  = self.rasterizer(traj)                # (B, 1024, 6, 64, 64)
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
        nbest = beam_ctc_decode_nbest(ctc_logits, beam_width=6, nbest=3) if use_beam else [
            [(text, 1.0)] for text in greedy_ctc_decode(ctc_logits)
        ]
        ctc_texts = [beams[0][0] if beams else "" for beams in nbest]

        prefix   = self.llm_proj(self.bridge(v_seq))
        embed_fn = self.llm.get_input_embeddings()

        llm_texts = []
        for b in range(traj.shape[0]):
            prompt_text = format_structured_prompt(nbest[b])
            prompt_ids  = tokenizer.encode(prompt_text, add_special_tokens=False,
                                           return_tensors="pt").to(traj.device)
            prompt_embs = embed_fn(prompt_ids)
            combined    = torch.cat([prefix[b:b+1], prompt_embs], dim=1)
            combined_attn = torch.ones(
                combined.shape[:2], dtype=torch.long, device=traj.device)
            out_ids     = self.llm.generate(
                inputs_embeds=combined, max_new_tokens=max_tokens,
                do_sample=False, attention_mask=combined_attn,
                pad_token_id=tokenizer.eos_token_id)
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


def load_v20_phase1_warmstart(model, ckpt_path="/v20/v20_phase1_best.pt"):
    """Load the proven V20 stack, expanding the old 4→3 adapter into 6→3."""
    if not Path(ckpt_path).exists():
        print(f"V20 Phase 1 checkpoint not found at {ckpt_path}")
        return False

    print(f"Loading V20 Phase 1 warm-start weights from {ckpt_path}...")
    src_sd = torch.load(ckpt_path, map_location="cpu")
    dst_sd = model.state_dict()
    prefixes = ("rasterizer.", "video_enc.", "ch_adapt.", "ctc_head.", "bridge.", "llm_proj.")
    n_copied = n_skipped = 0
    for k, v in src_sd.items():
        if not k.startswith(prefixes):
            continue
        if k == "ch_adapt.conv.weight" and k in dst_sd:
            dst = dst_sd[k]
            if v.shape[0] == dst.shape[0] and v.shape[1] == 4 and dst.shape[1] == 6:
                expanded = torch.zeros_like(dst)
                expanded[:, :4] = v
                dst_sd[k] = expanded
                n_copied += 1
                continue
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            n_copied += 1
        else:
            n_skipped += 1
    model.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Loaded {n_copied} V20 Phase 1 tensors, skipped {n_skipped}")
    return n_copied > 0


def load_phase1_backbone(model, ckpt_path):
    """Load only the EEG-side Phase 1 backbone after text denoiser warmup."""
    if not Path(ckpt_path).exists():
        print(f"Phase 1 backbone checkpoint not found at {ckpt_path}")
        return False

    print(f"Loading Phase 1 backbone from {ckpt_path}...")
    src_sd = torch.load(ckpt_path, map_location="cpu")
    dst_sd = model.state_dict()
    prefixes = ("rasterizer.", "video_enc.", "ch_adapt.", "ctc_head.", "bridge.", "llm_proj.")
    n_copied = n_skipped = 0
    for k, v in src_sd.items():
        if not k.startswith(prefixes):
            continue
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            n_copied += 1
        else:
            n_skipped += 1
    model.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Loaded {n_copied} backbone tensors, skipped {n_skipped}")
    return n_copied > 0


# ─── Data loading ─────────────────────────────────────────────────────────────

def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000: return False
    try:
        with open(path, 'rb') as f: return b'MATLAB' in f.read(128)
    except: return False

def download_zuco(base_path, vol):
    zuco_v1 = base_path / "ZuCo_v1"
    zuco_v2 = base_path / "ZuCo_v2"
    inner_sp = base_path / "InnerSpeech"
    for d in [zuco_v1, zuco_v2, inner_sp]:
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

    if not any(inner_sp.rglob("*-epo.fif")):
        import openneuro
        print("  Fetching InnerSpeech ds003626 derivatives/sub-01 ...")
        openneuro.download(dataset="ds003626", target_dir=str(inner_sp),
                           include=["derivatives/sub-01/"])
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
            return np.convolve(env.flatten(), np.ones(5)/5, mode='same').reshape(env.shape)
        except:
            return np.zeros_like(data)

    alpha = band_env(8,  13,  eeg)
    beta  = band_env(13, 30,  eeg)
    gamma = band_env(30, 100, eeg)
    delta = np.diff(eeg, axis=1, prepend=eeg[:, :1])
    gamma_delta = np.diff(gamma, axis=1, prepend=gamma[:, :1])

    def rezscore(x):
        mu  = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True).clip(min=1e-6)
        return (x - mu) / std

    delta = rezscore(delta)
    gamma_delta = rezscore(gamma_delta)

    combined = np.concatenate(
        [eeg.T, alpha.T, beta.T, gamma.T, delta.T, gamma_delta.T], axis=1
    )  # (1024, 384)
    return combined.astype(np.float32)


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, augment='none',
                 include_zuco=True, include_innerspeech=False):
        self.samples = []
        self.augment = augment
        p = Path(base_path)
        skipped = 0
        zuco_count = 0
        inner_count = 0

        if include_zuco:
            for zp in [p/"ZuCo_v1", p/"ZuCo_v2"]:
                for mat in zp.rglob("*.mat"):
                    for raw, text in load_mat_any(mat):
                        if not text or "placeholder" in text.lower():
                            skipped += 1
                            continue
                        normed = normalize_eeg(raw)
                        if normed is not None:
                            self.samples.append({'eeg': normed, 'text': text})
                            zuco_count += 1

        if include_innerspeech:
            import mne
            for fif in (p/"InnerSpeech").rglob("*-epo.fif"):
                try:
                    epochs = mne.read_epochs(str(fif), preload=True, verbose=False)
                    epochs.resample(128)
                    data = epochs.get_data()
                    labels = (
                        epochs.metadata['condition'].tolist()
                        if epochs.metadata is not None and 'condition' in epochs.metadata.columns
                        else [f"word_{i}" for i in range(len(data))]
                    )
                    for d, label in zip(data, labels):
                        normed = normalize_eeg(d)
                        if normed is not None:
                            self.samples.append({'eeg': normed, 'text': str(label)})
                            inner_count += 1
                except:
                    continue

        print(
            f"[Dataset] total={len(self.samples)} zuco={zuco_count} inner={inner_count} "
            f"skipped={skipped} augment={augment}"
        )

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
        nbest      = beam_ctc_decode_nbest(ctc_logits, beam_width=6, nbest=3)
    prompts = []
    for beams in nbest:
        prompts.append(format_structured_prompt(beams))
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def build_teacher_hint_batch(texts, tokenizer, device, corruption_rate=0.3):
    prompts = build_teacher_prompt_texts(texts, corruption_rate=corruption_rate)
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def build_teacher_prompt_texts(texts, corruption_rate=0.3):
    prompts = []
    for text in texts:
        severities = [0.70 + corruption_rate, 0.95 + corruption_rate, 1.20 + corruption_rate]
        beams = []
        for i, severity in enumerate(severities, start=1):
            hint = ctc_like_corrupt_text(text, severity=severity)
            score = 1.0 / i
            beams.append((hint, score))
        score_sum = sum(score for _, score in beams)
        norm_beams = [(hint, score / score_sum) for hint, score in beams]
        prompts.append(format_structured_prompt(norm_beams))
    return prompts


class TextWarmupDataset(torch.utils.data.Dataset):
    def __init__(self, texts, repeats=4):
        self.texts = [t for t in texts if t and t.strip()]
        self.repeats = repeats

    def __len__(self):
        return len(self.texts) * self.repeats

    def __getitem__(self, idx):
        return self.texts[idx % len(self.texts)]


def train_text_denoiser_warmup(model, texts, tokenizer, ckpt_path, epochs=3, batch_size=16):
    device = next(model.parameters()).device
    dataset = TextWarmupDataset(texts, repeats=4)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("── Warmup: text-only structured denoiser ──")
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.llm.named_parameters():
        if "lora_" in n:
            p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Warmup trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    embed_fn = model.llm.get_input_embeddings()

    for epoch in range(1, epochs + 1):
        model.train()
        tot_lm = 0.0
        optimizer.zero_grad()

        for step, batch_texts in enumerate(loader):
            prompts = build_teacher_prompt_texts(batch_texts, corruption_rate=0.35)
            prompt_enc = tokenizer(prompts, padding=True, truncation=True,
                                   max_length=128, return_tensors="pt")
            target_enc = tokenizer(list(batch_texts), padding=True, truncation=True,
                                   max_length=256, return_tensors="pt")

            prompt_ids = prompt_enc.input_ids.to(device)
            prompt_attn = prompt_enc.attention_mask.to(device)
            input_ids = target_enc.input_ids.to(device)
            attn_mask = target_enc.attention_mask.to(device)
            labels = input_ids.clone()
            labels[attn_mask == 0] = -100

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prompt_embs = embed_fn(prompt_ids)
                tok_embs = embed_fn(input_ids)
                combined = torch.cat([prompt_embs, tok_embs], dim=1)
                combined_labels = torch.cat([
                    torch.full((input_ids.shape[0], prompt_ids.shape[1]), -100,
                               device=device, dtype=labels.dtype),
                    labels,
                ], dim=1)
                combined_attn = torch.cat([prompt_attn, attn_mask], dim=1)
                lm_out = model.llm(
                    inputs_embeds=combined,
                    labels=combined_labels,
                    attention_mask=combined_attn,
                )
                loss = lm_out.loss / ACCUM_STEPS

            loss.backward()
            tot_lm += lm_out.loss.item()

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if len(loader) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        avg_lm = tot_lm / max(len(loader), 1)
        print(f"Warmup Epoch {epoch:2d} | LM:{avg_lm:.3f} | LR:{optimizer.param_groups[0]['lr']:.2e}")

    torch.save(model.state_dict(), ckpt_path)
    ckpt_vol.commit()
    print(f"  ✓ Saved warmup checkpoint to {ckpt_path}")


# ─── Training loop ────────────────────────────────────────────────────────────

ACCUM_STEPS = 8   # effective batch = 4 × 8 = 32

def make_phase1_optimizer(model, steps_per_epoch, epochs):
    groups = []

    def add_group(params, lr, name):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append({"name": name, "params": params, "lr": lr, "weight_decay": 0.01})

    add_group(model.ch_adapt.parameters(), 3e-4, "adapter")
    add_group(model.ctc_head.parameters(), 1e-4, "ctc")
    add_group(model.video_enc.parameters(), 5e-5, "encoder")
    add_group(model.bridge.parameters(), 5e-5, "bridge")

    seen = {id(p) for g in groups for p in g["params"]}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    add_group(other, 5e-5, "other")

    printable = ", ".join(
        f"{g['name']}:{sum(p.numel() for p in g['params'])/1e6:.1f}M@{g['lr']:.0e}"
        for g in groups
    )
    optimizer = torch.optim.AdamW([
        {"params": g["params"], "lr": g["lr"], "weight_decay": g["weight_decay"]}
        for g in groups
    ])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in groups],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
    )
    return optimizer, scheduler, printable

def _train_phase(model, phase, epochs, train_loader, val_loader,
                 tokenizer, ckpt_prefix, mem_bank=None, epoch_offset=0,
                 best_ctc=float("inf")):
    device = next(model.parameters()).device

    if phase == 1:
        print("── Phase 1: flow-aware warm-start + char CTC curriculum ──")
        for name, p in model.named_parameters():
            p.requires_grad = ("llm" not in name and "llm_proj" not in name)
        optimizer, scheduler, phase1_groups = make_phase1_optimizer(
            model, len(train_loader), epochs)
        trainable = [p for p in model.parameters() if p.requires_grad]
        print(f"  Phase 1 trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
        print(f"  Phase 1 groups: {phase1_groups}")
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

    for local_epoch in range(1, epochs + 1):
        epoch = epoch_offset + local_epoch
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
              f"Lock:{tot_lock/n:.4f} Div:{tot_div/n:.4f} | LR0:{lr:.2e}")

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

    return best_ctc


def run_phase1_curriculum(model, epochs_p1, train_loader_mix, train_loader_pure,
                          val_loader, tokenizer, ckpt_prefix, start_epoch=0):
    best_ctc = float("inf")
    mix_epochs = min(PHASE1_MIX_EPOCHS, epochs_p1)
    current_epoch = start_epoch

    if current_epoch < mix_epochs and train_loader_mix is not None:
        stage_epochs = mix_epochs - current_epoch
        print(f"[Phase1] Stage A: mixed lexical alignment for {stage_epochs} epochs")
        best_ctc = _train_phase(
            model, phase=1, epochs=stage_epochs,
            train_loader=train_loader_mix, val_loader=val_loader,
            tokenizer=tokenizer, ckpt_prefix=ckpt_prefix,
            epoch_offset=current_epoch, best_ctc=best_ctc)
        current_epoch = mix_epochs

    if current_epoch < epochs_p1:
        stage_epochs = epochs_p1 - current_epoch
        print(f"[Phase1] Stage B: pure ZuCo consolidation for {stage_epochs} epochs")
        best_ctc = _train_phase(
            model, phase=1, epochs=stage_epochs,
            train_loader=train_loader_pure, val_loader=val_loader,
            tokenizer=tokenizer, ckpt_prefix=ckpt_prefix,
            epoch_offset=current_epoch, best_ctc=best_ctc)

    return best_ctc


# ─── Modal functions ──────────────────────────────────────────────────────────

@app.function(image=image, gpu="H100", timeout=86400,
              volumes={"/data": data_vol, "/persist": ckpt_vol, "/v20": v20_vol, "/v50": v50_vol, "/v61": v61_vol},
              retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=10.0))
def run_pipeline(text_warm_epochs: int = 3, epochs_p2: int = 40):
    from transformers import AutoTokenizer
    q_name    = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(q_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    download_zuco(Path("/data/EEG_Text"), data_vol)

    ds_zuco = EEGDataset(
        "/data/EEG_Text", augment='none',
        include_zuco=True, include_innerspeech=False)
    n_train  = int(0.9 * len(ds_zuco))
    indices  = list(range(len(ds_zuco)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_zuco = torch.utils.data.Subset(ds_zuco, train_idx)
    val_ds     = torch.utils.data.Subset(ds_zuco, val_idx)

    mkloader = lambda d, shuf: torch.utils.data.DataLoader(
        d, batch_size=4, shuffle=shuf, collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader        = mkloader(val_ds, False)
    train_texts = [ds_zuco.samples[i]["text"] for i in train_idx]

    ds_inner = EEGDataset(
        "/data/EEG_Text", augment='none',
        include_zuco=False, include_innerspeech=True)
    if len(ds_inner) > 0:
        inner_texts = [sample["text"] for sample in ds_inner.samples if sample["text"]]
        train_texts.extend(inner_texts[: min(len(inner_texts), len(train_idx))])
        print(f"[Warmup] Added {min(len(inner_texts), len(train_idx))} InnerSpeech phrases")
    else:
        print("[Warmup] InnerSpeech not found; using ZuCo train texts only")

    model = EEG_VLM_V62(q_name).to(torch.bfloat16).cuda()

    # ── Resume logic ──────────────────────────────────────────────────────────
    import glob as _glob
    p2_ckpts  = sorted(_glob.glob("/persist/v62_phase2_ep*.pt"))
    warm_ready = "/persist/v62_textwarm_ready.pt"
    final     = "/persist/v62_final.pt"
    source_v61_best = "/v61/v61_phase1_best.pt"
    source_v50_best = "/v50/v50_phase1_best.pt"

    if Path(final).exists():
        print("✓ V6.2 already complete.")
        return

    epochs_p2_remaining = epochs_p2

    if p2_ckpts:
        last_p2       = p2_ckpts[-1]
        last_p2_epoch = int(last_p2.split("_ep")[-1].replace(".pt", ""))
        model.load_state_dict(torch.load(last_p2, map_location="cuda"))
        print(f"✓ Resuming Phase 2 from epoch {last_p2_epoch}")
        epochs_p2_remaining = epochs_p2 - last_p2_epoch
    else:
        if Path(warm_ready).exists():
            model.load_state_dict(torch.load(warm_ready, map_location="cuda"))
            print("✓ Loaded warmup-ready V6.2 checkpoint.")
        else:
            train_text_denoiser_warmup(
                model,
                train_texts,
                tokenizer,
                ckpt_path=warm_ready,
                epochs=text_warm_epochs,
                batch_size=16,
            )
            backbone_loaded = False
            for source_ckpt in (source_v61_best, source_v50_best):
                if load_phase1_backbone(model, source_ckpt):
                    backbone_loaded = True
                    break
            if not backbone_loaded:
                if not load_v20_phase1_warmstart(model):
                    load_pretrained_videomae_encoder(model.video_enc)
            torch.save(model.state_dict(), warm_ready)
            ckpt_vol.commit()
            print("✓ Saved warmup + Phase 1 backbone checkpoint.")

    print(f"\n{'='*60}")
    print(f" V6.2: base-LM denoiser warmup + EEG-conditioned Phase 2")
    print(f" {'(RESUMED)' if p2_ckpts or Path(warm_ready).exists() else 'FRESH START'}")
    print(f"{'='*60}\n")

    if epochs_p2_remaining > 0:
        ds_train_p2 = EEGDataset(
            "/data/EEG_Text", augment='light',
            include_zuco=True, include_innerspeech=False)
        train_ds_p2 = torch.utils.data.Subset(ds_train_p2, train_idx)
        train_loader_p2 = mkloader(train_ds_p2, True)

        mem_bank = MemoryBank(size=512, dim=896)
        _train_phase(model, phase=2, epochs=epochs_p2_remaining,
                     train_loader=train_loader_p2, val_loader=val_loader,
                     tokenizer=tokenizer, ckpt_prefix="/persist/v62",
                     mem_bank=mem_bank)

    torch.save(model.state_dict(), "/persist/v62_final.pt")
    ckpt_vol.commit()
    print("\n✓ V6.2 complete. Saved /persist/v62_final.pt")


@app.local_entrypoint()
def main(mode: str = "pipeline", text_warm_epochs: int = 3, epochs_p2: int = 40):
    if mode == "pipeline":
        print(f"Launching V6.2: warmup={text_warm_epochs}ep, Phase2={epochs_p2}ep")
        run_pipeline.remote(text_warm_epochs=text_warm_epochs, epochs_p2=epochs_p2)
    else:
        print(f"Unknown mode '{mode}'. Use: pipeline")
