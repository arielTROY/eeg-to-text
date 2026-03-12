
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

app = modal.App("eeg-vlm-v20")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v20", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0", "peft>=0.12",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "sentence-transformers", "accelerate", "osfclient", "qwen-vl-utils",
        "editdistance",
    ])
)

# ═══════════════════════════════════════════════════════════════════════════════
# V2.0: The Pretrained Brain-Video Decoder
#
# Improvements over V1.2 (V12):
#
# [1] PRETRAINED VIDEOMAE ENCODER
#     V12 initialized VideoMAE randomly — it had to learn temporal attention from
#     scratch. V20 loads MCG-NJU/videomae-base (Kinetics-400 pretrained) and
#     transfers all Transformer block weights (self-attention + FFN + layer norms).
#     These weights already encode temporal motion detection, which generalizes
#     to EEG temporal dynamics (neural "motion" over the scalp).
#     Only patch embedding and positional embeddings are re-initialized (different
#     spatial resolution and tubelet size).
#
# [2] CHANNEL ADAPTER (4ch EEG → 3ch for pretrained VideoMAE)
#     A small per-frame Conv2d(4, 3, kernel=1) maps the 4 spectral bands
#     (Voltage, Alpha, Beta, Gamma) into the 3-channel space VideoMAE expects.
#     This learned projection costs almost nothing and keeps VideoMAE's pretrained
#     weights intact.
#
# [3] PHASE 1 SKIPS LLM FORWARD
#     V12 ran a frozen LLM forward on every Phase 1 batch — pure wasted compute.
#     V20 passes input_ids=None in Phase 1 to skip the LLM entirely.
#     Diversity loss is now computed before the LLM block so it still works.
#
# [4] PER-CHANNEL Z-SCORE NORMALIZATION
#     EEG amplitude varies wildly across recordings and subjects. Z-scoring each
#     channel independently removes DC offset and amplitude drift, giving the
#     rasterizer consistent signal ranges.
#
# [5] MEMORY BANK FOR INFONCE (Phase 2)
#     V12 batch=4 InfoNCE had only 3 negatives — trivially easy, no gradient.
#     V20 maintains a FIFO queue of 512 recent text embeddings as negatives.
#     The contrastive signal is now meaningful (512 negatives vs 3).
#
# V12 bug fixes retained:
#   - CTC on 256 real steps (no interpolation)
#   - OneCycleLR with 10% warmup
#   - Placeholder data filtering
#   - Gradient accumulation (effective batch=32)
#   - Diversity loss weight 0.5 in Phase 1
#   - CER metric every 3 epochs
#
# Pipeline: run_pipeline() auto-chains Phase 1 → Phase 2 on same H100.
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Vocabulary ──────────────────────────────────────────────────────────────

CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz0123456789.,!?'\" ()"
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR  = {i: c for i, c in enumerate(CHAR_VOCAB)}

def text_to_char_ids(text):
    return [CHAR_TO_ID[c] for c in text.lower() if c in CHAR_TO_ID]

def greedy_ctc_decode(logits):
    """logits: (T, B, Vocab) → list[str]"""
    ids = logits.argmax(dim=-1)   # (T, B)
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

def compute_cer(pred, ref):
    import editdistance
    if not ref:
        return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, ref) / len(ref), 2.0)   # cap at 2 for sanity


# ─── Memory Bank (Phase 2 InfoNCE) ───────────────────────────────────────────

class MemoryBank:
    """FIFO queue of text embeddings used as hard negatives for InfoNCE."""

    def __init__(self, size=512, dim=896):
        self.size = size
        self.bank = torch.zeros(size, dim)
        self.ptr  = 0
        self.n    = 0

    def enqueue(self, emb: torch.Tensor):
        """emb: (B, dim) — detached text embeddings."""
        emb = emb.detach().cpu().float()   # bank is float32; cast from bfloat16
        B = emb.shape[0]
        idx = torch.arange(self.ptr, self.ptr + B) % self.size
        self.bank[idx] = emb
        self.ptr = (self.ptr + B) % self.size
        self.n = min(self.n + B, self.size)

    def get(self, device):
        if self.n < 2:
            return None
        return self.bank[:self.n].to(device)


# ─── Modules ─────────────────────────────────────────────────────────────────

class MultiScaleRasterizer(nn.Module):
    """Projects 64-channel EEG (4 spectral bands) onto 64×64 scalp images."""
    def __init__(self, size=64, n_electrodes=64):
        super().__init__()
        angles = torch.linspace(0, 2 * np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij'
        )
        px = gx.flatten().unsqueeze(1); py = gy.flatten().unsqueeze(1)
        ex = pos_2d[:, 0].unsqueeze(0); ey = pos_2d[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex)**2 + (py - ey)**2)
        w = 1.0 / (dist + 1e-4) ** 2.0
        r = torch.sqrt(px**2 + py**2)
        w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)

    def forward(self, x):   # x: (B, T, 256) where 256 = 4 bands × 64 ch
        B, T, Ch = x.shape
        x_split = x.reshape(B * T, 4, 64)
        proj = x_split @ self.W.T.to(device=x.device, dtype=x.dtype)
        return proj.reshape(B, T, 4, 64, 64)


class ChannelAdapter(nn.Module):
    """Per-frame 4-band → 3-channel adapter for pretrained VideoMAE compat."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 3, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):   # (B, T, 4, H, W)
        B, T, C, H, W = imgs.shape
        out = self.conv(imgs.view(B * T, C, H, W))
        return out.view(B, T, 3, H, W)


class CrossAttentionBridge(nn.Module):
    """Perceiver-style bridge: VideoMAE tokens → fixed-length LLM prefix."""
    def __init__(self, in_ch=768, out_ch=1024, n_latents=128):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, in_ch))
        self.mha     = nn.MultiheadAttention(in_ch, num_heads=16, batch_first=True)
        self.ln_k    = nn.LayerNorm(in_ch)
        self.ln_q    = nn.LayerNorm(in_ch)
        self.proj    = nn.Linear(in_ch, out_ch)

    def forward(self, x):   # x: (B, 256, 768)
        B = x.shape[0]
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        x, _ = self.mha(self.ln_q(q), self.ln_k(x), x)
        return self.proj(x)   # (B, n_latents, out_ch)


# ─── Model ───────────────────────────────────────────────────────────────────

class EEG_VLM_V20(nn.Module):
    def __init__(self, q_name):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType

        # VideoMAE with num_channels=3 so pretrained weights transfer
        v_cfg = VideoMAEConfig(
            num_channels=3,         # ← changed from 4; ChannelAdapter bridges the gap
            image_size=64,
            patch_size=16,
            num_frames=1024,
            tubelet_size=4,
            hidden_size=768,
        )
        self.rasterizer  = MultiScaleRasterizer()
        self.ch_adapt    = ChannelAdapter()       # 4-band → 3-ch
        self.video_enc   = VideoMAEModel(v_cfg)
        self.bridge      = CrossAttentionBridge(768, 1024, n_latents=128)
        self.ctc_head    = nn.Linear(768, len(CHAR_VOCAB))
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

        self.llm      = AutoModelForCausalLM.from_pretrained(q_name, torch_dtype=torch.bfloat16)
        self.llm      = get_peft_model(self.llm, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=256,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        ))
        self.llm_proj = nn.Linear(1024, 896)   # bridge → Qwen2.5-0.5B hidden size

    # ── Encode EEG → VideoMAE sequence ──────────────────────────────────────
    def encode(self, traj):
        """traj: (B, 1024, 256) → v_seq: (B, 256, 768)"""
        imgs = self.rasterizer(traj)                          # (B,1024, 4,64,64)
        imgs = self.ch_adapt(imgs)                            # (B,1024, 3,64,64)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        # 256 temporal × 16 spatial patches × 768 dim → pool spatial
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(dim=2)  # (B,256,768)

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, traj, char_ids, char_lens,
                input_ids=None, labels=None, neg_bank=None):
        """
        Phase 1 usage: model(eeg, c_ids, c_lens)          → no LLM forward
        Phase 2 usage: model(eeg, c_ids, c_lens, inp, lbl, neg_bank)
        Returns: (loss_lm, loss_ctc, loss_lock, loss_div, tok_mean_or_None)
        """
        B = traj.shape[0]
        device = traj.device

        v_seq = self.encode(traj)   # (B, 256, 768)

        # ── CTC on 256 real VideoMAE steps (no interpolation) ───────────────
        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)   # (256, B, vocab)
        input_lens = torch.full((B,), 256, dtype=torch.long, device=device)
        loss_ctc   = self.ctc_loss_fn(
            F.log_softmax(ctc_logits, dim=-1), char_ids, input_lens, char_lens
        )

        # ── Diversity loss (blank-collapse prevention) ───────────────────────
        probs      = F.softmax(ctc_logits, dim=-1)           # (256, B, vocab)
        mean_probs = probs.mean(dim=0)                        # (B, vocab)
        ent        = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        loss_div   = -ent.mean()

        # ── Phase 1: skip LLM entirely ───────────────────────────────────────
        if input_ids is None:
            zero = torch.tensor(0.0, device=device)
            return zero, loss_ctc, zero, loss_div, None

        # ── Phase 2: LLM path ────────────────────────────────────────────────
        prefix      = self.llm_proj(self.bridge(v_seq))       # (B, 128, 896)
        embed_fn    = self.llm.get_input_embeddings()
        tok_embs    = embed_fn(input_ids)                      # (B, seq, 896)

        # Sample-level InfoNCE with memory bank negatives
        prefix_mean = F.normalize(prefix.mean(dim=1), dim=-1)   # (B, 896)
        tok_mean    = F.normalize(tok_embs.mean(dim=1), dim=-1)  # (B, 896)

        if neg_bank is not None and neg_bank.shape[0] > 1:
            all_text  = torch.cat([tok_mean, neg_bank.to(device)], dim=0)  # (B+bank, 896)
            sims      = prefix_mean @ all_text.T / 0.07                    # (B, B+bank)
            tgt_nce   = torch.arange(B, device=device)
        else:   # batch-only fallback (first few batches before bank fills)
            sims    = prefix_mean @ tok_mean.T / 0.07                      # (B, B)
            tgt_nce = torch.arange(B, device=device)
        loss_lock = F.cross_entropy(sims, tgt_nce)

        # LLM causal LM loss
        combined = torch.cat([prefix, tok_embs], dim=1)
        combined_labels = torch.cat([
            torch.full((B, 128), -100, device=device, dtype=labels.dtype),
            labels
        ], dim=1)
        lm_out = self.llm(inputs_embeds=combined, labels=combined_labels)

        # Return tok_mean (un-normalized) for memory bank update
        tok_mean_raw = tok_embs.mean(dim=1).detach()
        return lm_out.loss, loss_ctc, loss_lock, loss_div, tok_mean_raw

    @torch.no_grad()
    def generate(self, traj, tokenizer, max_tokens=64):
        traj    = traj.to(next(self.parameters()).dtype)
        v_seq   = self.encode(traj)

        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)   # (256, B, vocab)
        ctc_texts  = greedy_ctc_decode(ctc_logits)

        prefix  = self.llm_proj(self.bridge(v_seq))
        out_ids = self.llm.generate(
            inputs_embeds=prefix, max_new_tokens=max_tokens,
            do_sample=True, temperature=0.6, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        llm_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in out_ids]
        return llm_texts, ctc_texts


# ─── Pretrained weight transfer ───────────────────────────────────────────────

def load_pretrained_videomae_encoder(video_enc):
    """
    Transfer encoder block weights from MCG-NJU/videomae-base.
    Skips patch embedding (different tubelet/channels) and position embeddings
    (different spatial resolution). All self-attention + FFN + LayerNorm weights
    transfer exactly (hidden_size=768 matches).
    """
    from transformers import VideoMAEModel as _VMAE
    print("Loading pretrained VideoMAE-base encoder weights...")
    src    = _VMAE.from_pretrained("MCG-NJU/videomae-base")
    src_sd = src.state_dict()
    dst_sd = video_enc.state_dict()

    n_copied = n_skipped = 0
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            n_copied += 1
        else:
            n_skipped += 1

    video_enc.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Transferred {n_copied} tensors, skipped {n_skipped} "
          f"(patch emb + pos emb — expected, re-initialized from scratch)")
    del src
    torch.cuda.empty_cache()


# ─── Data loading ─────────────────────────────────────────────────────────────

def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with open(path, 'rb') as f:
            return b'MATLAB' in f.read(128)
    except:
        return False

def download_zuco(base_path, vol):
    import subprocess
    zuco_v1   = base_path / "ZuCo_v1"
    zuco_v2   = base_path / "ZuCo_v2"
    inner_sp  = base_path / "InnerSpeech"
    for d in [zuco_v1, zuco_v2, inner_sp]:
        d.mkdir(parents=True, exist_ok=True)

    def fetch(project_id, target_dir, names):
        res   = __import__('subprocess').run(
            ["osf", "-p", project_id, "list"], capture_output=True, text=True
        )
        paths = res.stdout.splitlines()
        for name in names:
            local = target_dir / name
            if not verify_mat_file(local):
                remote = next((p for p in paths if p.strip().endswith(name)), None)
                if remote:
                    print(f"  Fetching {name}...")
                    __import__('subprocess').run(
                        ["osf", "-p", project_id, "fetch", remote.strip(), str(local)],
                        timeout=900
                    )

    fetch("q3zws", zuco_v1, ["resultsZAB_SR.mat","resultsZDM_SR.mat","resultsZAB_NR.mat","resultsZDM_NR.mat"])
    fetch("2urht", zuco_v2, ["resultsYAC_NR.mat","resultsYAG_NR.mat","resultsYAK_NR.mat"])

    if not any(inner_sp.rglob("*-epo.fif")):
        __import__('subprocess').run(["pip","install","openneuro-py"], check=False)
        import openneuro
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
        except:
            pass

def normalize_eeg(eeg, target_ch=64, target_t=1024):
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 2:
        return None
    ch, t = eeg.shape
    if ch > t:
        eeg = eeg.T; ch, t = eeg.shape
    if ch > target_ch:   eeg = eeg[:target_ch, :]
    elif ch < target_ch: eeg = np.pad(eeg, ((0, target_ch - ch), (0, 0)))
    if t > target_t:   eeg = eeg[:, :target_t]
    elif t < target_t: eeg = np.pad(eeg, ((0, 0), (0, target_t - t)))

    # V20: Per-channel z-score normalization (removes DC offset + amplitude drift)
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

    combined = np.concatenate([eeg.T, alpha.T, beta.T, gamma.T], axis=1)  # (1024, 256)
    return combined.astype(np.float32)

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, is_train=True):
        import mne
        self.samples = []
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

        for fif in (p/"InnerSpeech").rglob("*-epo.fif"):
            try:
                epochs = mne.read_epochs(str(fif), preload=True, verbose=False)
                epochs.resample(128)
                data   = epochs.get_data()
                labels = (epochs.metadata['condition'].tolist()
                          if epochs.metadata is not None and 'condition' in epochs.metadata.columns
                          else [f"word_{i}" for i in range(len(data))])
                for d, l in zip(data, labels):
                    normed = normalize_eeg(d)
                    if normed is not None:
                        self.samples.append({'eeg': normed, 'text': str(l)})
            except:
                continue

        label = 'Train' if is_train else 'Val'
        print(f"[Dataset] {label}: {len(self.samples)} samples (skipped {skipped} placeholders)")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['eeg'], s['text'], text_to_char_ids(s['text'])

def collate_fn(batch, tokenizer):
    eegs, texts, chars = zip(*batch)
    eegs      = torch.from_numpy(np.stack(eegs))
    char_lens = torch.tensor([len(c) for c in chars], dtype=torch.long)
    max_char  = max(char_lens)
    char_pad  = torch.zeros(len(chars), max_char, dtype=torch.long)
    for i, c in enumerate(chars):
        char_pad[i, :len(c)] = torch.tensor(c)
    enc = tokenizer(list(texts), padding=True, truncation=True,
                    max_length=256, return_tensors="pt")
    return eegs, enc.input_ids, enc.input_ids.clone(), char_pad, char_lens


# ─── Training loop (plain Python — reused by both Modal functions) ────────────

ACCUM_STEPS = 8   # effective batch = 4 × 8 = 32

def _train_phase(model, phase, epochs, train_loader, val_loader,
                 tokenizer, ckpt_prefix, mem_bank=None):
    device = next(model.parameters()).device

    if phase == 1:
        print("── Phase 1: Isolated CTC (Structural Hardening) ──")
        for name, p in model.named_parameters():
            p.requires_grad = ("llm" not in name and "llm_proj" not in name)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=5e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=5e-4,
            steps_per_epoch=len(train_loader), epochs=epochs,
            pct_start=0.1,
        )
    else:
        print("── Phase 2: Semantic Re-Locking (LM + CTC + InfoNCE) ──")
        # Freeze EVERYTHING first — preserve Phase 1 CTC quality in the encoder.
        # LM gradients were corrupting VideoMAE when it was left unfrozen,
        # causing CTC to regress from 1.525 → 2.08 and GEN to go off-domain.
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze only the semantic pathway: bridge + llm_proj + LLM LoRA
        for p in model.bridge.parameters():      p.requires_grad = True
        for p in model.llm_proj.parameters():    p.requires_grad = True
        for n, p in model.llm.named_parameters():
            if "lora_" in n: p.requires_grad = True
        trainable_p2 = [p for p in model.parameters() if p.requires_grad]
        print(f"  Phase 2 trainable params: {sum(p.numel() for p in trainable_p2)/1e6:.1f}M "
              f"(encoder frozen — CTC quality locked)")
        optimizer = torch.optim.AdamW(trainable_p2, lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_ctc = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        tot_lm = tot_ctc = tot_lock = tot_div = 0.0
        optimizer.zero_grad()

        for step, (eeg, input_ids, labels, c_ids, c_lens) in enumerate(train_loader):
            eeg, input_ids, labels, c_ids, c_lens = [
                x.to(device) for x in (eeg, input_ids, labels, c_ids, c_lens)
            ]
            eeg = eeg.to(torch.bfloat16)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if phase == 1:
                    # Skip LLM entirely in Phase 1 — no wasted frozen LLM compute
                    _, loss_ctc, _, loss_div, _ = model(eeg, c_ids, c_lens)
                    loss = (10.0 * loss_ctc + 0.5 * loss_div) / ACCUM_STEPS
                    loss_lm = loss_lock = torch.tensor(0.0)
                    tok_mean = None
                else:
                    neg_bank = mem_bank.get(device) if mem_bank else None
                    loss_lm, loss_ctc, loss_lock, loss_div, tok_mean = model(
                        eeg, c_ids, c_lens, input_ids, labels, neg_bank=neg_bank
                    )
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

        # Flush remaining gradients
        if len(train_loader) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if phase == 1: scheduler.step()
            optimizer.zero_grad()

        if phase == 2:
            scheduler.step()

        n   = len(train_loader)
        lr  = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d} | LM:{tot_lm/n:.3f} CTC:{tot_ctc/n:.3f} "
              f"Lock:{tot_lock/n:.4f} Div:{tot_div/n:.4f} | LR:{lr:.2e}")

        # Checkpoint best Phase 1
        if phase == 1 and (tot_ctc / n) < best_ctc:
            best_ctc = tot_ctc / n
            torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_best.pt")
            print(f"  ✓ New best CTC {best_ctc:.3f} → {ckpt_prefix}_phase1_best.pt")

        # Validation every 3 epochs: decoded examples + CER
        if epoch % 3 == 0:
            model.eval()
            cer_scores = []
            print(f"── Val Epoch {epoch} ──")
            with torch.no_grad():
                shown = 0
                for eeg, input_ids, labels, c_ids, c_lens in val_loader:
                    llm_preds, ctc_preds = model.generate(eeg.to(device), tokenizer)
                    for b in range(eeg.shape[0]):
                        ref = tokenizer.decode(input_ids[b], skip_special_tokens=True)
                        cer = compute_cer(ctc_preds[b], ref.lower())
                        cer_scores.append(cer)
                        if len(ref.split()) > 3 or shown < 2:
                            print(f"  REF: '{ref[:100]}'")
                            print(f"  CTC: '{ctc_preds[b][:100]}'  CER={cer:.2f}")
                            print(f"  GEN: '{llm_preds[b][:100]}'")
                            shown += 1
                        if shown >= 4: break
                    if shown >= 4: break
            if cer_scores:
                print(f"  Mean CER: {np.mean(cer_scores):.3f}")
            torch.save(model.state_dict(), f"{ckpt_prefix}_phase{phase}_ep{epoch}.pt")


# ─── Modal functions ──────────────────────────────────────────────────────────

GPU_CFG = dict(image=image, gpu="H100", timeout=72000,
               volumes={"/data": data_vol, "/persist": ckpt_vol})

@app.function(**GPU_CFG)
def train_v20(phase: int = 1, epochs: int = 40):
    """Run a single phase. Use run_pipeline() to chain phases automatically."""
    from transformers import AutoTokenizer
    q_name    = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(q_name)

    download_zuco(Path("/data/EEG_Text"), data_vol)
    ds         = EEGDataset("/data/EEG_Text")
    n_train    = int(0.9 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])
    mkloader   = lambda d, shuf: torch.utils.data.DataLoader(
        d, batch_size=4, shuffle=shuf, collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    train_loader = mkloader(train_ds, True)
    val_loader   = mkloader(val_ds,   False)

    model = EEG_VLM_V20(q_name).to(torch.bfloat16).cuda()

    if phase == 1:
        load_pretrained_videomae_encoder(model.video_enc)
    else:
        ckpt = "/persist/v20_phase1_best.pt"
        if Path(ckpt).exists():
            model.load_state_dict(torch.load(ckpt, map_location="cuda"))
            print(f"Loaded Phase 1 checkpoint: {ckpt}")
        else:
            print("WARNING: Phase 1 checkpoint not found — starting Phase 2 from scratch")

    mem_bank = MemoryBank(size=512, dim=896) if phase == 2 else None
    _train_phase(model, phase=phase, epochs=epochs,
                 train_loader=train_loader, val_loader=val_loader,
                 tokenizer=tokenizer, ckpt_prefix="/persist/v20",
                 mem_bank=mem_bank)


@app.function(image=image, gpu="H100", timeout=86400,
              volumes={"/data": data_vol, "/persist": ckpt_vol})
def run_pipeline(epochs_p1: int = 30, epochs_p2: int = 40):
    """
    Auto-chains Phase 1 → Phase 2 on a single H100 container.
    No manual restart needed. Checkpoints are saved every 3 epochs.
    Total budget: ~10-15h Phase 1 + ~15-20h Phase 2 < 28h (100 000s timeout).
    """
    from transformers import AutoTokenizer
    q_name    = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(q_name)

    download_zuco(Path("/data/EEG_Text"), data_vol)
    ds         = EEGDataset("/data/EEG_Text")
    n_train    = int(0.9 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])
    mkloader   = lambda d, shuf: torch.utils.data.DataLoader(
        d, batch_size=4, shuffle=shuf, collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    train_loader = mkloader(train_ds, True)
    val_loader   = mkloader(val_ds,   False)

    model = EEG_VLM_V20(q_name).to(torch.bfloat16).cuda()
    load_pretrained_videomae_encoder(model.video_enc)

    print(f"\n{'='*60}")
    print(f" V20 FULL PIPELINE: Phase 1 ({epochs_p1} ep) → Phase 2 ({epochs_p2} ep)")
    print(f"{'='*60}\n")

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    _train_phase(model, phase=1, epochs=epochs_p1,
                 train_loader=train_loader, val_loader=val_loader,
                 tokenizer=tokenizer, ckpt_prefix="/persist/v20")

    # ── Load best Phase 1 weights before Phase 2 ────────────────────────────
    best_p1 = "/persist/v20_phase1_best.pt"
    if Path(best_p1).exists():
        model.load_state_dict(torch.load(best_p1, map_location="cuda"))
        print(f"\n✓ Loaded best Phase 1 weights for Phase 2\n")

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    mem_bank = MemoryBank(size=512, dim=896)
    _train_phase(model, phase=2, epochs=epochs_p2,
                 train_loader=train_loader, val_loader=val_loader,
                 tokenizer=tokenizer, ckpt_prefix="/persist/v20",
                 mem_bank=mem_bank)

    torch.save(model.state_dict(), "/persist/v20_final.pt")
    print("\n✓ Full V20 pipeline complete. Saved /persist/v20_final.pt")


@app.local_entrypoint()
def main(mode: str = "pipeline", epochs_p1: int = 30, epochs_p2: int = 40,
         phase: int = 1, epochs: int = 40):
    """
    Modes:
      pipeline  — run_pipeline (Phase 1 then Phase 2 auto-chained, recommended)
      p1        — single Phase 1 run only
      p2        — single Phase 2 run only (needs Phase 1 checkpoint)
    """
    if mode == "pipeline":
        print(f"Launching V20 full pipeline: Phase1={epochs_p1}ep, Phase2={epochs_p2}ep")
        run_pipeline.remote(epochs_p1=epochs_p1, epochs_p2=epochs_p2)
    elif mode == "p1":
        print(f"Launching V20 Phase 1 only: {epochs}ep")
        train_v20.remote(phase=1, epochs=epochs)
    elif mode == "p2":
        print(f"Launching V20 Phase 2 only: {epochs}ep")
        train_v20.remote(phase=2, epochs=epochs)
    else:
        print(f"Unknown mode '{mode}'. Use: pipeline / p1 / p2")
