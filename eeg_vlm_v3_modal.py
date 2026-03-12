"""
V3.0 – EEG-to-Text: Lightweight Transformer + ByT5 Cross-Attention Decoder
============================================================================
Architecture:
  • EEG Encoder:  6-layer transformer on band-power time series — NO VideoMAE, NO maps
  • Decoder:      ByT5-small decoder via cross-attention to EEG patches — NO CTC
  • Alignment:    MoCo-queue InfoNCE (K=256, τ=0.07) vs frozen ByT5 text encoder
  • Training:     Single phase, 40 epochs, batch=8, cosine LR

Why this is fundamentally different from all previous versions:
  ✗ VideoMAE pretrained on RGB action videos → zero EEG priors
  ✗ Fake circular electrode topographic maps → no real brain anatomy
  ✗ CTC on noisy EEG → garbage char fragments → ByT5 can't recover (proven V65)
  ✗ PHN InfoNCE with batch=4 → 3 negatives → chance level (proven V65)
  ✗ 86M VideoMAE on 1363 samples → catastrophic overfitting
"""

from pathlib import Path
import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Modal ─────────────────────────────────────────────────────────────────────

app      = modal.App("eeg-vlm-v3")
ckpt_vol = modal.Volume.from_name("bt-checkpoints-v3",  create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data",      create_if_missing=True)
v65_vol  = modal.Volume.from_name("bt-checkpoints-v65", create_if_missing=True)


def _download_models():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    AutoTokenizer.from_pretrained("google/byt5-small")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0",
        "numpy", "scipy", "jiwer", "h5py", "mne", "pandas",
        "accelerate", "osfclient", "editdistance", "openneuro-py",
    ])
    .run_function(_download_models)
)

# ── Constants ─────────────────────────────────────────────────────────────────

VOWELS     = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
F_EEG      = 384          # 6 band-envelopes × 64 channels (V65 preprocessing)
T_EEG      = 1024         # timesteps @ 500 Hz
D_EEG      = 256          # EEG encoder hidden dim
PATCH_SIZE = 16           # temporal patch size
N_PATCHES  = T_EEG // PATCH_SIZE  # 64

# ── EEG preprocessing (identical to V65 – reuses its cache) ──────────────────

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
    zuco_v1 = base_path / "ZuCo_v1"; zuco_v2 = base_path / "ZuCo_v2"
    for d in [zuco_v1, zuco_v2]:
        d.mkdir(parents=True, exist_ok=True)

    def fetch(pid, tdir, names):
        if all(verify_mat_file(tdir / n) for n in names):
            print(f"  Cached {tdir.name}"); return
        res = subprocess.run(["osf", "-p", pid, "list"],
                             capture_output=True, text=True, timeout=120)
        paths = res.stdout.splitlines()
        for name in names:
            loc = tdir / name
            if verify_mat_file(loc): continue
            rem = next((p for p in paths if p.strip().endswith(name)), None)
            if rem:
                print(f"  Fetching {name}…")
                subprocess.run(["osf", "-p", pid, "fetch", rem.strip(), str(loc)],
                               timeout=900)

    fetch("q3zws", zuco_v1, ["resultsZAB_SR.mat", "resultsZDM_SR.mat",
                              "resultsZAB_NR.mat",  "resultsZDM_NR.mat"])
    fetch("2urht", zuco_v2, ["resultsYAC_NR.mat", "resultsYAG_NR.mat",
                              "resultsYAK_NR.mat"])
    vol.commit()


def load_mat_any(path):
    import scipy.io as sio, h5py
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        sentences = data['sentenceData']
        if not isinstance(sentences, (np.ndarray, list)):
            sentences = [sentences]
        for s in sentences:
            if (hasattr(s, 'rawData') and hasattr(s, 'content')
                    and isinstance(s.rawData, np.ndarray)):
                yield s.rawData, str(s.content)
    except:
        try:
            with h5py.File(path, 'r') as f:
                if 'sentenceData' in f:
                    for key in f['sentenceData'].keys():
                        s = f['sentenceData'][key]
                        if 'rawData' in s and 'content' in s:
                            yield np.array(s['rawData']).T, "placeholder"
        except:
            pass


def normalize_eeg(eeg, target_ch=64, target_t=T_EEG):
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 2:
        return None
    ch, t = eeg.shape
    if ch > t: eeg = eeg.T; ch, t = eeg.shape
    if ch > target_ch:   eeg = eeg[:target_ch, :]
    elif ch < target_ch: eeg = np.pad(eeg, ((0, target_ch - ch), (0, 0)))
    if t > target_t:     eeg = eeg[:, :target_t]
    elif t < target_t:   eeg = np.pad(eeg, ((0, 0), (0, target_t - t)))

    mu  = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1,  keepdims=True).clip(min=1e-6)
    eeg = (eeg - mu) / std

    from scipy.signal import butter, filtfilt
    def band_env(lo, hi, data):
        try:
            b, a = butter(4, [lo / 64.0, hi / 64.0], btype='bandpass')
            env  = np.abs(filtfilt(b, a, data, axis=1))
            return np.convolve(env.flatten(), np.ones(5) / 5,
                               mode='same').reshape(env.shape)
        except:
            return np.zeros_like(data)

    alpha = band_env(8,  13,  eeg)
    beta  = band_env(13, 30,  eeg)
    gamma = band_env(30, 100, eeg)
    delta = np.diff(eeg,   axis=1, prepend=eeg[:, :1])
    gd    = np.diff(gamma, axis=1, prepend=gamma[:, :1])

    def rz(x):
        m = x.mean(axis=1, keepdims=True)
        s = x.std(axis=1,  keepdims=True).clip(min=1e-6)
        return (x - m) / s

    combined = np.concatenate(
        [eeg.T, alpha.T, beta.T, gamma.T, rz(delta).T, rz(gd).T], axis=1)
    return combined.astype(np.float32)  # (1024, 384)


def load_or_build_cache(base_path, ckpt_vol):
    # Prefer V65 cache (same format); fall back to building fresh
    for p in [Path("/v65/v65_data_cache.pt"), Path("/persist/v3_data_cache.pt")]:
        if p.exists():
            print(f"Loading EEG cache from {p}")
            raw = torch.load(p, weights_only=False)
            return raw["eegs"], raw["texts"]

    print("Building EEG dataset cache from mat files …")
    eegs, texts = [], []
    skipped = 0
    for zp in [base_path / "ZuCo_v1", base_path / "ZuCo_v2"]:
        for mat in sorted(zp.rglob("*.mat")):
            for raw_eeg, text in load_mat_any(mat):
                if not text or "placeholder" in text.lower():
                    skipped += 1; continue
                normed = normalize_eeg(raw_eeg)
                if normed is None:
                    skipped += 1; continue
                eegs.append(normed); texts.append(text)

    print(f"[Cache] total={len(eegs)} skipped={skipped}")
    cache = {"eegs": np.stack(eegs).astype(np.float32), "texts": texts}
    torch.save(cache, Path("/persist/v3_data_cache.pt"))
    ckpt_vol.commit()
    return cache["eegs"], cache["texts"]

# ── Dataset ───────────────────────────────────────────────────────────────────

def _augment(eeg):
    T, C = eeg.shape
    # Time shift
    shift = np.random.randint(-64, 65)
    if shift:
        eeg = np.roll(eeg, shift, 0)
        if shift > 0: eeg[:shift]  = 0
        else:         eeg[shift:]  = 0
    # Amplitude jitter
    eeg = eeg * np.random.uniform(0.85, 1.15, (1, C)).astype(np.float32)
    # Gaussian noise
    eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * 0.05
    # Channel dropout
    drop = np.random.choice(C, max(1, int(C * 0.10)), replace=False)
    eeg[:, drop] = 0
    # Time masking (2 spans)
    for _ in range(2):
        span  = np.random.randint(16, 64)
        start = np.random.randint(0, max(1, T - span))
        eeg[start:start + span] = 0
    return eeg


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, eegs, texts, indices, augment=False):
        self.eegs    = eegs
        self.texts   = texts
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        eeg = self.eegs[idx].copy()
        if self.augment:
            eeg = _augment(eeg)
        return eeg, self.texts[idx]

# ── Model ─────────────────────────────────────────────────────────────────────

class EEGTransformer(nn.Module):
    """
    Lightweight transformer encoder directly on band-power EEG time series.
    Input:  (B, T=1024, F=384)
    Output: (B, N=64, D=256)   — 64 temporal patches
    """
    def __init__(self, in_ch=F_EEG, d=D_EEG, heads=8, layers=6, patch=PATCH_SIZE):
        super().__init__()
        # Conv1D patch embedding: (B, F, T) → (B, D, N)
        self.patch_embed = nn.Conv1d(in_ch, d, kernel_size=patch, stride=patch)
        self.pos_embed   = nn.Parameter(torch.randn(1, N_PATCHES, d) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=d * 4,
            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm        = nn.LayerNorm(d)

    def forward(self, x):
        # x: (B, T, F) → (B, F, T) for Conv1D → (B, D, N) → (B, N, D)
        x = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(self.transformer(x + self.pos_embed))
        return x  # (B, 64, 256)


class EEGToText(nn.Module):
    """
    EEG encoder + contrastive alignment + ByT5 cross-attention decoder.

    Contrastive path:
      EEG mean pool → Linear → norm → q
      frozen ByT5 encoder(text) → mean pool → Linear → norm → k
      MoCo queue of K=256 past text embeddings as negatives

    Generation path:
      EEG patches → Linear(D→byt5_d) → feed as encoder_outputs to ByT5 decoder
    """
    def __init__(self, byt5_d: int, queue_K: int = 256,
                 contrast_d: int = 128, temp: float = 0.07):
        super().__init__()
        self.eeg_enc      = EEGTransformer()
        self.eeg_to_byt5  = nn.Linear(D_EEG, byt5_d)          # cross-attn projection
        self.eeg_contrast = nn.Sequential(                      # EEG → contrast space
            nn.Linear(D_EEG, D_EEG), nn.GELU(), nn.Linear(D_EEG, contrast_d))
        self.txt_contrast = nn.Sequential(                      # text → contrast space
            nn.Linear(byt5_d, D_EEG), nn.GELU(), nn.Linear(D_EEG, contrast_d))
        self.temp = temp
        # MoCo-style text embedding queue (fixed negatives from frozen encoder)
        self.register_buffer("queue",
            F.normalize(torch.randn(queue_K, contrast_d), dim=1))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.K = queue_K

    @torch.no_grad()
    def _enqueue(self, k):
        B   = k.shape[0]
        ptr = int(self.ptr)
        # Use .data to bypass autograd version counter
        end = min(ptr + B, self.K)
        self.queue.data[ptr:end] = k[:end - ptr].detach()
        if end < ptr + B:
            self.queue.data[:ptr + B - self.K] = k[end - ptr:].detach()
        self.ptr[0] = (ptr + B) % self.K

    def forward(self, eeg, txt_embed, byt5, labels):
        """
        eeg:       (B, T, F)   bfloat16 on GPU
        txt_embed: (B, byt5_d) from frozen ByT5 encoder (no grad)
        byt5:      T5ForConditionalGeneration
        labels:    (B, L)      tokenized target with -100 for pad
        Returns:   (loss_lm, loss_contrastive)
        """
        from transformers.modeling_outputs import BaseModelOutput

        patches = self.eeg_enc(eeg)           # (B, 64, D_EEG)

        # ── Contrastive (MoCo InfoNCE) ────────────────────────────────────────
        q = F.normalize(self.eeg_contrast(patches.mean(1)), dim=-1)       # (B, 128)
        k = F.normalize(self.txt_contrast(txt_embed.detach()), dim=-1)    # (B, 128)
        pos  = (q * k).sum(-1, keepdim=True) / self.temp                  # (B, 1)
        negs = q @ self.queue.T.detach() / self.temp                      # (B, K)
        loss_ctr = F.cross_entropy(
            torch.cat([pos, negs], dim=1),
            torch.zeros(len(q), dtype=torch.long, device=q.device))
        self._enqueue(k)

        # ── Generation (ByT5 cross-attention) ─────────────────────────────────
        enc_out  = BaseModelOutput(
            last_hidden_state=self.eeg_to_byt5(patches))  # (B, 64, byt5_d)
        loss_lm  = byt5(encoder_outputs=enc_out, labels=labels).loss

        return loss_lm, loss_ctr

    @torch.no_grad()
    def generate(self, eeg, byt5, **kw):
        from transformers.modeling_outputs import BaseModelOutput
        patches = self.eeg_enc(eeg)
        enc_out = BaseModelOutput(last_hidden_state=self.eeg_to_byt5(patches))
        return byt5.generate(encoder_outputs=enc_out, **kw)

# ── Training ──────────────────────────────────────────────────────────────────

def train_loop(model, byt5, tokenizer, train_ds, val_ds, epochs,
               ckpt_prefix, device, start_ep=0):
    tr_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=8, shuffle=True,  num_workers=0)
    vl_loader = torch.utils.data.DataLoader(
        val_ds,   batch_size=8, shuffle=False, num_workers=0)

    # Separate LRs: EEG encoder + projections higher, ByT5 decoder lower
    opt = torch.optim.AdamW([
        {"params": model.eeg_enc.parameters(),     "lr": 1e-4},
        {"params": list(model.eeg_to_byt5.parameters()) +
                   list(model.eeg_contrast.parameters()) +
                   list(model.txt_contrast.parameters()), "lr": 1e-4},
        {"params": [p for p in byt5.decoder.parameters()
                    if p.requires_grad],            "lr": 3e-5},
    ], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=1e-6)
    # Fast-forward scheduler if resuming
    for _ in range(start_ep):
        sched.step()

    best_wer = float("inf")
    ACCUM    = 4  # effective batch = 32

    for ep in range(start_ep + 1, start_ep + epochs + 1):
        model.train(); byt5.train()
        tot_lm = tot_ctr = n_steps = 0
        opt.zero_grad()

        for step, (eegs, texts) in enumerate(tr_loader, 1):
            eegs = eegs.to(device, dtype=torch.bfloat16)

            # Tokenize reference texts
            tok    = tokenizer(list(texts), padding=True, truncation=True,
                               max_length=192, return_tensors="pt")
            labels = tok.input_ids.to(device).clone()
            labels[labels == tokenizer.pad_token_id] = -100

            # Frozen text embeddings for contrastive
            with torch.no_grad():
                enc_out = byt5.encoder(
                    input_ids=tok.input_ids.to(device),
                    attention_mask=tok.attention_mask.to(device))
                mask    = tok.attention_mask.to(device).unsqueeze(-1).float()
                txt_emb = ((enc_out.last_hidden_state * mask).sum(1)
                           / mask.sum(1).clamp(min=1e-6))
                txt_emb = txt_emb.to(dtype=torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_lm, loss_ctr = model(eegs, txt_emb, byt5, labels)
                loss = (loss_lm + 0.5 * loss_ctr) / ACCUM

            loss.backward()
            tot_lm  += loss_lm.item()
            tot_ctr += loss_ctr.item()
            n_steps += 1

            if step % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) +
                    [p for p in byt5.decoder.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()

        # Flush tail
        if n_steps % ACCUM:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) +
                [p for p in byt5.decoder.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()

        sched.step()
        n = max(n_steps, 1)
        print(f"Ep{ep:3d} | LM:{tot_lm/n:.3f} CTR:{tot_ctr/n:.3f} "
              f"| lr:{opt.param_groups[0]['lr']:.1e}")

        if ep % 5 == 0 or ep == start_ep + epochs:
            wer = _validate(model, byt5, tokenizer, vl_loader, device, ep)
            state = {"model": model.state_dict(),
                     "decoder": byt5.decoder.state_dict(), "epoch": ep}
            torch.save(state, f"{ckpt_prefix}_ep{ep}.pt")
            if wer < best_wer:
                best_wer = wer
                torch.save(state, f"{ckpt_prefix}_best.pt")
                print(f"  ✓ New best WER {best_wer:.3f}")
            ckpt_vol.commit()


def _validate(model, byt5, tokenizer, loader, device, epoch):
    import editdistance
    model.eval(); byt5.eval()
    wers, cers = [], []
    v_corr = v_tot = c_corr = c_tot = 0
    shown = 0
    print(f"\n── Val Epoch {epoch} ──")

    with torch.no_grad():
        for eegs, texts in loader:
            eegs = eegs.to(device, dtype=torch.bfloat16)
            out  = model.generate(
                eegs, byt5,
                max_new_tokens=192, num_beams=8,
                repetition_penalty=1.3, length_penalty=0.8,
                no_repeat_ngram_size=4)
            preds = tokenizer.batch_decode(out, skip_special_tokens=True)

            for pred, ref in zip(preds, texts):
                p, r = pred.strip().lower(), ref.strip().lower()
                pw, rw = p.split(), r.split()
                wers.append(
                    min(editdistance.eval(pw, rw) / max(len(rw), 1), 2.0))
                cers.append(
                    min(editdistance.eval(p, r)  / max(len(r),  1), 2.0))
                for c in r:
                    if c in VOWELS:
                        v_tot += 1; v_corr += int(c in p)
                    elif c in CONSONANTS:
                        c_tot += 1; c_corr += int(c in p)
                if shown < 4:
                    print(f"  REF: '{ref[:80]}'")
                    print(f"  GEN: '{pred[:80]}'  WER={wers[-1]:.2f}")
                    shown += 1

    print(f"  Mean CER:{np.mean(cers):.3f}  WER:{np.mean(wers):.3f}")
    print(f"  VowelRecall:{v_corr/max(v_tot,1):.2f}  "
          f"ConsRecall:{c_corr/max(c_tot,1):.2f}")
    model.train(); byt5.train()
    return float(np.mean(wers))

# ── Modal function ────────────────────────────────────────────────────────────

@app.function(
    image=image, gpu="H100", timeout=86400,
    volumes={"/data": data_vol, "/persist": ckpt_vol, "/v65": v65_vol},
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=10.0))
def run_training(epochs: int = 40):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    download_zuco(Path("/data/EEG_Text"), data_vol)

    # ── Dataset ───────────────────────────────────────────────────────────────
    eegs, texts = load_or_build_cache(Path("/data/EEG_Text"), ckpt_vol)
    print(f"Dataset: {len(texts)} samples")

    np.random.seed(42)
    idx  = np.random.permutation(len(texts)).tolist()
    n_tr = int(0.9 * len(idx))
    train_ds = EEGDataset(eegs, texts, idx[:n_tr], augment=True)
    val_ds   = EEGDataset(eegs, texts, idx[n_tr:], augment=False)

    # ── ByT5 ──────────────────────────────────────────────────────────────────
    print("Loading ByT5-small …")
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    byt5 = AutoModelForSeq2SeqLM.from_pretrained(
        "google/byt5-small", torch_dtype=torch.bfloat16).cuda()

    # Freeze ByT5 encoder transformer blocks (keep shared embeddings trainable)
    for layer in byt5.encoder.block:
        for p in layer.parameters():
            p.requires_grad = False
    for p in byt5.encoder.final_layer_norm.parameters():
        p.requires_grad = False

    enc_frozen = sum(p.numel() for p in byt5.encoder.parameters()
                     if not p.requires_grad)
    dec_train  = sum(p.numel() for p in byt5.decoder.parameters()
                     if p.requires_grad)
    print(f"  ByT5 encoder frozen: {enc_frozen/1e6:.1f}M | "
          f"decoder trainable: {dec_train/1e6:.1f}M")

    # ── EEG model ─────────────────────────────────────────────────────────────
    byt5_d = byt5.config.d_model
    model  = EEGToText(byt5_d=byt5_d).to(torch.bfloat16).cuda()
    eeg_p  = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  EEG model: {eeg_p:.1f}M params  |  ByT5 d_model: {byt5_d}")

    # ── Resume if possible ────────────────────────────────────────────────────
    ckpts    = sorted(Path("/persist").glob("v3_ep*.pt"))
    start_ep = 0
    if ckpts:
        last = ckpts[-1]
        d    = torch.load(last, map_location="cuda")
        model.load_state_dict(d["model"])
        byt5.decoder.load_state_dict(d["decoder"])
        start_ep = d.get("epoch", int(last.stem.split("_ep")[-1]))
        print(f"Resuming from {last.name} (epoch {start_ep})")
        remaining = epochs - start_ep
        if remaining <= 0:
            print("Already complete."); return
        epochs = remaining

    print(f"\n{'='*60}")
    print(f" V3.0: EEG Transformer + ByT5 Cross-Attention")
    print(f" total_epochs={start_ep + epochs}  batch=8  accum=4  queue_K=256")
    print(f"{'='*60}\n")

    train_loop(model, byt5, tokenizer, train_ds, val_ds,
               epochs=epochs, ckpt_prefix="/persist/v3",
               device=torch.device("cuda"), start_ep=start_ep)

    print("\n✓ V3.0 complete.")


@app.local_entrypoint()
def main(epochs: int = 40):
    run_training.remote(epochs=epochs)
