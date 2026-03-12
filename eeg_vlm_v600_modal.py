"""
V600: EEG-as-Video → Frozen GPT-2 Prefix Generation (Open-Text Decoding)
=========================================================================
ROOT CAUSE of all previous decoder failures:
  V3   – ByT5 decoder trainable  → learned language prior, ignored EEG entirely
  V61  – Qwen trainable (LoRA)   → same; "The the she and of thing thor..."
  V400 – ByT5 trainable          → alphabet-like token bags
  V401 – Whisper decoder         → drifted to biography text

THE FIX — ClipCap-style prefix tuning with FROZEN LLM:
  A FROZEN LLM physically cannot learn to ignore its prefix.
  Train only a small prefix projector (< 5M params) on 1363 samples.
  No mode collapse possible because GPT-2 is not trainable.

ARCHITECTURE:
  EEG (1024, 384)
    → MultiScaleRasterizer (video topographic maps)
    → ChannelAdapter (6 bands → 3 RGB channels)
    → VideoMAE-base (ALL 12 layers, fine-tuned, differential LR)
    → mean-pool (768-dim)
    → sem_proj  MLP(768→256→128)  L2-normalized   [alignment head]
    → prefix_proj MLP(128→512→20×768)              [generation head]
    → 20 prefix token embeddings prepended to GPT-2 input
    → GPT-2-small (117M, COMPLETELY FROZEN, no LoRA, no adapters)
    → beam generation at inference

TWO TRAINING LOSSES (jointly, single phase):
  LM loss   : cross-entropy over target text tokens    weight=1.0
  Align loss: InfoNCE(EEG sem vs SBERT text) K=4096   weight=0.5

INFERENCE:
  EEG → VideoMAE → sem_proj (128) → prefix_proj → 20 prefix embeds
  GPT-2 beam search from [prefix | "The sentence reads: "] → text
  Also report: top beam candidates for analysis
"""

from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Modal ──────────────────────────────────────────────────────────────────────

app      = modal.App("eeg-vlm-v600")
ckpt_vol = modal.Volume.from_name("bt-checkpoints-v600", create_if_missing=True)
v61_vol  = modal.Volume.from_name("bt-checkpoints-v61",  create_if_missing=False)
data_vol = modal.Volume.from_name("mindvoice-data",       create_if_missing=True)


def _download_models():
    from transformers import VideoMAEModel, GPT2LMHeadModel, GPT2Tokenizer
    from sentence_transformers import SentenceTransformer
    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    GPT2LMHeadModel.from_pretrained("gpt2")
    GPT2Tokenizer.from_pretrained("gpt2")
    SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "accelerate", "osfclient", "editdistance", "openneuro-py",
        "sentence-transformers",
    ])
    .run_function(_download_models)
)

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_EEG_BANDS = 6
ACCUM_STEPS   = 8
QUEUE_SIZE    = 4096
SEM_DIM       = 128
SBERT_DIM     = 768
GPT2_DIM      = 768      # gpt2-small hidden size
N_PREFIX      = 20       # prefix token count (larger = more EEG influence)
MAX_TEXT_LEN  = 80       # max GPT-2 target tokens
PROMPT        = "The sentence reads: "   # generation primer

VOWELS     = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")

# ── EEG preprocessing (identical to V65/V100 — reuses cache) ──────────────────

def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000: return False
    try:
        with open(path, 'rb') as f: return b'MATLAB' in f.read(128)
    except: return False


def download_zuco(base_path, vol):
    import subprocess
    zuco_v1 = base_path / "ZuCo_v1"; zuco_v2 = base_path / "ZuCo_v2"
    for d in [zuco_v1, zuco_v2]: d.mkdir(parents=True, exist_ok=True)

    def fetch(pid, tdir, names):
        if not any(not verify_mat_file(tdir / n) for n in names):
            print(f"  Cached {tdir.name}"); return
        res = subprocess.run(["osf", "-p", pid, "list"],
                             capture_output=True, text=True, timeout=120)
        for name in names:
            loc = tdir / name
            if verify_mat_file(loc): continue
            rem = next((p for p in res.stdout.splitlines()
                        if p.strip().endswith(name)), None)
            if rem:
                print(f"  Fetching {name}…")
                subprocess.run(["osf", "-p", pid, "fetch", rem.strip(), str(loc)],
                               timeout=900)

    fetch("q3zws", zuco_v1, ["resultsZAB_SR.mat", "resultsZDM_SR.mat",
                              "resultsZAB_NR.mat", "resultsZDM_NR.mat"])
    fetch("2urht", zuco_v2, ["resultsYAC_NR.mat", "resultsYAG_NR.mat",
                              "resultsYAK_NR.mat"])
    vol.commit()


def load_mat_any(path):
    import scipy.io as sio, h5py
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        sentences = data['sentenceData']
        if not isinstance(sentences, (np.ndarray, list)): sentences = [sentences]
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
                            yield np.array(s['rawData']).T, "hdf5_placeholder"
        except: pass


def normalize_eeg(eeg, target_ch=64, target_t=1024):
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 2: return None
    ch, t = eeg.shape
    if ch > t: eeg = eeg.T; ch, t = eeg.shape
    if ch > target_ch:   eeg = eeg[:target_ch, :]
    elif ch < target_ch: eeg = np.pad(eeg, ((0, target_ch - ch), (0, 0)))
    if t > target_t:   eeg = eeg[:, :target_t]
    elif t < target_t: eeg = np.pad(eeg, ((0, 0), (0, target_t - t)))
    mu = eeg.mean(1, keepdims=True); std = eeg.std(1, keepdims=True).clip(1e-6)
    eeg = (eeg - mu) / std

    from scipy.signal import butter, filtfilt
    def band_env(lo, hi, d):
        try:
            b, a = butter(4, [lo/64., hi/64.], btype='bandpass')
            env = np.abs(filtfilt(b, a, d, axis=1))
            return np.convolve(env.flatten(), np.ones(5)/5, mode='same').reshape(env.shape)
        except: return np.zeros_like(d)

    alpha = band_env(8, 13, eeg); beta = band_env(13, 30, eeg)
    gamma = band_env(30, 100, eeg)
    delta = np.diff(eeg, axis=1, prepend=eeg[:, :1])
    gd    = np.diff(gamma, axis=1, prepend=gamma[:, :1])
    def rz(x):
        m = x.mean(1, keepdims=True); s = x.std(1, keepdims=True).clip(1e-6)
        return (x - m) / s
    return np.concatenate(
        [eeg.T, alpha.T, beta.T, gamma.T, rz(delta).T, rz(gd).T], axis=1
    ).astype(np.float32)  # (1024, 384)


def load_or_build_cache(base_path, vol):
    cache_path = Path("/persist/v65_data_cache.pt")
    if cache_path.exists():
        print("EEG cache hit …"); return torch.load(cache_path, weights_only=False)
    print("Building EEG cache …")
    eegs, texts, skipped = [], [], 0
    for zp in [base_path / "ZuCo_v1", base_path / "ZuCo_v2"]:
        for mat in sorted(zp.rglob("*.mat")):
            for raw, text in load_mat_any(mat):
                if not text or "placeholder" in text.lower():
                    skipped += 1; continue
                n = normalize_eeg(raw)
                if n is None: skipped += 1; continue
                eegs.append(n); texts.append(text)
    cache = {"eegs": np.stack(eegs).astype(np.float32), "texts": texts,
             "fingerprints": np.zeros((len(eegs), 7), dtype=np.float32)}
    torch.save(cache, cache_path); vol.commit()
    print(f"Cache: {len(eegs)} samples, {skipped} skipped")
    return cache


# ── Augmentation ───────────────────────────────────────────────────────────────

def _augment(eeg):
    T, C = eeg.shape
    shift = np.random.randint(-64, 65)
    if shift:
        eeg = np.roll(eeg, shift, 0)
        if shift > 0: eeg[:shift] = 0
        else:         eeg[shift:] = 0
    eeg *= np.random.uniform(0.88, 1.12, (1, C)).astype(np.float32)
    eeg += np.random.randn(*eeg.shape).astype(np.float32) * 0.04
    eeg[:, np.random.choice(C, max(1, int(C * 0.10)), replace=False)] = 0
    for _ in range(3):
        span  = np.random.randint(16, 64)
        start = np.random.randint(0, max(1, T - span))
        eeg[start:start + span] = 0
    return eeg


# ── Dataset ────────────────────────────────────────────────────────────────────

class EEGTextDataset(torch.utils.data.Dataset):
    def __init__(self, eegs, texts, sbert_np, input_ids_list, attn_mask_list,
                 indices, augment=False):
        self.eegs           = eegs
        self.texts          = texts
        self.sbert_np       = sbert_np
        self.input_ids_list = input_ids_list
        self.attn_mask_list = attn_mask_list
        self.indices        = indices
        self.augment        = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        gi  = self.indices[i]
        eeg = self.eegs[gi].copy()
        if self.augment: eeg = _augment(eeg)
        return (eeg,
                self.texts[gi],
                self.sbert_np[gi],
                self.input_ids_list[gi],
                self.attn_mask_list[gi])


def collate_fn(batch):
    eegs, texts, sberts, input_ids_list, attn_mask_list = zip(*batch)
    eegs   = torch.from_numpy(np.stack(eegs))
    sberts = torch.from_numpy(np.stack(sberts))
    # Pad input_ids and attention masks to same length in batch
    max_len = max(ids.shape[0] for ids in input_ids_list)
    B = len(batch)
    padded_ids  = torch.zeros(B, max_len, dtype=torch.long)
    padded_mask = torch.zeros(B, max_len, dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(input_ids_list, attn_mask_list)):
        L = ids.shape[0]
        padded_ids[i, :L]  = ids
        padded_mask[i, :L] = mask
    return eegs, list(texts), sberts, padded_ids, padded_mask


# ── Model ──────────────────────────────────────────────────────────────────────

class MultiScaleRasterizer(nn.Module):
    def __init__(self, size=64, n_electrodes=64, n_bands=NUM_EEG_BANDS):
        super().__init__()
        self.n_bands = n_bands; self.n_electrodes = n_electrodes
        angles = torch.linspace(0, 2 * np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
        px = gx.flatten().unsqueeze(1); py = gy.flatten().unsqueeze(1)
        ex = pos_2d[:, 0].unsqueeze(0); ey = pos_2d[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex)**2 + (py - ey)**2)
        w    = 1.0 / (dist + 1e-4) ** 2.0
        r    = torch.sqrt(px**2 + py**2)
        w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)

    def forward(self, x):
        B, T, _ = x.shape
        xs = x.reshape(B * T, self.n_bands, self.n_electrodes)
        return (xs @ self.W.T.to(x.device, x.dtype)).reshape(B, T, self.n_bands, 64, 64)


class ChannelAdapter(nn.Module):
    def __init__(self, in_ch=NUM_EEG_BANDS):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3, 1)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        return self.conv(imgs.view(B*T, C, H, W)).view(B, T, 3, H, W)


class EEGV600(nn.Module):
    """
    EEG-as-Video + Frozen GPT-2 prefix generation.
    Only VideoMAE (fine-tune, diff LR), sem_proj, txt_proj, prefix_proj are trainable.
    GPT-2 is 100% frozen — cannot learn to ignore the EEG prefix.
    """
    def __init__(self):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel, GPT2LMHeadModel

        # EEG video encoder
        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=1024, tubelet_size=4, hidden_size=768)
        self.rasterizer = MultiScaleRasterizer()
        self.ch_adapt   = ChannelAdapter()
        self.video_enc  = VideoMAEModel(v_cfg)

        # Alignment projectors
        self.sem_proj = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256), nn.GELU(),
            nn.Linear(256, SEM_DIM))
        self.txt_proj = nn.Sequential(
            nn.Linear(SBERT_DIM, 256), nn.GELU(),
            nn.Linear(256, SEM_DIM))

        # MoCo memory bank
        self.register_buffer("queue",
            F.normalize(torch.randn(QUEUE_SIZE, SEM_DIM), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Prefix projector: 128-dim → N_PREFIX × GPT2_DIM
        self.prefix_proj = nn.Sequential(
            nn.Linear(SEM_DIM, 512), nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(512, 1024), nn.GELU(),
            nn.Linear(1024, N_PREFIX * GPT2_DIM))

        # GPT-2: COMPLETELY FROZEN (no LoRA, no adapters, nothing)
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        for p in self.gpt2.parameters():
            p.requires_grad = False

    # ── Encoder ───────────────────────────────────────────────────────────────

    def encode(self, traj):
        imgs  = self.rasterizer(traj)
        imgs  = self.ch_adapt(imgs)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state  # (B,4096,768)
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(2)   # (B,256,768)

    def get_sem_emb(self, traj):
        v_seq  = self.encode(traj)
        v_mean = v_seq.mean(1)
        return F.normalize(self.sem_proj(v_mean), dim=-1)

    # ── MoCo enqueue ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def _enqueue(self, k):
        B   = k.shape[0]
        ptr = int(self.queue_ptr.item())
        end = ptr + B
        if end <= QUEUE_SIZE:
            self.queue.data[ptr:end] = k.detach().float()
        else:
            p1 = QUEUE_SIZE - ptr
            self.queue.data[ptr:]   = k[:p1].detach().float()
            self.queue.data[:B-p1] = k[p1:].detach().float()
        self.queue_ptr.data[0] = end % QUEUE_SIZE

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(self, traj, sbert_embs, prompt_ids, target_ids, target_mask):
        """
        traj:        (B, 1024, 384) EEG
        sbert_embs:  (B, 768)  SBERT embedding of target text
        prompt_ids:  (P,)      tokenized PROMPT (shared across batch)
        target_ids:  (B, T)    tokenized target text (padded)
        target_mask: (B, T)    1 where real tokens, 0 for padding
        """
        B, device = traj.shape[0], traj.device
        dtype = next(p for p in self.parameters() if p.requires_grad).dtype

        # ── EEG encode ────────────────────────────────────────────────────────
        v_seq  = self.encode(traj)
        v_mean = v_seq.mean(1)                                    # (B, 768)
        q = F.normalize(self.sem_proj(v_mean), dim=-1)            # (B, 128)

        # ── InfoNCE alignment ─────────────────────────────────────────────────
        k = F.normalize(
            self.txt_proj(sbert_embs.to(device, dtype=dtype)), dim=-1)  # (B, 128)
        pos        = (q * k).sum(-1, keepdim=True) / 0.07
        negs       = q @ self.queue.T.detach() / 0.07
        loss_align = F.cross_entropy(
            torch.cat([pos, negs], dim=1),
            torch.zeros(B, dtype=torch.long, device=device))
        self._enqueue(k)

        # ── Prefix tokens ─────────────────────────────────────────────────────
        prefix_flat = self.prefix_proj(q)                          # (B, N*768)
        prefix_embs = prefix_flat.view(B, N_PREFIX, GPT2_DIM)     # (B, N, 768)

        # ── GPT-2 forward (completely frozen) ─────────────────────────────────
        # Build input: [prefix | prompt | target]
        # Labels:      [-100…  | -100…  | target_ids]
        wte = self.gpt2.transformer.wte  # word-token embedding

        # Prompt embedding (same for every sample)
        prompt_ids_dev  = prompt_ids.to(device)                    # (P,)
        prompt_embs     = wte(prompt_ids_dev).unsqueeze(0).expand(B, -1, -1)  # (B,P,768)
        P = prompt_embs.shape[1]

        # Target embedding
        target_ids_dev = target_ids.to(device)                     # (B, T)
        target_embs    = wte(target_ids_dev)                       # (B, T, 768)
        T_len = target_embs.shape[1]

        # Concatenate: prefix + prompt + target
        combined_embs = torch.cat(
            [prefix_embs.float(), prompt_embs.float(), target_embs.float()], dim=1)
        # (B, N + P + T, 768)

        # Labels: -100 for prefix+prompt, target token IDs for target
        ignore = torch.full((B, N_PREFIX + P), -100, dtype=torch.long, device=device)
        # Mask padding positions in target with -100
        target_labels = target_ids_dev.clone()
        target_labels[target_mask.to(device) == 0] = -100
        combined_labels = torch.cat([ignore, target_labels], dim=1)

        # GPT-2 attention mask (1 for all positions)
        combined_attn = torch.ones(B, N_PREFIX + P + T_len,
                                   dtype=torch.long, device=device)

        out       = self.gpt2(inputs_embeds=combined_embs,
                              attention_mask=combined_attn,
                              labels=combined_labels)
        loss_lm   = out.loss

        return loss_lm, loss_align

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, traj, prompt_ids, tokenizer, max_new_tokens=80, num_beams=5):
        """
        Generate text for a batch of EEG trials.
        Returns list of generated strings.
        """
        traj = traj.to(next(self.parameters()).dtype)
        B    = traj.shape[0]
        device = next(self.parameters()).device

        q = self.get_sem_emb(traj)                                 # (B, 128)
        prefix_embs = self.prefix_proj(q).view(B, N_PREFIX, GPT2_DIM)  # (B,N,768)

        wte         = self.gpt2.transformer.wte
        prompt_embs = wte(prompt_ids.to(device)).unsqueeze(0).expand(B, -1, -1)
        combined    = torch.cat([prefix_embs.float(), prompt_embs.float()], dim=1)

        results = []
        for b in range(B):
            inp_embs = combined[b:b+1]                             # (1, N+P, 768)
            attn     = torch.ones(1, inp_embs.shape[1],
                                  dtype=torch.long, device=device)
            out = self.gpt2.generate(
                inputs_embeds=inp_embs,
                attention_mask=attn,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                repetition_penalty=1.20,
                length_penalty=0.60,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            # Strip prompt if GPT-2 echoes it
            if PROMPT.lower() in text.lower():
                idx  = text.lower().index(PROMPT.lower()) + len(PROMPT)
                text = text[idx:]
            results.append(text.strip())
        return results


# ── Warm-start ─────────────────────────────────────────────────────────────────

def load_warm_start(model):
    for path, label in [
        ("/v61/v61_phase1_best.pt", "V61"),
        ("/persist/v65_phase1_best.pt", "V65"),
        ("/persist/v100_best_ctc.pt", "V100"),
    ]:
        if Path(path).exists():
            print(f"Warm-starting VideoMAE from {label}: {path}")
            src = torch.load(path, map_location="cpu")
            dst = model.state_dict()
            prefixes = ("rasterizer.", "video_enc.", "ch_adapt.")
            n_ok = n_skip = 0
            for k, v in src.items():
                if not any(k.startswith(p) for p in prefixes): continue
                if k in dst and dst[k].shape == v.shape:
                    dst[k] = v.clone(); n_ok += 1
                else: n_skip += 1
            model.load_state_dict(dst, strict=False)
            print(f"  Loaded {n_ok} tensors, skipped {n_skip}")
            return

    print("No checkpoint — loading pretrained VideoMAE-base …")
    from transformers import VideoMAEModel as _VMAE
    src = _VMAE.from_pretrained("MCG-NJU/videomae-base")
    dst = model.video_enc.state_dict()
    n_ok = n_skip = 0
    for k, v in src.state_dict().items():
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v.clone(); n_ok += 1
        else: n_skip += 1
    model.video_enc.load_state_dict(dst, strict=False)
    print(f"  Transferred {n_ok} VideoMAE tensors")


# ── Optimizer ─────────────────────────────────────────────────────────────────

def make_optimizer(model, steps_per_epoch, epochs):
    layers = model.video_enc.encoder.layer
    n      = len(layers)
    groups = [
        {"params": list(model.video_enc.embeddings.parameters()),
         "lr": 5e-7,  "name": "vmae_emb"},
        {"params": list(layers[:n//2].parameters()),
         "lr": 1e-6,  "name": "vmae_0-5"},
        {"params": list(layers[n//2:n-2].parameters()),
         "lr": 5e-6,  "name": "vmae_6-9"},
        {"params": list(layers[n-2:].parameters()),
         "lr": 1e-5,  "name": "vmae_top"},
        {"params": (list(model.rasterizer.parameters()) +
                    list(model.ch_adapt.parameters())),
         "lr": 1e-4,  "name": "adapters"},
        {"params": (list(model.sem_proj.parameters()) +
                    list(model.txt_proj.parameters()) +
                    list(model.prefix_proj.parameters())),
         "lr": 2e-4,  "name": "projectors"},
    ]
    total_trainable = sum(p.numel() for g in groups for p in g["params"])
    gpt2_frozen     = sum(p.numel() for p in model.gpt2.parameters())
    print(f"Trainable: {total_trainable/1e6:.1f}M  |  GPT-2 frozen: {gpt2_frozen/1e6:.1f}M")
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[g["lr"] for g in groups],
        steps_per_epoch=max(steps_per_epoch // ACCUM_STEPS, 1),
        epochs=epochs, pct_start=0.10)
    return opt, sch


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_wer(pred, ref):
    import editdistance
    pw, rw = pred.strip().split(), ref.strip().split()
    if not rw: return 0.0 if not pw else 1.0
    return min(editdistance.eval(pw, rw) / len(rw), 2.0)


def compute_cer(pred, ref):
    import editdistance
    if not ref: return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, ref) / len(ref), 2.0)


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, val_loader, device, epoch, prompt_ids, tokenizer):
    model.eval()
    wers, cers, v_correct, v_total, c_correct, c_total = [], [], 0, 0, 0, 0
    print(f"\n── Val Epoch {epoch} ──")
    shown = 0

    with torch.no_grad():
        for batch in val_loader:
            eegs, texts, sberts, tgt_ids, tgt_mask = batch
            eegs = eegs.to(device, dtype=torch.bfloat16)

            generated = model.generate(eegs, prompt_ids, tokenizer,
                                       max_new_tokens=80, num_beams=5)

            for pred, ref in zip(generated, texts):
                ref_l = ref.lower().strip()
                pred_l = pred.lower().strip()
                wers.append(compute_wer(pred_l, ref_l))
                cers.append(compute_cer(pred_l, ref_l))
                for c in ref_l:
                    if c in VOWELS:     v_total += 1; v_correct += int(c in pred_l)
                    elif c in CONSONANTS: c_total += 1; c_correct += int(c in pred_l)

                if shown < 6:
                    print(f"  REF: '{ref[:80]}'")
                    print(f"  GEN: '{pred[:80]}'  WER={wers[-1]:.2f}")
                    shown += 1

    print(f"  WER:{np.mean(wers):.3f}  CER:{np.mean(cers):.3f}")
    print(f"  VowelRecall:{v_correct/max(v_total,1):.2f}  "
          f"ConsRecall:{c_correct/max(c_total,1):.2f}")
    model.train()
    return np.mean(wers)


# ── Training ───────────────────────────────────────────────────────────────────

def train(model, train_ds, val_ds, prompt_ids, tokenizer,
          epochs, ckpt_prefix, device):
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True,
        collate_fn=collate_fn, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False,
        collate_fn=collate_fn, num_workers=0)

    opt, sch = make_optimizer(model, len(train_loader), epochs)
    best_wer = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        tot = dict(lm=0., align=0.)
        opt.zero_grad()

        for step, batch in enumerate(train_loader):
            eegs, texts, sberts, tgt_ids, tgt_mask = batch
            eegs     = eegs.to(device, dtype=torch.bfloat16)
            sberts   = sberts.to(device)
            tgt_ids  = tgt_ids.to(device)
            tgt_mask = tgt_mask.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss_lm, loss_align = model(
                    eegs, sberts, prompt_ids, tgt_ids, tgt_mask)
                # GPT-2 is frozen → only lm/align flow through trainable params
                loss = (1.0 * loss_lm + 0.5 * loss_align) / ACCUM_STEPS

            loss.backward()
            tot["lm"]    += loss_lm.item()
            tot["align"] += loss_align.item()

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sch.step(); opt.zero_grad()

        if len(train_loader) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sch.step(); opt.zero_grad()

        n   = max(len(train_loader), 1)
        lr  = opt.param_groups[5]["lr"]   # projectors LR
        ptr = int(model.queue_ptr.item())
        print(f"Ep{epoch:2d} | LM:{tot['lm']/n:.3f} ALN:{tot['align']/n:.3f} "
              f"| LR:{lr:.1e} Queue:{min(ptr,QUEUE_SIZE)}/{QUEUE_SIZE}")

        if epoch % 5 == 0 or epoch == epochs:
            wer = validate(model, val_loader, device, epoch, prompt_ids, tokenizer)
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "wer":   wer,
            }, f"{ckpt_prefix}_ep{epoch}.pt")
            ckpt_vol.commit()

            if wer < best_wer:
                best_wer = wer
                torch.save(model.state_dict(), f"{ckpt_prefix}_best.pt")
                print(f"  ★ New best WER {best_wer:.3f}")


# ── Modal pipeline ─────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={"/data": data_vol, "/persist": ckpt_vol, "/v61": v61_vol},
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0,
                          initial_delay=10.0))
def run_training(epochs: int = 40):
    from transformers import GPT2Tokenizer
    from sentence_transformers import SentenceTransformer

    # ── Data ──────────────────────────────────────────────────────────────────
    download_zuco(Path("/data/EEG_Text"), data_vol)
    cache       = load_or_build_cache(Path("/data/EEG_Text"), data_vol)
    eegs, texts = cache["eegs"], cache["texts"]
    print(f"Dataset: {len(texts)} samples")

    # ── Tokenizer + prompt ────────────────────────────────────────────────────
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    prompt_ids = torch.tensor(
        tokenizer.encode(PROMPT, add_special_tokens=False), dtype=torch.long)
    print(f"Prompt tokens: {prompt_ids.tolist()} → {PROMPT!r}")

    # Tokenize all target texts (target only, NOT including prompt)
    def tok(text):
        enc = tokenizer(text.strip(),
                        max_length=MAX_TEXT_LEN,
                        truncation=True,
                        padding=False,
                        return_tensors="pt")
        return enc.input_ids[0], enc.attention_mask[0]

    print("Tokenizing targets …")
    input_ids_list  = []
    attn_mask_list  = []
    for t in texts:
        ids, mask = tok(t)
        input_ids_list.append(ids)
        attn_mask_list.append(mask)

    # ── SBERT embeddings ──────────────────────────────────────────────────────
    sbert_cache = Path("/persist/v100_sbert_embs.pt")
    if sbert_cache.exists():
        print("Loading cached SBERT embeddings …")
        sbert_all = torch.load(sbert_cache, weights_only=False)
    else:
        print("Computing SBERT embeddings …")
        sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        embs        = sbert_model.encode(texts, batch_size=64,
                                         show_progress_bar=True,
                                         normalize_embeddings=True)
        sbert_all   = torch.from_numpy(embs.astype(np.float32))
        torch.save(sbert_all, sbert_cache); ckpt_vol.commit()
        del sbert_model

    # ── Split ─────────────────────────────────────────────────────────────────
    np.random.seed(42)
    indices  = np.random.permutation(len(texts)).tolist()
    n_train  = int(0.9 * len(indices))
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    sbert_np = sbert_all.numpy()
    train_ds = EEGTextDataset(eegs, texts, sbert_np,
                              input_ids_list, attn_mask_list,
                              train_idx, augment=True)
    val_ds   = EEGTextDataset(eegs, texts, sbert_np,
                              input_ids_list, attn_mask_list,
                              val_idx, augment=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda")
    model  = EEGV600().to(torch.bfloat16).to(device)
    # GPT-2 stays in float32 for stability; cast only trainable parts to bf16
    model.gpt2 = model.gpt2.float().to(device)

    load_warm_start(model)

    prompt_ids = prompt_ids.to(device)

    # ── Train ─────────────────────────────────────────────────────────────────
    train(model, train_ds, val_ds, prompt_ids, tokenizer,
          epochs=epochs, ckpt_prefix="/persist/v600", device=device)

    print("\n[V600] Training complete.")


@app.local_entrypoint()
def main():
    run_training.remote(epochs=40)
