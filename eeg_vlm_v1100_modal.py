"""
V1100: Stage 2 Relaunch — Frozen VideoMAE + Fresh eeg_to_t5 + Strong LM
=========================================================================
ROOT CAUSE ANALYSIS — Why V1000 Stage 2 failed:
  1. OneCycleLR exhausted: By epoch 16, scheduler completed 15/50 = 30% of cosine decay
     at lm_weight=0. Stage 2 inherits a dying LR → eeg_to_t5 can barely update.
  2. eeg_to_t5 drifted: Stage 1 shaped it as alignment encoder (InfoNCE + rel_dist),
     NOT as a T5 cross-attention conditioner. Wrong optimization basin for LM.
  3. lm_weight=0.15 too weak: Can't escape Stage 1 momentum with decaying LR.

V1100 DESIGN:
  Load V1000 ep15 (best Stage 1: Diversity=51.7% → most discriminative VideoMAE ever)
  Freeze VideoMAE 100% — Stage 1's discriminative features are preserved exactly.
  Reinitialize eeg_to_t5 randomly — clean slate, no Stage 1 alignment bias.
  Fresh OneCycleLR — optimizer only covers Stage 2 epochs (no burnout).
  High LM weight (1.0) — strongly train eeg_to_t5 to condition T5 from discriminative features.
  Keep align (0.5) + aux heads — semantic regularization, prevent arbitrary drift.
  No div/rel loss — VideoMAE is frozen, pointless to push its features.

WHY THIS WORKS:
  Stage 1 (V1000) gave us: VideoMAE features that vary significantly per EEG input (51.7% div)
  Stage 2 (V1100) goal:    Train eeg_to_t5 to map THOSE diverse features → T5 conditioning
                            such that T5 generates DIFFERENT text for different EEG.

  With VideoMAE frozen, the diverse enc features are fixed inputs.
  eeg_to_t5 (random init) has a clean surface to learn the mapping.
  LM=1.0 is the same weight that made V700 work (30.2% diversity at ep5).
  The key difference from V700: VideoMAE is NOW discriminative (was collapsing in V700).
"""

from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Modal ──────────────────────────────────────────────────────────────────────

app       = modal.App("eeg-vlm-v1100")
ckpt_vol  = modal.Volume.from_name("bt-checkpoints-v1100", create_if_missing=True)
v61_vol   = modal.Volume.from_name("bt-checkpoints-v61",   create_if_missing=False)
v600_vol  = modal.Volume.from_name("bt-checkpoints-v600",  create_if_missing=False)
v700_vol  = modal.Volume.from_name("bt-checkpoints-v700",  create_if_missing=False)
v1000_vol = modal.Volume.from_name("bt-checkpoints-v1000", create_if_missing=False)
data_vol  = modal.Volume.from_name("mindvoice-data",        create_if_missing=True)


def _download_models():
    from transformers import VideoMAEModel, T5ForConditionalGeneration, T5Tokenizer
    from sentence_transformers import SentenceTransformer
    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    T5Tokenizer.from_pretrained("google/flan-t5-base")
    SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "accelerate", "osfclient", "editdistance", "openneuro-py",
        "sentence-transformers", "vaderSentiment",
    ])
    .run_function(_download_models)
)

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_EEG_BANDS = 6
ACCUM_STEPS   = 8
QUEUE_SIZE    = 4096
SEM_DIM       = 128
SBERT_DIM     = 768
T5_DIM        = 768
EEG_SEQ_LEN   = 256
MAX_TEXT_LEN  = 80

# Loss weights for Stage 2 (no stage 1 here — pure generation fine-tuning)
W_LM    = 1.0   # strong — primary objective
W_ALIGN = 0.5   # keeps EEG semantically grounded
W_SENT  = 0.20  # auxiliary
W_LEN   = 0.10  # auxiliary

# ── EEG preprocessing (identical pipeline, reuses V600 cache) ──────────────────

def download_zuco(base_path, vol):
    import subprocess
    zuco_v1 = base_path / "ZuCo_v1"; zuco_v2 = base_path / "ZuCo_v2"
    if (zuco_v1.exists() and any(zuco_v1.rglob("*.mat")) and
            zuco_v2.exists() and any(zuco_v2.rglob("*.mat"))):
        print("ZuCo already downloaded."); return
    print("Downloading ZuCo via osf …")
    for cmd in [["osf","-p","q3zws","clone",str(zuco_v1)],
                ["osf","-p","2urht","clone",str(zuco_v2)]]:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0: print(f"osf warn: {r.stderr[:200]}")
    vol.commit()


def load_mat_any(path):
    try:
        import scipy.io as sio
        raw = sio.loadmat(str(path), simplify_cells=True)
    except Exception:
        try:
            import h5py
            with h5py.File(str(path), 'r') as f:
                raw = {k: np.array(f[k]) for k in f.keys()}
        except Exception:
            return []
    results = []
    for key in raw:
        if not isinstance(raw[key], dict): continue
        block = raw[key]
        eeg_keys  = [k for k in block if 'rawData' in k or k == 'rawData']
        word_keys = [k for k in block if 'word' in k.lower() or 'content' in k.lower()]
        if not eeg_keys or not word_keys: continue
        eeg   = np.array(block[eeg_keys[0]], dtype=np.float32)
        words = block[word_keys[0]]
        if isinstance(words, np.ndarray): words = [str(w) for w in words.flatten() if w]
        elif isinstance(words, str):      words = [words]
        else:                             words = [str(words)]
        text = " ".join(words).strip()
        if len(text) < 5: continue
        results.append((eeg, text))
    return results


def normalize_eeg(eeg, target_ch=64, target_t=1024):
    if eeg.ndim == 3: eeg = eeg.mean(0)
    if eeg.ndim != 2: return None
    if eeg.shape[0] < eeg.shape[1]: eeg = eeg.T
    T, C = eeg.shape
    if C > target_ch: eeg = eeg[:, :target_ch]
    elif C < target_ch:
        eeg = np.concatenate([eeg, np.zeros((T, target_ch-C), dtype=np.float32)], axis=1)
    if T > target_t: eeg = eeg[:target_t]
    elif T < target_t:
        eeg = np.concatenate([eeg, np.zeros((target_t-T, target_ch), dtype=np.float32)], axis=0)
    bands = []
    try:
        import mne
        info = mne.create_info(target_ch, 500., 'eeg')
        raw  = mne.io.RawArray(eeg.T, info, verbose=False)
        raw.filter(0.5, 40., fir_design='firwin', verbose=False)
        bands.append(raw.get_data().T)
        for lo, hi in [(8,13),(13,30),(30,45),(1,4),(30,45)]:
            filt = raw.copy().filter(lo, hi, fir_design='firwin', verbose=False)
            d = filt.get_data().T
            if hi == 30 and lo == 1:  d = np.diff(d, axis=0, prepend=d[:1])
            elif hi == 45 and lo == 30 and len(bands) > 4:
                d = np.diff(d, axis=0, prepend=d[:1])
            bands.append(d)
        result = np.concatenate(bands, axis=1).astype(np.float32)
    except Exception:
        result = np.tile(eeg, (1, NUM_EEG_BANDS)).astype(np.float32)
    mu  = result.mean(0, keepdims=True)
    sig = result.std(0, keepdims=True) + 1e-6
    return ((result - mu) / sig).astype(np.float32)


def load_or_build_cache(base_path, vol):
    for cp in ["/persist/v65_data_cache.pt", "/v600_ckpt/v65_data_cache.pt"]:
        if Path(cp).exists():
            print(f"EEG cache hit at {cp} …")
            return torch.load(cp, weights_only=False)
    cache_path = Path("/persist/v65_data_cache.pt")
    print("Building EEG cache …")
    eegs, texts, skipped = [], [], 0
    for zp in [base_path/"ZuCo_v1", base_path/"ZuCo_v2"]:
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


def _augment(eeg):
    T, C = eeg.shape
    shift = np.random.randint(-64, 65)
    if shift:
        eeg = np.roll(eeg, shift, 0)
        if shift > 0: eeg[:shift] = 0
        else:         eeg[shift:] = 0
    eeg *= np.random.uniform(0.88, 1.12, (1, C)).astype(np.float32)
    eeg += np.random.randn(*eeg.shape).astype(np.float32) * 0.04
    eeg[:, np.random.choice(C, max(1, int(C*0.10)), replace=False)] = 0
    for _ in range(3):
        span  = np.random.randint(16, 64)
        start = np.random.randint(0, max(1, T-span))
        eeg[start:start+span] = 0
    return eeg


# ── Dataset ────────────────────────────────────────────────────────────────────

def build_labels(texts, t5_tokenizer):
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        sentiments = np.array(
            [1 if sia.polarity_scores(t)['compound'] >= 0.05 else 0 for t in texts],
            dtype=np.float32)
    except Exception:
        sentiments = np.zeros(len(texts), dtype=np.float32)
    lengths = np.array(
        [np.log1p(len(t5_tokenizer(t, return_tensors='pt').input_ids[0]))
         for t in texts], dtype=np.float32)
    return sentiments, lengths


class EEGTextDataset(torch.utils.data.Dataset):
    def __init__(self, eegs, texts, sbert_np, input_ids_list, attn_mask_list,
                 sentiments, lengths, indices, augment=False):
        self.eegs=eegs; self.texts=texts; self.sbert_np=sbert_np
        self.input_ids_list=input_ids_list; self.attn_mask_list=attn_mask_list
        self.sentiments=sentiments; self.lengths=lengths
        self.indices=indices; self.augment=augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        gi  = self.indices[i]
        eeg = self.eegs[gi].copy()
        if self.augment: eeg = _augment(eeg)
        return (eeg, self.texts[gi], self.sbert_np[gi],
                self.input_ids_list[gi], self.attn_mask_list[gi],
                self.sentiments[gi], self.lengths[gi])


def collate_fn(batch):
    eegs, texts, sberts, ids_list, mask_list, sents, lens = zip(*batch)
    eegs   = torch.from_numpy(np.stack(eegs))
    sberts = torch.from_numpy(np.stack(sberts))
    sents  = torch.tensor(sents, dtype=torch.float32)
    lens   = torch.tensor(lens,  dtype=torch.float32)
    max_len = max(ids.shape[0] for ids in ids_list)
    B = len(batch)
    padded_ids  = torch.full((B, max_len), -100, dtype=torch.long)
    padded_attn = torch.zeros(B, max_len, dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
        L = ids.shape[0]
        padded_ids[i,:L] = ids; padded_attn[i,:L] = mask
    return eegs, list(texts), sberts, padded_ids, padded_attn, sents, lens


# ── Model ──────────────────────────────────────────────────────────────────────

class MultiScaleRasterizer(nn.Module):
    def __init__(self, size=64, n_electrodes=64, n_bands=NUM_EEG_BANDS):
        super().__init__()
        self.n_bands = n_bands; self.n_electrodes = n_electrodes
        angles = torch.linspace(0, 2*np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(
            torch.linspace(-1,1,size), torch.linspace(-1,1,size), indexing='ij')
        px = gx.flatten().unsqueeze(1); py = gy.flatten().unsqueeze(1)
        ex = pos_2d[:,0].unsqueeze(0); ey = pos_2d[:,1].unsqueeze(0)
        dist = torch.sqrt((px-ex)**2 + (py-ey)**2)
        w    = 1.0 / (dist + 1e-4)**2.0
        r    = torch.sqrt(px**2 + py**2)
        w[(r>1.1).squeeze(1),:] = 0.0
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)

    def forward(self, x):
        B, T, _ = x.shape
        xs = x.reshape(B*T, self.n_bands, self.n_electrodes)
        return (xs @ self.W.T.to(x.device, x.dtype)).reshape(B, T, self.n_bands, 64, 64)


class ChannelAdapter(nn.Module):
    def __init__(self, in_ch=NUM_EEG_BANDS):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3, 1)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        return self.conv(imgs.view(B*T,C,H,W)).view(B,T,3,H,W)


class EEGV1100(nn.Module):
    """
    V1100: VideoMAE FROZEN (from V1000 ep15, 51.7% diversity).
    eeg_to_t5 RANDOMLY INITIALIZED — fresh mapping from discriminative features → T5.
    Only trainable: eeg_to_t5, sem_proj, txt_proj, ch_adapt, aux heads.
    VideoMAE + T5: completely frozen.
    """
    def __init__(self):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel, T5ForConditionalGeneration

        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=1024, tubelet_size=4, hidden_size=768)
        self.rasterizer = MultiScaleRasterizer()
        self.ch_adapt   = ChannelAdapter()
        self.video_enc  = VideoMAEModel(v_cfg)

        self.sem_proj = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768,256), nn.GELU(), nn.Linear(256,SEM_DIM))
        self.txt_proj = nn.Sequential(
            nn.Linear(SBERT_DIM,256), nn.GELU(), nn.Linear(256,SEM_DIM))

        self.register_buffer("queue",
            F.normalize(torch.randn(QUEUE_SIZE, SEM_DIM), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.sentiment_head = nn.Linear(SEM_DIM, 1)
        self.length_head    = nn.Linear(SEM_DIM, 1)

        # eeg_to_t5: randomly initialized — learns fresh mapping from
        # V1000's discriminative VideoMAE features → T5 cross-attention K/V
        self.eeg_to_t5 = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, T5_DIM), nn.GELU(),
            nn.Linear(T5_DIM, T5_DIM))

        self.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        for p in self.t5.parameters():
            p.requires_grad = False

    def freeze_videomae(self):
        """Freeze VideoMAE completely — preserve Stage 1 discriminative features."""
        for p in self.video_enc.parameters():
            p.requires_grad = False
        for p in self.rasterizer.parameters():
            p.requires_grad = False
        # ch_adapt stays trainable — small adapter on top of frozen rasterizer

    def encode(self, traj):
        imgs  = self.rasterizer(traj)
        imgs  = self.ch_adapt(imgs)
        with torch.no_grad():  # VideoMAE frozen — no gradient through it
            v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(2)   # (B,256,768)

    def get_eeg_enc(self, traj):
        v_seq  = self.encode(traj)                                # (B,256,768) frozen
        v_mean = v_seq.mean(1)                                    # (B,768)
        q      = F.normalize(self.sem_proj(v_mean), dim=-1)      # (B,128)
        enc    = self.eeg_to_t5(v_seq)                            # (B,256,768) trainable
        return q, enc

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

    def forward(self, traj, sbert_embs, target_ids, target_mask,
                sentiment_labels, length_labels):
        from transformers.modeling_outputs import BaseModelOutput

        B, device = traj.shape[0], traj.device
        dtype = next(p for p in self.parameters() if p.requires_grad).dtype
        traj  = traj.to(device, dtype)

        q, enc = self.get_eeg_enc(traj)

        # InfoNCE
        k = F.normalize(self.txt_proj(sbert_embs.to(device, dtype)), dim=-1)
        pos        = (q * k).sum(-1, keepdim=True) / 0.07
        negs       = q @ self.queue.T.detach() / 0.07
        loss_align = F.cross_entropy(
            torch.cat([pos, negs], dim=1),
            torch.zeros(B, dtype=torch.long, device=device))
        self._enqueue(k)

        # Auxiliary
        loss_sent = F.binary_cross_entropy_with_logits(
            self.sentiment_head(q).squeeze(-1), sentiment_labels.to(device))
        loss_len  = F.mse_loss(
            self.length_head(q).squeeze(-1), length_labels.to(device))

        # T5 LM (main objective)
        enc_hs   = enc.to(dtype)
        enc_attn = torch.ones(B, EEG_SEQ_LEN, dtype=torch.long, device=device)
        labels   = target_ids.to(device).clone()
        labels[target_mask.to(device) == 0] = -100
        enc_out  = BaseModelOutput(last_hidden_state=enc_hs)
        out      = self.t5(encoder_outputs=enc_out, attention_mask=enc_attn, labels=labels)
        loss_lm  = out.loss

        return loss_lm, loss_align, loss_sent, loss_len

    @torch.no_grad()
    def generate(self, traj, tokenizer, max_new_tokens=80, num_beams=5):
        from transformers.modeling_outputs import BaseModelOutput

        traj   = traj.to(next(p for p in self.parameters() if p.requires_grad).dtype)
        device = next(p for p in self.parameters() if p.requires_grad).device
        B      = traj.shape[0]

        _, enc = self.get_eeg_enc(traj)
        enc_hs   = enc
        enc_attn = torch.ones(B, EEG_SEQ_LEN, dtype=torch.long, device=device)

        results = []
        for b in range(B):
            eo  = BaseModelOutput(last_hidden_state=enc_hs[b:b+1])
            out = self.t5.generate(
                encoder_outputs=eo,
                attention_mask=enc_attn[b:b+1],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                repetition_penalty=1.20,
                length_penalty=0.70,
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
            results.append(tokenizer.decode(out[0], skip_special_tokens=True).strip())
        return results


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def load_v1000_stage1(model):
    """
    Load V1000 ep15 (best Stage 1 — 51.7% diversity).
    VideoMAE weights: loaded and then FROZEN.
    eeg_to_t5 weights: NOT loaded — stays randomly initialized.
    sem_proj, txt_proj, aux heads: loaded from V1000 ep15.
    """
    import glob

    v1000_dir = Path("/v1000_ckpt")
    # Prefer ep15 (last Stage 1, highest diversity)
    candidates = sorted(glob.glob(str(v1000_dir / "v1000_ep15_S1*.pt")))
    if not candidates:
        candidates = sorted(glob.glob(str(v1000_dir / "v1000_ep*_S1*.pt")))
    if not candidates:
        candidates = sorted(glob.glob(str(v1000_dir / "v1000_ep*.pt")))

    if not candidates:
        print("No V1000 checkpoint found — training from scratch with V61 VideoMAE only.")
        _load_v61_videomae(model)
        return

    src  = candidates[-1]   # latest = ep15 if sorted by name
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd   = ckpt.get("model_state", ckpt)
    own  = model.state_dict()

    # Separate eeg_to_t5 keys — we DO NOT load these (keep random init)
    skip_prefixes = ("eeg_to_t5.",)

    loaded, skipped_shape, skipped_prefix = [], [], []
    for k, v in sd.items():
        if any(k.startswith(p) for p in skip_prefixes):
            skipped_prefix.append(k)
            continue
        if k in own and own[k].shape == v.shape:
            own[k] = v; loaded.append(k)
        else:
            skipped_shape.append(k)

    model.load_state_dict(own, strict=False)
    print(f"Loaded from V1000 ep15: {len(loaded)} params")
    print(f"  eeg_to_t5 KEPT RANDOM: {len(skipped_prefix)} keys (fresh T5 conditioner)")
    print(f"  Shape mismatch skipped: {len(skipped_shape)}")
    print(f"  Source: {src}")


def _load_v61_videomae(model):
    import glob
    ckpts = sorted(glob.glob("/v61_ckpt/*.pt"))
    if not ckpts:
        print("No V61 checkpoint either — pure random init."); return
    ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    sd   = ckpt.get("model_state", ckpt)
    own  = model.state_dict()
    loaded = [k for k, v in sd.items()
              if k in own and own[k].shape == v.shape]
    for k in loaded: own[k] = sd[k]
    model.load_state_dict(own, strict=False)
    print(f"Fallback: V61 VideoMAE only, {len(loaded)} params.")


# ── Optimizer (adapters only, fresh schedule) ──────────────────────────────────

def make_optimizer(model, steps_per_epoch, epochs):
    """
    VideoMAE is frozen — only optimize adapters + eeg_to_t5.
    Fresh OneCycleLR for the full Stage 2 run.
    High max_lr for eeg_to_t5 (random init needs fast learning).
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    adapt_p    = []   # sem_proj, txt_proj, aux heads, ch_adapt
    eeg_t5_p   = []   # eeg_to_t5 (random init — needs higher LR)

    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("eeg_to_t5."):
            eeg_t5_p.append(p)
        else:
            adapt_p.append(p)

    param_groups = [
        {"params": adapt_p,  "lr": 1e-4},   # alignment heads
        {"params": eeg_t5_p, "lr": 3e-4},   # eeg_to_t5: higher LR, random init
    ]
    opt   = AdamW(param_groups, weight_decay=1e-4)
    sched = OneCycleLR(opt,
        max_lr=[1e-4, 3e-4],
        total_steps=steps_per_epoch * epochs,
        pct_start=0.10, anneal_strategy="cos")
    return opt, sched


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_wer(pred, ref):
    import editdistance
    p_words = pred.split(); r_words = ref.split()
    if not r_words: return 1.0
    return editdistance.eval(p_words, r_words) / len(r_words)


def compute_cer(pred, ref):
    import editdistance
    if not ref: return 1.0
    return editdistance.eval(pred, ref) / len(ref)


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(model, val_loader, device, epoch, tokenizer):
    model.eval()
    wers, cers, all_preds, all_refs = [], [], [], []

    for batch in val_loader:
        eegs, texts, sberts, tgt_ids, tgt_mask, sents, lens = batch
        with torch.no_grad():
            preds = model.generate(eegs.to(device), tokenizer, max_new_tokens=80)
        for pred, ref in zip(preds, texts):
            wers.append(compute_wer(pred.lower(), ref.lower()))
            cers.append(compute_cer(pred.lower(), ref.lower()))
            all_preds.append(pred); all_refs.append(ref)

    mean_wer  = float(np.mean(wers))
    mean_cer  = float(np.mean(cers))
    diversity = len(set(all_preds)) / max(1, len(all_preds))

    def cons_recall(preds, refs):
        hits=0; total=0
        for p,r in zip(preds,refs):
            rw=set(r.lower().split()); pw=set(p.lower().split())
            hits+=len(rw&pw); total+=max(1,len(rw))
        return hits/total
    cr = cons_recall(all_preds, all_refs)

    print(f"\n[Epoch {epoch}] WER={mean_wer:.3f}  CER={mean_cer:.3f}  "
          f"Diversity={diversity:.3f}  ConsRecall={cr:.3f}")
    print("── Sample outputs ──")
    for p, r in zip(all_preds[:6], all_refs[:6]):
        print(f"  REF : {r[:80]}")
        print(f"  PRED: {p[:80]}")
        print()

    model.train()
    return mean_wer, diversity, cr


# ── Training loop ──────────────────────────────────────────────────────────────

def train(model, train_ds, val_ds, tokenizer, device, epochs=35):
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    steps_per_epoch = max(1, len(train_loader) // ACCUM_STEPS)
    opt, sched      = make_optimizer(model, steps_per_epoch, epochs)

    ckpt_dir       = Path("/persist"); ckpt_dir.mkdir(exist_ok=True)
    best_score     = -1.0
    best_wer       = 1e9
    low_div_streak = 0
    EARLY_STOP_DIV    = 0.08
    EARLY_STOP_STREAK = 5

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0; n_accum = 0

        for step, batch in enumerate(train_loader, 1):
            eegs, texts, sberts, tgt_ids, tgt_mask, sents, lens = batch
            eegs = eegs.to(device, torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lm, align, sent_l, len_l = model(
                    eegs, sberts, tgt_ids, tgt_mask, sents, lens)
                total = W_LM*lm + W_ALIGN*align + W_SENT*sent_l + W_LEN*len_l
                loss  = total / ACCUM_STEPS

            loss.backward()
            n_accum += 1

            if n_accum % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                sched.step()
                total_loss += total.item()

                if (n_accum // ACCUM_STEPS) % 10 == 0:
                    print(f"  ep={epoch} step={n_accum//ACCUM_STEPS} "
                          f"lm={lm.item():.3f} align={align.item():.3f} "
                          f"sent={sent_l.item():.3f}")

        print(f"\nEpoch {epoch}: avg_loss={total_loss/max(1,n_accum//ACCUM_STEPS):.3f}")

        wer, div, cr = validate(model, val_loader, device, epoch, tokenizer)
        score = div * cr

        if score > best_score or wer < best_wer:
            best_score = max(best_score, score)
            best_wer   = min(best_wer, wer)
            ckpt_path  = ckpt_dir / f"v1100_ep{epoch}_div{div:.3f}_cr{cr:.3f}_wer{wer:.3f}.pt"
            torch.save({
                "epoch": epoch, "wer": wer, "diversity": div, "cr": cr,
                "model_state": model.state_dict(),
                "opt_state":   opt.state_dict(),
            }, str(ckpt_path))
            ckpt_vol.commit()
            print(f"  ✓ Best checkpoint: {ckpt_path.name}")

        if div < EARLY_STOP_DIV:
            low_div_streak += 1
            print(f"  ⚠ Low diversity ({div:.3f}) streak={low_div_streak}/{EARLY_STOP_STREAK}")
            if low_div_streak >= EARLY_STOP_STREAK:
                print(f"  ✗ Early stop: div < {EARLY_STOP_DIV} for {EARLY_STOP_STREAK} epochs")
                break
        else:
            low_div_streak = 0


# ── Modal entrypoint ───────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={
        "/persist":    ckpt_vol,
        "/v61_ckpt":   v61_vol,
        "/v600_ckpt":  v600_vol,
        "/v700_ckpt":  v700_vol,
        "/v1000_ckpt": v1000_vol,
        "/data":       data_vol,
    },
)
def run_training(epochs: int = 35):
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import T5Tokenizer

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    base_path = Path("/data")
    eeg_cache_exists = any(Path(p).exists()
        for p in ["/persist/v65_data_cache.pt", "/v600_ckpt/v65_data_cache.pt"])
    if not eeg_cache_exists:
        download_zuco(base_path, data_vol)
    else:
        print("EEG cache found — skipping ZuCo download.")
    cache = load_or_build_cache(base_path, data_vol)

    eegs  = cache["eegs"]; texts = cache["texts"]; N = len(texts)
    print(f"Dataset: {N} samples")

    sbert_cache = None
    for sp in ["/persist/v100_sbert_embs.pt", "/v600_ckpt/v100_sbert_embs.pt"]:
        if Path(sp).exists(): sbert_cache = Path(sp); break
    if sbert_cache:
        sbert_np = torch.load(sbert_cache, weights_only=False)
        print(f"SBERT cache hit at {sbert_cache}")
    else:
        print("Computing SBERT embeddings …")
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        sbert_np    = sbert_model.encode(texts, batch_size=64, show_progress_bar=True,
                                         convert_to_numpy=True)
        sbert_cache = Path("/persist/v100_sbert_embs.pt")
        torch.save(sbert_np, sbert_cache); ckpt_vol.commit()
    print(f"SBERT: {sbert_np.shape}")

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    print("Tokenizing targets …")
    input_ids_list, attn_mask_list = [], []
    for text in texts:
        enc = tokenizer(text, max_length=MAX_TEXT_LEN, truncation=True,
                        padding=False, return_tensors='pt')
        input_ids_list.append(enc.input_ids[0])
        attn_mask_list.append(enc.attention_mask[0])

    print("Computing auxiliary labels …")
    sentiments, lengths = build_labels(texts, tokenizer)

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(N)
    split = int(0.85 * N)
    train_idx = idx[:split].tolist(); val_idx = idx[split:].tolist()

    train_ds = EEGTextDataset(eegs, texts, sbert_np, input_ids_list, attn_mask_list,
                              sentiments, lengths, train_idx, augment=True)
    val_ds   = EEGTextDataset(eegs, texts, sbert_np, input_ids_list, attn_mask_list,
                              sentiments, lengths, val_idx, augment=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    print("Building model …")
    model = EEGV1100().to(device)
    load_v1000_stage1(model)   # loads VideoMAE from V1000 ep15, eeg_to_t5 stays random
    model.freeze_videomae()    # freeze VideoMAE completely
    model = model.to(torch.bfloat16)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable/1e6:.1f}M  Frozen: {frozen/1e6:.1f}M")
    print(f"Strategy: VideoMAE FROZEN (Stage 1 features preserved) + fresh eeg_to_t5")
    print(f"Loss weights: LM={W_LM} align={W_ALIGN} sent={W_SENT} len={W_LEN}")

    train(model, train_ds, val_ds, tokenizer, device, epochs=epochs)
    print("Training complete.")


@app.local_entrypoint()
def main():
    run_training.remote(epochs=35)
