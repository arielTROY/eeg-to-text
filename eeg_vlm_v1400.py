"""
V1400: VideoMAE → Q-Former → T5-base  (Open-Vocabulary Brain BCI)
==================================================================
KEY IMPROVEMENTS over V1200:

Stage 1 (VideoMAE discriminative pretraining) — improved:
  + Subject-adversarial training (GRL) → subject-invariant EEG features
  + Cross-subject multi-positive InfoNCE → same sentence across subjects = positives
  + 25 epochs (vs 15) with cosine LR warm restarts
  + Larger queue (8192 vs 4096) for harder negatives

Stage 2 (EEG → text) — new architecture:
  OLD (V1200): VideoMAE → eeg_to_t5 (768→768 single linear) → T5
    Problem: too shallow, T5 ignores weak conditioning, falls back to LM prior.
  NEW (V1400): VideoMAE → QFormerBridge (32 query tokens, 4-layer cross-attn) → T5
    Fix: 32 learnable queries each attend to full VideoMAE output sequence.
    Q-Former cannot collapse: 32 diverse queries extract complementary EEG features.
    + Relational distillation on qformer_mean (same critical anti-collapse from V1200)

Target: CR 11.2% → 25%+, Diversity 100%, WER < 1.0

Usage:
  python eeg_vlm_v1400.py --stage 1 --epochs 25
  python eeg_vlm_v1400.py --stage 1 --dry-run
  python eeg_vlm_v1400.py --stage 2 --epochs 40
  python eeg_vlm_v1400.py --stage 2 --epochs 40 --s1-ckpt ./ckpts/v1400/s1_best.pt
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_EEG_BANDS  = 6
ACCUM_STEPS    = 8
QUEUE_SIZE     = 8192
SEM_DIM        = 128
SBERT_DIM      = 768
T5_DIM         = 768
EEG_SEQ_LEN    = 256   # VideoMAE output sequence length (256 temporal positions)
QFORMER_TOKENS = 32    # Q-Former query tokens → fed to T5 cross-attention
MAX_TEXT_LEN   = 80
N_SUBJECTS     = 20    # upper bound for ZuCo subject count

# Stage 1 loss weights
S1_W_ALIGN = 3.0
S1_W_DIV   = 2.0
S1_W_ADV   = 0.1
S1_W_SENT  = 0.30
S1_W_LEN   = 0.10

# Stage 2 loss weights
S2_W_LM    = 1.0
S2_W_ALIGN = 0.5
S2_W_REL   = 0.5
S2_W_SENT  = 0.15
S2_W_LEN   = 0.05

DATA_DIR  = Path("./data")
CKPT_DIR  = Path("./ckpts/v1400")

# ── EEG Preprocessing ─────────────────────────────────────────────────────────

def load_mat_any(path):
    """Load ZuCo .mat file, returns list of (eeg_array, text_string) pairs."""
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
        if not isinstance(raw[key], dict):
            continue
        block = raw[key]
        eeg_keys  = [k for k in block if 'rawData' in k or k == 'rawData']
        word_keys = [k for k in block if 'word' in k.lower() or 'content' in k.lower()]
        if not eeg_keys or not word_keys:
            continue
        eeg   = np.array(block[eeg_keys[0]], dtype=np.float32)
        words = block[word_keys[0]]
        if isinstance(words, np.ndarray):
            words = [str(w) for w in words.flatten() if w]
        elif isinstance(words, str):
            words = [words]
        else:
            words = [str(words)]
        text = " ".join(words).strip()
        if len(text) < 5:
            continue
        results.append((eeg, text))
    return results


def normalize_eeg(eeg, target_ch=64, target_t=1024):
    """Normalize EEG to (target_t, target_ch * NUM_EEG_BANDS) float32 array."""
    if eeg.ndim == 3:
        eeg = eeg.mean(0)
    if eeg.ndim != 2:
        return None
    if eeg.shape[0] < eeg.shape[1]:
        eeg = eeg.T
    T, C = eeg.shape
    if C > target_ch:
        eeg = eeg[:, :target_ch]
    elif C < target_ch:
        eeg = np.concatenate([eeg, np.zeros((T, target_ch - C), dtype=np.float32)], axis=1)
    if T > target_t:
        eeg = eeg[:target_t]
    elif T < target_t:
        eeg = np.concatenate([eeg, np.zeros((target_t - T, target_ch), dtype=np.float32)], axis=0)
    bands = []
    try:
        import mne
        info = mne.create_info(target_ch, 500., 'eeg')
        raw  = mne.io.RawArray(eeg.T, info, verbose=False)
        raw.filter(0.5, 40., fir_design='firwin', verbose=False)
        bands.append(raw.get_data().T)
        for lo, hi in [(8, 13), (13, 30), (30, 45), (1, 4), (30, 45)]:
            filt = raw.copy().filter(lo, hi, fir_design='firwin', verbose=False)
            d = filt.get_data().T
            if hi == 30 and lo == 1:
                d = np.diff(d, axis=0, prepend=d[:1])
            elif hi == 45 and lo == 30 and len(bands) > 4:
                d = np.diff(d, axis=0, prepend=d[:1])
            bands.append(d)
        result = np.concatenate(bands, axis=1).astype(np.float32)
    except Exception:
        result = np.tile(eeg, (1, NUM_EEG_BANDS)).astype(np.float32)
    mu  = result.mean(0, keepdims=True)
    sig = result.std(0, keepdims=True) + 1e-6
    return ((result - mu) / sig).astype(np.float32)


def _augment(eeg):
    """Standard EEG augmentation: temporal shift, amplitude scaling, noise, dropouts."""
    T, C = eeg.shape
    shift = np.random.randint(-64, 65)
    if shift:
        eeg = np.roll(eeg, shift, 0)
        if shift > 0:
            eeg[:shift] = 0
        else:
            eeg[shift:] = 0
    eeg *= np.random.uniform(0.88, 1.12, (1, C)).astype(np.float32)
    eeg += np.random.randn(*eeg.shape).astype(np.float32) * 0.04
    eeg[:, np.random.choice(C, max(1, int(C * 0.10)), replace=False)] = 0
    for _ in range(3):
        span  = np.random.randint(16, 64)
        start = np.random.randint(0, max(1, T - span))
        eeg[start:start + span] = 0
    return eeg


def load_or_build_cache(base_path=DATA_DIR):
    """Load or build EEG+text cache from ZuCo v1+v2 .mat files."""
    cache_path = CKPT_DIR / "eeg_cache.pt"
    if cache_path.exists():
        print(f"EEG cache hit: {cache_path}")
        return torch.load(cache_path, weights_only=False)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building EEG cache from ZuCo .mat files...")
    eegs, texts, subject_ids = [], [], []
    skipped = 0

    # Build subject ID mapping from file paths
    subject_map = {}
    next_subj_id = [0]

    def get_subject_id(mat_path):
        # Extract subject identifier from path (e.g. ZuCo_v1/ZAB/... → "ZAB")
        parts = Path(mat_path).parts
        # Look for the subject folder (usually a 3-letter code or name)
        for i, part in enumerate(parts):
            if 'ZuCo' in part and i + 1 < len(parts):
                subj_name = parts[i + 1]
                if subj_name not in subject_map:
                    subject_map[subj_name] = next_subj_id[0]
                    next_subj_id[0] += 1
                return subject_map[subj_name]
        return 0

    for zp in [base_path / "ZuCo_v1", base_path / "ZuCo_v2"]:
        if not zp.exists():
            print(f"  WARNING: {zp} not found. Run setup_server.sh first.")
            continue
        for mat in sorted(zp.rglob("*.mat")):
            sid = get_subject_id(mat)
            for raw, text in load_mat_any(mat):
                if not text or "placeholder" in text.lower():
                    skipped += 1
                    continue
                n = normalize_eeg(raw)
                if n is None:
                    skipped += 1
                    continue
                eegs.append(n)
                texts.append(text)
                subject_ids.append(sid)

    if not eegs:
        raise RuntimeError("No EEG samples found. Check data directory and ZuCo download.")

    # Build sentence_ids: map each unique text to an integer ID
    unique_texts = list(dict.fromkeys(texts))  # preserves order, deduplicates
    text_to_sid  = {t: i for i, t in enumerate(unique_texts)}
    sentence_ids = [text_to_sid[t] for t in texts]

    cache = {
        "eegs":        np.stack(eegs).astype(np.float32),
        "texts":       texts,
        "subject_ids": np.array(subject_ids, dtype=np.int64),
        "sentence_ids": np.array(sentence_ids, dtype=np.int64),
        "n_subjects":  next_subj_id[0],
    }
    torch.save(cache, cache_path)
    print(f"Cache: {len(eegs)} samples, {skipped} skipped, "
          f"{len(unique_texts)} unique sentences, "
          f"{next_subj_id[0]} subjects → {cache_path}")
    return cache


# ── Dataset ───────────────────────────────────────────────────────────────────

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
                 sentiments, lengths, subject_ids, sentence_ids, indices, augment=False):
        self.eegs            = eegs
        self.texts           = texts
        self.sbert_np        = sbert_np
        self.input_ids_list  = input_ids_list
        self.attn_mask_list  = attn_mask_list
        self.sentiments      = sentiments
        self.lengths         = lengths
        self.subject_ids     = subject_ids
        self.sentence_ids    = sentence_ids
        self.indices         = indices
        self.augment         = augment

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        gi  = self.indices[i]
        eeg = self.eegs[gi].copy()
        if self.augment:
            eeg = _augment(eeg)
        return (eeg, self.texts[gi], self.sbert_np[gi],
                self.input_ids_list[gi], self.attn_mask_list[gi],
                self.sentiments[gi], self.lengths[gi],
                self.subject_ids[gi], self.sentence_ids[gi])


def collate_fn(batch):
    eegs, texts, sberts, ids_list, mask_list, sents, lens, subj_ids, sent_ids = zip(*batch)
    eegs    = torch.from_numpy(np.stack(eegs))
    sberts  = torch.from_numpy(np.stack(sberts))
    sents   = torch.tensor(sents, dtype=torch.float32)
    lens    = torch.tensor(lens,  dtype=torch.float32)
    subj_ids = torch.tensor(subj_ids, dtype=torch.long)
    sent_ids = torch.tensor(sent_ids, dtype=torch.long)
    max_len = max(ids.shape[0] for ids in ids_list)
    B = len(batch)
    padded_ids  = torch.full((B, max_len), -100, dtype=torch.long)
    padded_attn = torch.zeros(B, max_len, dtype=torch.long)
    for i, (ids, mask) in enumerate(zip(ids_list, mask_list)):
        L = ids.shape[0]
        padded_ids[i, :L]  = ids
        padded_attn[i, :L] = mask
    return (eegs, list(texts), sberts, padded_ids, padded_attn,
            sents, lens, subj_ids, sent_ids)


# ── Gradient Reversal (from eval_vlm_final_bench.py) ─────────────────────────

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


# ── Shared EEG Encoder components ─────────────────────────────────────────────

class MultiScaleRasterizer(nn.Module):
    """IDW topographic rasterizer: EEG (B,T,64*6) → topographic video (B,T,6,64,64)."""
    def __init__(self, size=64, n_electrodes=64, n_bands=NUM_EEG_BANDS):
        super().__init__()
        self.n_bands = n_bands
        self.n_electrodes = n_electrodes
        angles = torch.linspace(0, 2 * np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
        px = gx.flatten().unsqueeze(1)
        py = gy.flatten().unsqueeze(1)
        ex = pos_2d[:, 0].unsqueeze(0)
        ey = pos_2d[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex)**2 + (py - ey)**2)
        w = 1.0 / (dist + 1e-4) ** 2.0
        r = torch.sqrt(px**2 + py**2)
        w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)

    def forward(self, x):
        B, T, _ = x.shape
        xs = x.reshape(B * T, self.n_bands, self.n_electrodes)
        return (xs @ self.W.T.to(x.device, x.dtype)).reshape(B, T, self.n_bands, 64, 64)


class ChannelAdapter(nn.Module):
    """1×1 conv: 6 EEG frequency bands → 3 RGB channels for VideoMAE."""
    def __init__(self, in_ch=NUM_EEG_BANDS):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 3, 1)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        return self.conv(imgs.view(B * T, C, H, W)).view(B, T, 3, H, W)


# ── Stage 1 Model: EEGV1400_S1 ────────────────────────────────────────────────

class EEGV1400_S1(nn.Module):
    """
    V1400 Stage 1: VideoMAE discriminative pretraining with:
    - Subject-adversarial training (GRL) → subject-invariant features
    - Multi-positive InfoNCE (same sentence across subjects = positives)
    - Diversity loss directly on v_mean (from V1000)
    - MoCo queue (8192) for hard negatives
    """
    def __init__(self, n_subjects=N_SUBJECTS):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel, T5ForConditionalGeneration

        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=1024, tubelet_size=4, hidden_size=768)
        self.rasterizer = MultiScaleRasterizer()
        self.ch_adapt   = ChannelAdapter()
        self.video_enc  = VideoMAEModel(v_cfg)

        self.sem_proj = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Linear(256, SEM_DIM))
        self.txt_proj = nn.Sequential(
            nn.Linear(SBERT_DIM, 256), nn.GELU(), nn.Linear(256, SEM_DIM))

        self.register_buffer("queue",
            F.normalize(torch.randn(QUEUE_SIZE, SEM_DIM), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.sentiment_head = nn.Linear(SEM_DIM, 1)
        self.length_head    = nn.Linear(SEM_DIM, 1)

        # Subject-adversarial classifier (GRL reverses its gradient)
        self.subject_grl = nn.Sequential(
            nn.Linear(768, 128), nn.GELU(), nn.Linear(128, n_subjects))

        # T5 generation path (not trained in Stage 1, but present for compatibility)
        self.eeg_to_t5 = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, T5_DIM), nn.GELU(), nn.Linear(T5_DIM, T5_DIM))
        self.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        for p in self.t5.parameters():
            p.requires_grad = False

    def encode(self, traj):
        imgs  = self.rasterizer(traj)
        imgs  = self.ch_adapt(imgs)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state  # (B, 4096, 768)
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(2)    # (B, 256, 768)

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
                sentiment_labels, length_labels, subject_ids, sentence_ids):
        B, device = traj.shape[0], traj.device
        dtype     = next(p for p in self.parameters() if p.requires_grad).dtype
        traj      = traj.to(device, dtype)

        v_seq  = self.encode(traj)                              # (B, 256, 768) frozen later
        v_mean = v_seq.mean(1)                                  # (B, 768)
        q      = F.normalize(self.sem_proj(v_mean), dim=-1)    # (B, 128)

        # ── Multi-positive InfoNCE ────────────────────────────────────────────
        # In-batch: treat same sentence (across subjects) as positives
        k = F.normalize(self.txt_proj(sbert_embs.to(device, dtype)), dim=-1)   # (B, 128)
        # Build positive mask from sentence_ids
        s_ids     = sentence_ids.to(device)
        pos_mask  = (s_ids.unsqueeze(0) == s_ids.unsqueeze(1))  # (B, B) bool
        logits_ib = (q @ k.T) / 0.07                            # (B, B)
        # Multi-positive: -log(sum_pos / sum_all)
        log_denom  = torch.logsumexp(logits_ib, dim=1)          # (B,)
        # For samples with no other positive (all sentences unique in batch), fall back to standard
        has_pos = pos_mask.sum(1).gt(0)
        log_num = torch.where(
            has_pos,
            torch.logsumexp(logits_ib.masked_fill(~pos_mask, -1e9), dim=1),
            logits_ib.diagonal())
        loss_align_ib = -(log_num - log_denom).mean()

        # Queue negatives: standard InfoNCE (all queue entries = negatives)
        pos_queue  = (q * k).sum(-1, keepdim=True) / 0.07      # (B, 1)
        neg_queue  = q @ self.queue.T.detach() / 0.07          # (B, QUEUE_SIZE)
        loss_align_q = F.cross_entropy(
            torch.cat([pos_queue, neg_queue], dim=1),
            torch.zeros(B, dtype=torch.long, device=device))
        self._enqueue(k)

        loss_align = 0.5 * loss_align_ib + 0.5 * loss_align_q

        # ── Diversity loss on raw v_mean (V1000 fix) ─────────────────────────
        if B > 1:
            v_norm   = F.normalize(v_mean, dim=-1)
            cos_mat  = v_norm @ v_norm.T
            off_mask = ~torch.eye(B, dtype=torch.bool, device=device)
            loss_div = cos_mat[off_mask].clamp(min=0).mean()
        else:
            loss_div = torch.zeros(1, device=device, dtype=dtype).squeeze()

        # ── Subject-adversarial loss (GRL) ───────────────────────────────────
        # GRL makes gradient from this loss push VideoMAE to remove subject info
        v_grl      = grad_reverse(v_mean.to(dtype), alpha=1.0)
        subj_logit = self.subject_grl(v_grl)
        loss_adv   = F.cross_entropy(
            subj_logit, subject_ids.to(device).clamp(max=subj_logit.shape[1]-1))

        # ── Auxiliary heads ───────────────────────────────────────────────────
        loss_sent = F.binary_cross_entropy_with_logits(
            self.sentiment_head(q).squeeze(-1), sentiment_labels.to(device))
        loss_len  = F.mse_loss(
            self.length_head(q).squeeze(-1), length_labels.to(device))

        return loss_align, loss_div, loss_adv, loss_sent, loss_len

    @torch.no_grad()
    def generate(self, traj, tokenizer, max_new_tokens=80):
        from transformers.modeling_outputs import BaseModelOutput
        traj   = traj.to(next(p for p in self.parameters() if p.requires_grad).dtype)
        device = next(p for p in self.parameters() if p.requires_grad).device
        B      = traj.shape[0]
        v_seq  = self.encode(traj)
        enc    = self.eeg_to_t5(v_seq)
        enc_attn = torch.ones(B, EEG_SEQ_LEN, dtype=torch.long, device=device)
        results = []
        for b in range(B):
            eo  = BaseModelOutput(last_hidden_state=enc[b:b+1])
            out = self.t5.generate(
                encoder_outputs=eo, attention_mask=enc_attn[b:b+1],
                max_new_tokens=max_new_tokens, num_beams=4, do_sample=False,
                no_repeat_ngram_size=3, pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)
            results.append(tokenizer.decode(out[0], skip_special_tokens=True).strip())
        return results


# ── Q-Former Bridge ────────────────────────────────────────────────────────────

class QFormerLayer(nn.Module):
    """Single Q-Former layer: self-attn among queries + cross-attn to encoder + FFN."""
    def __init__(self, d_model=768, n_heads=12, ff_dim=3072, dropout=0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, queries, memory):
        # Self-attention among query tokens
        q2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + self.drop(q2))
        # Cross-attention: queries attend to VideoMAE output sequence
        q2, _ = self.cross_attn(queries, memory, memory)
        queries = self.norm2(queries + self.drop(q2))
        # Feed-forward
        queries = self.norm3(queries + self.ffn(queries))
        return queries


class QFormerBridge(nn.Module):
    """
    Q-Former bridge: 32 learnable queries × 4-layer cross-attention to VideoMAE output.
    Produces 32 rich EEG-conditioned tokens that drive T5 cross-attention.
    Each query learns to extract complementary aspects of the EEG signal.
    """
    def __init__(self, d_model=768, n_queries=QFORMER_TOKENS, n_layers=4, n_heads=12):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.layers       = nn.ModuleList([QFormerLayer(d_model, n_heads) for _ in range(n_layers)])
        self.out_proj     = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, v_seq):
        """v_seq: (B, 256, 768) VideoMAE output → returns eeg_q: (B, 32, 768)"""
        B = v_seq.shape[0]
        queries = self.query_tokens.expand(B, -1, -1)  # (B, 32, 768)
        for layer in self.layers:
            queries = layer(queries, v_seq)
        return self.out_proj(queries)                   # (B, 32, 768)


# ── Stage 2 Model: EEGV1400_S2 ────────────────────────────────────────────────

class EEGV1400_S2(nn.Module):
    """
    V1400 Stage 2: QFormerBridge + T5-base with relational distillation.

    Architecture:
      EEG → MultiScaleRasterizer → ChannelAdapter → VideoMAE (FROZEN)
          → QFormerBridge (32 query tokens, 4-layer) → eeg_to_t5_proj
          → T5 decoder cross-attention (FROZEN)

    Key: QFormerBridge cannot collapse — 32 diverse queries each extract different
    VideoMAE features. Relational distillation on qformer_mean prevents subsequent
    collapse of the projection.
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

        # Q-Former bridge (randomly initialized, trainable)
        self.qformer = QFormerBridge(d_model=768, n_queries=QFORMER_TOKENS,
                                     n_layers=4, n_heads=12)

        # Linear projection from Q-Former dim → T5 cross-attention dim
        self.eeg_to_t5_proj = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, T5_DIM))

        # Alignment heads (operate on qformer_mean)
        self.sem_proj = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(), nn.Linear(256, SEM_DIM))
        self.txt_proj = nn.Sequential(
            nn.Linear(SBERT_DIM, 256), nn.GELU(), nn.Linear(256, SEM_DIM))

        self.register_buffer("queue",
            F.normalize(torch.randn(QUEUE_SIZE, SEM_DIM), dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.sentiment_head = nn.Linear(SEM_DIM, 1)
        self.length_head    = nn.Linear(SEM_DIM, 1)

        # T5 (frozen)
        self.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        for p in self.t5.parameters():
            p.requires_grad = False

    def freeze_videomae(self):
        for p in self.video_enc.parameters():
            p.requires_grad = False
        for p in self.rasterizer.parameters():
            p.requires_grad = False

    def encode(self, traj):
        imgs  = self.rasterizer(traj)
        imgs  = self.ch_adapt(imgs)
        with torch.no_grad():
            v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(2)  # (B, 256, 768)

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
                sentiment_labels, length_labels, **kwargs):
        from transformers.modeling_outputs import BaseModelOutput

        B, device = traj.shape[0], traj.device
        dtype     = next(p for p in self.parameters() if p.requires_grad).dtype
        traj      = traj.to(device, dtype)

        v_seq      = self.encode(traj)                               # (B, 256, 768) frozen
        eeg_q      = self.qformer(v_seq)                             # (B, 32, 768) trainable
        eeg_kv     = self.eeg_to_t5_proj(eeg_q)                     # (B, 32, 768) for T5
        eeg_mean   = eeg_q.mean(1)                                   # (B, 768) for rel_loss

        q = F.normalize(self.sem_proj(eeg_mean), dim=-1)             # (B, 128)

        # ── InfoNCE alignment ────────────────────────────────────────────────
        k = F.normalize(self.txt_proj(sbert_embs.to(device, dtype)), dim=-1)
        pos        = (q * k).sum(-1, keepdim=True) / 0.07
        negs       = q @ self.queue.T.detach() / 0.07
        loss_align = F.cross_entropy(
            torch.cat([pos, negs], dim=1),
            torch.zeros(B, dtype=torch.long, device=device))
        self._enqueue(k)

        # ── Relational distillation on eeg_mean ─────────────────────────────
        # Forces Q-Former output to preserve pairwise similarity structure of SBERT.
        # This is the CRITICAL anti-collapse fix (same principle as V1200 enc_mean).
        loss_rel = torch.tensor(0.0, device=device)
        if B > 1:
            k_norm  = F.normalize(sbert_embs.to(device, dtype), dim=-1)
            e_norm  = F.normalize(eeg_mean.to(dtype), dim=-1)
            eeg_sim = e_norm @ e_norm.T                              # (B, B)
            txt_sim = (k_norm @ k_norm.T).detach()
            off     = ~torch.eye(B, dtype=torch.bool, device=device)
            loss_rel = F.mse_loss(eeg_sim[off], txt_sim[off])

        # ── Auxiliary heads ───────────────────────────────────────────────────
        loss_sent = F.binary_cross_entropy_with_logits(
            self.sentiment_head(q).squeeze(-1), sentiment_labels.to(device))
        loss_len  = F.mse_loss(
            self.length_head(q).squeeze(-1), length_labels.to(device))

        # ── T5 LM (primary objective) ─────────────────────────────────────────
        enc_attn = torch.ones(B, QFORMER_TOKENS, dtype=torch.long, device=device)
        labels   = target_ids.to(device).clone()
        labels[target_mask.to(device) == 0] = -100
        enc_out  = BaseModelOutput(last_hidden_state=eeg_kv.to(dtype))
        out      = self.t5(encoder_outputs=enc_out, attention_mask=enc_attn, labels=labels)
        loss_lm  = out.loss

        return loss_lm, loss_align, loss_rel, loss_sent, loss_len

    @torch.no_grad()
    def generate(self, traj, tokenizer, max_new_tokens=80, diverse=True):
        from transformers.modeling_outputs import BaseModelOutput

        traj   = traj.to(next(p for p in self.parameters() if p.requires_grad).dtype)
        device = next(p for p in self.parameters() if p.requires_grad).device
        B      = traj.shape[0]

        v_seq  = self.encode(traj)
        eeg_q  = self.qformer(v_seq)
        eeg_kv = self.eeg_to_t5_proj(eeg_q)
        enc_attn = torch.ones(B, QFORMER_TOKENS, dtype=torch.long, device=device)

        results = []
        for b in range(B):
            eo  = BaseModelOutput(last_hidden_state=eeg_kv[b:b+1])
            if diverse:
                out = self.t5.generate(
                    encoder_outputs=eo, attention_mask=enc_attn[b:b+1],
                    max_new_tokens=max_new_tokens,
                    num_beams=10, num_beam_groups=5, diversity_penalty=1.0,
                    no_repeat_ngram_size=3, repetition_penalty=1.20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            else:
                out = self.t5.generate(
                    encoder_outputs=eo, attention_mask=enc_attn[b:b+1],
                    max_new_tokens=max_new_tokens,
                    do_sample=True, temperature=0.85, top_k=50, top_p=0.92,
                    no_repeat_ngram_size=3, repetition_penalty=1.20,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            results.append(tokenizer.decode(out[0], skip_special_tokens=True).strip())
        return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_wer(pred, ref):
    import editdistance
    p = pred.split(); r = ref.split()
    if not r: return 1.0
    return editdistance.eval(p, r) / len(r)


def compute_cer(pred, ref):
    import editdistance
    if not ref: return 1.0
    return editdistance.eval(pred, ref) / len(ref)


def cons_recall(preds, refs):
    hits = 0; total = 0
    for p, r in zip(preds, refs):
        rw = set(r.lower().split()); pw = set(p.lower().split())
        hits += len(rw & pw); total += max(1, len(rw))
    return hits / total


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, val_loader, device, epoch, tokenizer, stage):
    model.eval()
    wers, cers, all_preds, all_refs = [], [], [], []

    for batch in val_loader:
        eegs = batch[0].to(device)
        texts = batch[1]
        with torch.no_grad():
            preds = model.generate(eegs, tokenizer, max_new_tokens=80,
                                   diverse=(stage == 2))
        for pred, ref in zip(preds, texts):
            wers.append(compute_wer(pred.lower(), ref.lower()))
            cers.append(compute_cer(pred.lower(), ref.lower()))
            all_preds.append(pred)
            all_refs.append(ref)

    mean_wer  = float(np.mean(wers))
    mean_cer  = float(np.mean(cers))
    diversity = len(set(all_preds)) / max(1, len(all_preds))
    cr        = cons_recall(all_preds, all_refs)

    print(f"\n[Ep{epoch}/S{stage}] WER={mean_wer:.3f}  CER={mean_cer:.3f}  "
          f"Div={diversity:.3f}  CR={cr:.4f}")
    print("── Samples ──")
    for p, r in zip(all_preds[:5], all_refs[:5]):
        print(f"  REF : {r[:90]}")
        print(f"  PRED: {p[:90]}")
        print()
    model.train()
    return mean_wer, diversity, cr


# ── Checkpoint utilities ──────────────────────────────────────────────────────

def load_s1_checkpoint_into_s2(model_s2, s1_ckpt_path):
    """
    Transfer Stage 1 weights into Stage 2 model.
    Loads: VideoMAE, rasterizer, ch_adapt, sem_proj, txt_proj, queue.
    Keeps random init: qformer, eeg_to_t5_proj (fresh mapping).
    """
    ckpt = torch.load(s1_ckpt_path, map_location="cpu", weights_only=False)
    sd   = ckpt.get("model_state", ckpt)
    own  = model_s2.state_dict()

    skip = ("qformer.", "eeg_to_t5_proj.", "sentiment_head.", "length_head.")
    loaded, skipped_prefix, skipped_shape = [], [], []

    for k, v in sd.items():
        if any(k.startswith(p) for p in skip):
            skipped_prefix.append(k); continue
        if k in own and own[k].shape == v.shape:
            own[k] = v; loaded.append(k)
        elif k in own:
            skipped_shape.append(k)

    model_s2.load_state_dict(own, strict=False)
    print(f"S1→S2 transfer: {len(loaded)} params loaded")
    print(f"  qformer+proj KEPT RANDOM: {len(skipped_prefix)} keys")
    print(f"  Shape mismatch skipped:   {len(skipped_shape)} keys")


def load_videomae_pretrained(model):
    """Fallback: load ImageNet VideoMAE weights when no Stage 1 checkpoint."""
    from transformers import VideoMAEModel
    pretrained = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    own = model.state_dict()
    loaded = []
    for k, v in pretrained.state_dict().items():
        full_k = f"video_enc.{k}"
        if full_k in own and own[full_k].shape == v.shape:
            own[full_k] = v; loaded.append(full_k)
    model.load_state_dict(own, strict=False)
    print(f"Loaded pretrained VideoMAE: {len(loaded)} params")


def find_best_s1_checkpoint(ckpt_dir=CKPT_DIR):
    """Find the best Stage 1 checkpoint by diversity × CR score from filename."""
    candidates = sorted(glob.glob(str(ckpt_dir / "s1_ep*.pt")))
    if not candidates:
        return None
    # Prefer 's1_best.pt' if it exists
    best = ckpt_dir / "s1_best.pt"
    if best.exists():
        return str(best)
    return candidates[-1]  # latest


# ── Optimizer factories ────────────────────────────────────────────────────────

def make_s1_optimizer(model, steps_per_epoch, epochs):
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    emb_p, early_p, mid_p, top_p, adapt_p = [], [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if "video_enc.embeddings" in n:                               emb_p.append(p)
        elif any(f"video_enc.encoder.layer.{i}." in n for i in range(0, 4)):  early_p.append(p)
        elif any(f"video_enc.encoder.layer.{i}." in n for i in range(4, 8)):  mid_p.append(p)
        elif any(f"video_enc.encoder.layer.{i}." in n for i in range(8, 12)): top_p.append(p)
        else:                                                          adapt_p.append(p)

    param_groups = [
        {"params": emb_p,   "lr": 5e-7},
        {"params": early_p, "lr": 1e-6},
        {"params": mid_p,   "lr": 5e-6},
        {"params": top_p,   "lr": 1e-5},
        {"params": adapt_p, "lr": 2e-4},
    ]
    # Filter empty groups
    param_groups = [g for g in param_groups if g["params"]]
    max_lrs = [g["lr"] for g in param_groups]

    opt   = AdamW(param_groups, weight_decay=1e-4)
    sched = OneCycleLR(opt, max_lr=max_lrs,
                       total_steps=steps_per_epoch * epochs,
                       pct_start=0.10, anneal_strategy="cos")
    return opt, sched


def make_s2_optimizer(model, steps_per_epoch, epochs):
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR

    qformer_p, proj_p, adapt_p = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("qformer."):
            qformer_p.append(p)
        elif n.startswith("eeg_to_t5_proj."):
            proj_p.append(p)
        else:
            adapt_p.append(p)

    param_groups = [
        {"params": qformer_p, "lr": 3e-4},   # Q-Former: random init, higher LR
        {"params": proj_p,    "lr": 1e-4},   # projection
        {"params": adapt_p,   "lr": 5e-5},   # alignment heads, ch_adapt
    ]
    param_groups = [g for g in param_groups if g["params"]]
    max_lrs = [g["lr"] for g in param_groups]

    opt   = AdamW(param_groups, weight_decay=1e-4)
    sched = OneCycleLR(opt, max_lr=max_lrs,
                       total_steps=steps_per_epoch * epochs,
                       pct_start=0.10, anneal_strategy="cos")
    return opt, sched


# ── Training loops ────────────────────────────────────────────────────────────

def train_stage1(model, train_ds, val_ds, tokenizer, device, epochs=25,
                 dry_run=False):
    from torch.utils.data import DataLoader

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    steps_per_epoch = max(1, len(train_loader) // ACCUM_STEPS)
    opt, sched      = make_s1_optimizer(model, steps_per_epoch, epochs)

    best_score = -1.0
    best_wer   = 1e9
    low_div_streak = 0
    EARLY_STOP_DIV    = 0.10
    EARLY_STOP_STREAK = 5

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0; n_accum = 0

        for step, batch in enumerate(train_loader, 1):
            (eegs, texts, sberts, tgt_ids, tgt_mask,
             sents, lens, subj_ids, sent_ids) = batch
            eegs = eegs.to(device, torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                align, div_l, adv_l, sent_l, len_l = model(
                    eegs, sberts, tgt_ids, tgt_mask,
                    sents, lens, subj_ids, sent_ids)
                total = (S1_W_ALIGN * align + S1_W_DIV * div_l
                         + S1_W_ADV * adv_l
                         + S1_W_SENT * sent_l + S1_W_LEN * len_l)
                loss  = total / ACCUM_STEPS

            loss.backward()
            n_accum += 1

            if n_accum % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                sched.step()
                total_loss += total.item()

                step_num = n_accum // ACCUM_STEPS
                if step_num % 10 == 0:
                    print(f"  ep={epoch}[S1] step={step_num} "
                          f"align={align.item():.3f} div={div_l.item():.3f} "
                          f"adv={adv_l.item():.3f} sent={sent_l.item():.3f}")

            if dry_run and n_accum >= ACCUM_STEPS * 5:
                print("  [dry-run] 5 steps done.")
                break

        avg = total_loss / max(1, n_accum // ACCUM_STEPS)
        print(f"\nEpoch {epoch}[S1]: avg_loss={avg:.3f}")

        wer, div, cr = validate(model, val_loader, device, epoch, tokenizer, stage=1)
        score = div * cr

        if score > best_score or wer < best_wer:
            best_score = max(best_score, score)
            best_wer   = min(best_wer, wer)
            name = f"s1_ep{epoch}_div{div:.3f}_cr{cr:.3f}_wer{wer:.3f}.pt"
            path = CKPT_DIR / name
            torch.save({
                "epoch": epoch, "stage": 1,
                "wer": wer, "diversity": div, "cr": cr,
                "model_state": model.state_dict(),
            }, str(path))
            # Always keep a 's1_best.pt' symlink-style copy
            best_path = CKPT_DIR / "s1_best.pt"
            torch.save({
                "epoch": epoch, "stage": 1,
                "wer": wer, "diversity": div, "cr": cr,
                "model_state": model.state_dict(),
            }, str(best_path))
            print(f"  ✓ Saved: {name}")

        if div < EARLY_STOP_DIV:
            low_div_streak += 1
            print(f"  ⚠ Low diversity ({div:.3f}) streak={low_div_streak}/{EARLY_STOP_STREAK}")
            if low_div_streak >= EARLY_STOP_STREAK:
                print(f"  ✗ Early stop.")
                break
        else:
            low_div_streak = 0

        if dry_run:
            break

    print(f"\nStage 1 complete. Best: div×cr={best_score:.4f}  WER={best_wer:.3f}")
    print(f"Best checkpoint: {CKPT_DIR / 's1_best.pt'}")


def train_stage2(model, train_ds, val_ds, tokenizer, device, epochs=40,
                 dry_run=False):
    from torch.utils.data import DataLoader

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=0, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    steps_per_epoch = max(1, len(train_loader) // ACCUM_STEPS)
    opt, sched      = make_s2_optimizer(model, steps_per_epoch, epochs)

    best_score = -1.0
    best_wer   = 1e9
    low_div_streak = 0
    EARLY_STOP_DIV    = 0.08
    EARLY_STOP_STREAK = 5

    model.train()
    for epoch in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0; n_accum = 0

        for step, batch in enumerate(train_loader, 1):
            (eegs, texts, sberts, tgt_ids, tgt_mask,
             sents, lens, subj_ids, sent_ids) = batch
            eegs = eegs.to(device, torch.bfloat16)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lm, align, rel, sent_l, len_l = model(
                    eegs, sberts, tgt_ids, tgt_mask, sents, lens)
                total = (S2_W_LM * lm + S2_W_ALIGN * align + S2_W_REL * rel
                         + S2_W_SENT * sent_l + S2_W_LEN * len_l)
                loss  = total / ACCUM_STEPS

            loss.backward()
            n_accum += 1

            if n_accum % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                sched.step()
                total_loss += total.item()

                step_num = n_accum // ACCUM_STEPS
                if step_num % 10 == 0:
                    print(f"  ep={epoch}[S2] step={step_num} "
                          f"lm={lm.item():.3f} align={align.item():.3f} "
                          f"rel={rel.item():.4f} sent={sent_l.item():.3f}")

            if dry_run and n_accum >= ACCUM_STEPS * 5:
                print("  [dry-run] 5 steps done.")
                break

        avg = total_loss / max(1, n_accum // ACCUM_STEPS)
        print(f"\nEpoch {epoch}[S2]: avg_loss={avg:.3f}")

        wer, div, cr = validate(model, val_loader, device, epoch, tokenizer, stage=2)
        score = div * cr

        if score > best_score or wer < best_wer:
            best_score = max(best_score, score)
            best_wer   = min(best_wer, wer)
            name = f"s2_ep{epoch}_div{div:.3f}_cr{cr:.3f}_wer{wer:.3f}.pt"
            path = CKPT_DIR / name
            torch.save({
                "epoch": epoch, "stage": 2,
                "wer": wer, "diversity": div, "cr": cr,
                "model_state": model.state_dict(),
            }, str(path))
            best_path = CKPT_DIR / "s2_best.pt"
            torch.save({
                "epoch": epoch, "stage": 2,
                "wer": wer, "diversity": div, "cr": cr,
                "model_state": model.state_dict(),
            }, str(best_path))
            print(f"  ✓ Saved: {name}")

        if div < EARLY_STOP_DIV:
            low_div_streak += 1
            print(f"  ⚠ Low diversity ({div:.3f}) streak={low_div_streak}/{EARLY_STOP_STREAK}")
            if low_div_streak >= EARLY_STOP_STREAK:
                print(f"  ✗ Early stop.")
                break
        else:
            low_div_streak = 0

        if dry_run:
            break

    print(f"\nStage 2 complete. Best: div×cr={best_score:.4f}  WER={best_wer:.3f}")
    print(f"Best checkpoint: {CKPT_DIR / 's2_best.pt'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EEG-to-Text V1400 Training")
    parser.add_argument("--stage",    type=int, default=1, choices=[1, 2],
                        help="Training stage (1=VideoMAE discriminative, 2=Q-Former+T5)")
    parser.add_argument("--epochs",   type=int, default=None,
                        help="Number of epochs (default: 25 for S1, 40 for S2)")
    parser.add_argument("--s1-ckpt",  type=str, default=None,
                        help="Stage 1 checkpoint to load for Stage 2 (auto-finds if not given)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Run 5 gradient steps per stage to verify no crashes")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Directory containing ZuCo_v1 and ZuCo_v2")
    parser.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR),
                        help="Directory for checkpoints")
    args = parser.parse_args()

    global DATA_DIR, CKPT_DIR
    DATA_DIR = Path(args.data_dir)
    CKPT_DIR = Path(args.ckpt_dir)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs or (25 if args.stage == 1 else 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: Running on CPU — will be very slow.")

    # ── Data ──────────────────────────────────────────────────────────────────
    cache = load_or_build_cache(DATA_DIR)
    eegs         = cache["eegs"]
    texts        = cache["texts"]
    subject_ids  = cache["subject_ids"]
    sentence_ids = cache["sentence_ids"]
    n_subjects   = min(int(cache.get("n_subjects", N_SUBJECTS)), N_SUBJECTS)
    N = len(texts)
    print(f"Dataset: {N} samples, {n_subjects} subjects")

    # SBERT embeddings
    sbert_cache_path = CKPT_DIR / "sbert_embs.pt"
    if sbert_cache_path.exists():
        sbert_np = torch.load(sbert_cache_path, weights_only=False)
        print(f"SBERT cache hit: {sbert_cache_path}")
    else:
        print("Computing SBERT embeddings (this takes a few minutes)...")
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        sbert_np    = sbert_model.encode(texts, batch_size=64, show_progress_bar=True,
                                         convert_to_numpy=True)
        torch.save(sbert_np, sbert_cache_path)
        print(f"SBERT saved: {sbert_cache_path}")
    print(f"SBERT: {sbert_np.shape}")

    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    print("Tokenizing targets...")
    input_ids_list, attn_mask_list = [], []
    for text in texts:
        enc = tokenizer(text, max_length=MAX_TEXT_LEN, truncation=True,
                        padding=False, return_tensors='pt')
        input_ids_list.append(enc.input_ids[0])
        attn_mask_list.append(enc.attention_mask[0])

    sentiments, lengths = build_labels(texts, tokenizer)

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(N)
    split = int(0.85 * N)
    train_idx = idx[:split].tolist()
    val_idx   = idx[split:].tolist()

    train_ds = EEGTextDataset(
        eegs, texts, sbert_np, input_ids_list, attn_mask_list,
        sentiments, lengths, subject_ids, sentence_ids, train_idx, augment=True)
    val_ds = EEGTextDataset(
        eegs, texts, sbert_np, input_ids_list, attn_mask_list,
        sentiments, lengths, subject_ids, sentence_ids, val_idx, augment=False)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.stage == 1:
        print(f"\n=== Stage 1: VideoMAE Discriminative Pretraining ({epochs} epochs) ===")
        print("  + Subject-adversarial GRL (subject-invariant features)")
        print("  + Cross-subject multi-positive InfoNCE")
        print("  + Diversity loss on v_mean")
        print(f"  + Queue size: {QUEUE_SIZE}")
        model = EEGV1400_S1(n_subjects=n_subjects).to(device)
        load_videomae_pretrained(model)
        model = model.to(torch.bfloat16)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"Trainable: {trainable/1e6:.1f}M  Frozen: {frozen/1e6:.1f}M")

        train_stage1(model, train_ds, val_ds, tokenizer, device,
                     epochs=epochs, dry_run=args.dry_run)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    elif args.stage == 2:
        print(f"\n=== Stage 2: Q-Former Bridge + T5 ({epochs} epochs) ===")
        print("  + Q-Former: 32 query tokens, 4-layer cross-attention to VideoMAE")
        print("  + Relational distillation on qformer_mean (anti-collapse)")
        print("  + Diverse beam search (10 beams, 5 groups)")

        # Find Stage 1 checkpoint
        s1_path = args.s1_ckpt or find_best_s1_checkpoint()
        if s1_path and Path(s1_path).exists():
            print(f"Loading Stage 1 checkpoint: {s1_path}")
        else:
            print("WARNING: No Stage 1 checkpoint found. Using ImageNet VideoMAE.")
            s1_path = None

        model = EEGV1400_S2().to(device)
        if s1_path:
            load_s1_checkpoint_into_s2(model, s1_path)
        else:
            load_videomae_pretrained(model)
        model.freeze_videomae()
        model = model.to(torch.bfloat16)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"Trainable: {trainable/1e6:.1f}M  Frozen: {frozen/1e6:.1f}M")
        print(f"Loss: LM×{S2_W_LM}  ALIGN×{S2_W_ALIGN}  REL×{S2_W_REL}  "
              f"SENT×{S2_W_SENT}  LEN×{S2_W_LEN}")

        train_stage2(model, train_ds, val_ds, tokenizer, device,
                     epochs=epochs, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
