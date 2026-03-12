
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

app = modal.App("eeg-vlm-v30")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v30", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)

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
# V3.0: Subword CTC + CTC-Guided Denoising
#
# Breakthrough changes over V2.0:
#
# [1] SUBWORD CTC (BPE ~300 tokens)
#     Character-level CTC was limited — EEG temporal resolution (~128Hz after
#     downsampling) can't resolve individual phonemes. 256 VideoMAE steps for
#     ~20 words = ~13 steps/word. Subword BPE (using the LLM's own tokenizer
#     truncated to top-300 most frequent tokens + blank) gives each CTC step
#     a semantically richer target. The model now predicts word-pieces instead
#     of individual characters, matching EEG's natural temporal grain.
#
# [2] CTC-GUIDED DENOISING (Phase 2)
#     V2.0 Phase 2 had exposure bias: the LLM learned to ignore brain prefix
#     tokens and generate purely from teacher-forced text. V3.0 uses CTC
#     decode output as an explicit noisy prompt:
#       "EEG reading (noisy): '{ctc_decoded}'. Reconstruct: "
#     This turns Phase 2 into a denoising task the LLM already knows how to do.
#     The CTC gives a skeleton and the LLM repairs it using both language priors
#     and brain prefix embeddings.
#
# [3] PREFIX DROPOUT (50%)
#     During Phase 2, 50% of batches have brain prefix zeroed out, forcing the
#     LLM to sometimes rely only on the CTC hint. This prevents the LLM from
#     learning a shortcut through the prefix and ensures both pathways
#     (prefix + CTC hint) contribute meaningfully.
#
# [4] EXTENDED PHASE 1 (40 epochs)
#     V2.0 Phase 1 ran 30 epochs but loss was still dropping. 40 epochs lets
#     the subword CTC fully converge.
#
# Architecture unchanged: VideoMAE → CTC + Bridge → LLM (Qwen2.5-0.5B + LoRA)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Subword Vocabulary ──────────────────────────────────────────────────────

class SubwordCTCVocab:
    """
    Build a small CTC vocabulary from the LLM tokenizer's most frequent tokens.
    Token 0 = CTC blank. Tokens 1..N = top-N most frequent subwords.
    """
    def __init__(self, tokenizer, max_vocab=300):
        self.tokenizer = tokenizer
        self.max_vocab = max_vocab
        self.blank_id = 0

        # Build frequency table: use tokenizer on real English text to find
        # the most frequently occurring tokens, then take top-N.
        # This ensures we get real word-pieces, not control chars.
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "He was president of the Union Pacific Railroad from 1884 to 1890.",
            "I love the robust middle of this picture.",
            "No French people were harmed during the making of this movie.",
            "This is a very funny movie that is a good laugh all over.",
            "She walked into the room and sat down quietly.",
            "The results of the experiment were inconclusive at best.",
            "It was a dark and stormy night when the phone rang.",
            "The government announced new policies regarding education.",
            "Scientists discovered a new species of butterfly in the Amazon.",
            "The restaurant served excellent food at reasonable prices.",
            "He ran quickly through the forest to escape the approaching storm.",
            "The book was published in nineteen forty two during the war.",
            "Music has the power to transform our emotional state completely.",
            "The children played happily in the park until sunset.",
            "Economic growth remained steady throughout the fiscal year.",
            "The professor explained the theory with remarkable clarity.",
            "Ancient civilizations developed sophisticated systems of writing.",
            "The film received mixed reviews from critics and audiences alike.",
            "Technology continues to reshape how we communicate with each other.",
        ] * 10  # repeat for better frequency estimates

        from collections import Counter
        freq = Counter()
        for text in sample_texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            freq.update(ids)

        # Filter: only keep tokens that decode to printable, non-empty strings
        candidates = []
        for tok_id, count in freq.most_common():
            text = tokenizer.decode([tok_id])
            if text.strip() and text.isprintable() and not text.startswith('<'):
                candidates.append((tok_id, text, count))
            if len(candidates) >= max_vocab:
                break

        # If we don't have enough from sample text, fill from full vocab
        if len(candidates) < max_vocab:
            used_ids = {c[0] for c in candidates}
            for tok_id in range(tokenizer.vocab_size):
                if tok_id in used_ids:
                    continue
                text = tokenizer.decode([tok_id])
                if (text.strip() and text.isprintable()
                        and len(text) <= 6 and not text.startswith('<')):
                    candidates.append((tok_id, text, 0))
                if len(candidates) >= max_vocab:
                    break

        selected = candidates[:max_vocab]

        # CTC vocab: 0=blank, 1..N = subword tokens
        self.ctc_to_llm = {}  # ctc_id → llm_token_id
        self.llm_to_ctc = {}  # llm_token_id → ctc_id
        self.ctc_to_text = {0: '<blank>'}

        for i, (llm_id, text, _) in enumerate(selected):
            ctc_id = i + 1  # 0 is blank
            self.ctc_to_llm[ctc_id] = llm_id
            self.llm_to_ctc[llm_id] = ctc_id
            self.ctc_to_text[ctc_id] = text

        self.vocab_size = len(selected) + 1  # +1 for blank
        print(f"[SubwordCTC] Built vocab: {self.vocab_size} tokens "
              f"(1 blank + {len(selected)} subwords)")
        # Show a few examples
        examples = [self.ctc_to_text[i] for i in range(1, min(11, self.vocab_size))]
        print(f"  First 10 tokens: {examples}")

    def encode(self, text):
        """Convert text → list of CTC subword IDs (no blank)."""
        llm_ids = self.tokenizer.encode(text, add_special_tokens=False)
        ctc_ids = []
        for lid in llm_ids:
            if lid in self.llm_to_ctc:
                ctc_ids.append(self.llm_to_ctc[lid])
            # else: OOV token — skip (rare with top-300)
        return ctc_ids

    def decode(self, ctc_ids):
        """Convert list of CTC IDs → text string."""
        llm_ids = [self.ctc_to_llm[c] for c in ctc_ids if c in self.ctc_to_llm]
        return self.tokenizer.decode(llm_ids)


def greedy_ctc_decode_subword(logits, vocab):
    """logits: (T, B, Vocab) → list[str] using subword vocab."""
    ids = logits.argmax(dim=-1)  # (T, B)
    results = []
    for b in range(ids.shape[1]):
        seq = ids[:, b].tolist()
        out_ids = []
        prev = -1
        for i in seq:
            if i != prev and i != vocab.blank_id:
                out_ids.append(i)
            prev = i
        results.append(vocab.decode(out_ids))
    return results


def compute_cer(pred, ref):
    import editdistance
    if not ref:
        return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, ref) / len(ref), 2.0)


def compute_wer(pred, ref):
    """Word-level error rate — more meaningful for subword CTC."""
    import editdistance
    pred_words = pred.strip().split()
    ref_words = ref.strip().split()
    if not ref_words:
        return 0.0 if not pred_words else 1.0
    return min(editdistance.eval(pred_words, ref_words) / len(ref_words), 2.0)


# ─── Memory Bank (Phase 2 InfoNCE) ───────────────────────────────────────────

class MemoryBank:
    """FIFO queue of text embeddings used as hard negatives for InfoNCE."""

    def __init__(self, size=512, dim=896):
        self.size = size
        self.bank = torch.zeros(size, dim)
        self.ptr  = 0
        self.n    = 0

    def enqueue(self, emb: torch.Tensor):
        emb = emb.detach().cpu().float()
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

    def forward(self, x):
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

    def forward(self, imgs):
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

    def forward(self, x):
        B = x.shape[0]
        q = self.latents.unsqueeze(0).expand(B, -1, -1)
        x, _ = self.mha(self.ln_q(q), self.ln_k(x), x)
        return self.proj(x)


# ─── Model ───────────────────────────────────────────────────────────────────

class EEG_VLM_V30(nn.Module):
    def __init__(self, q_name, ctc_vocab_size):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType

        v_cfg = VideoMAEConfig(
            num_channels=3,
            image_size=64,
            patch_size=16,
            num_frames=1024,
            tubelet_size=4,
            hidden_size=768,
        )
        self.rasterizer  = MultiScaleRasterizer()
        self.ch_adapt    = ChannelAdapter()
        self.video_enc   = VideoMAEModel(v_cfg)
        self.bridge      = CrossAttentionBridge(768, 1024, n_latents=128)
        self.ctc_head    = nn.Linear(768, ctc_vocab_size)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

        self.llm      = AutoModelForCausalLM.from_pretrained(q_name, torch_dtype=torch.bfloat16)
        self.llm      = get_peft_model(self.llm, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=256,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        ))
        self.llm_proj = nn.Linear(1024, 896)

    def encode(self, traj):
        """traj: (B, 1024, 256) → v_seq: (B, 256, 768)"""
        imgs = self.rasterizer(traj)
        imgs = self.ch_adapt(imgs)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(dim=2)

    def forward(self, traj, ctc_ids, ctc_lens,
                input_ids=None, labels=None, neg_bank=None,
                ctc_prompt_ids=None, ctc_prompt_mask=None,
                prefix_dropout=False):
        """
        Phase 1: model(eeg, ctc_ids, ctc_lens)  → CTC only, no LLM
        Phase 2: model(eeg, ctc_ids, ctc_lens, inp, lbl, neg_bank,
                       ctc_prompt_ids, ctc_prompt_mask, prefix_dropout)
        Returns: (loss_lm, loss_ctc, loss_lock, loss_div, tok_mean_or_None)
        """
        B = traj.shape[0]
        device = traj.device

        v_seq = self.encode(traj)  # (B, 256, 768)

        # ── CTC on 256 real VideoMAE steps ───────────────────────────────────
        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)  # (256, B, vocab)
        input_lens = torch.full((B,), 256, dtype=torch.long, device=device)
        loss_ctc   = self.ctc_loss_fn(
            F.log_softmax(ctc_logits, dim=-1), ctc_ids, input_lens, ctc_lens
        )

        # ── Diversity loss (blank-collapse prevention) ───────────────────────
        probs      = F.softmax(ctc_logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        ent        = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        loss_div   = -ent.mean()

        # ── Phase 1: skip LLM entirely ───────────────────────────────────────
        if input_ids is None:
            zero = torch.tensor(0.0, device=device)
            return zero, loss_ctc, zero, loss_div, None

        # ── Phase 2: CTC-guided denoising LLM path ──────────────────────────
        prefix = self.llm_proj(self.bridge(v_seq))  # (B, 128, 896)

        # Prefix dropout: zero out brain prefix 50% of the time during training
        if prefix_dropout and self.training:
            mask = torch.rand(B, 1, 1, device=device) > 0.5  # True = keep
            prefix = prefix * mask.to(prefix.dtype)

        embed_fn = self.llm.get_input_embeddings()
        tok_embs = embed_fn(input_ids)  # (B, seq, 896)

        # InfoNCE with memory bank
        prefix_mean = F.normalize(prefix.mean(dim=1), dim=-1)
        tok_mean    = F.normalize(tok_embs.mean(dim=1), dim=-1)

        if neg_bank is not None and neg_bank.shape[0] > 1:
            all_text = torch.cat([tok_mean, neg_bank.to(device)], dim=0)
            sims     = prefix_mean @ all_text.T / 0.07
            tgt_nce  = torch.arange(B, device=device)
        else:
            sims    = prefix_mean @ tok_mean.T / 0.07
            tgt_nce = torch.arange(B, device=device)
        loss_lock = F.cross_entropy(sims, tgt_nce)

        # CTC-guided prompt: [brain_prefix | ctc_hint_tokens | target_text]
        # ctc_prompt_ids contains the tokenized CTC hint prompt
        ctc_embs = embed_fn(ctc_prompt_ids)  # (B, prompt_len, 896)

        combined = torch.cat([prefix, ctc_embs, tok_embs], dim=1)
        # Labels: -100 for prefix + CTC hint (don't predict those), then target text
        prefix_len = prefix.shape[1]
        prompt_len = ctc_embs.shape[1]
        combined_labels = torch.cat([
            torch.full((B, prefix_len + prompt_len), -100, device=device, dtype=labels.dtype),
            labels
        ], dim=1)

        # Attention mask for combined sequence
        combined_attn = torch.cat([
            torch.ones(B, prefix_len, device=device, dtype=torch.long),
            ctc_prompt_mask,
            torch.ones_like(input_ids),
        ], dim=1)

        lm_out = self.llm(inputs_embeds=combined, labels=combined_labels,
                          attention_mask=combined_attn)

        tok_mean_raw = tok_embs.mean(dim=1).detach()
        return lm_out.loss, loss_ctc, loss_lock, loss_div, tok_mean_raw

    @torch.no_grad()
    def generate(self, traj, tokenizer, ctc_vocab, max_tokens=64):
        traj  = traj.to(next(self.parameters()).dtype)
        v_seq = self.encode(traj)

        # CTC decode → subword text
        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)
        ctc_texts  = greedy_ctc_decode_subword(ctc_logits, ctc_vocab)

        # Build CTC-guided prompt for each sample
        prefix = self.llm_proj(self.bridge(v_seq))  # (B, 128, 896)
        embed_fn = self.llm.get_input_embeddings()

        llm_texts = []
        for b in range(traj.shape[0]):
            hint = ctc_texts[b]
            prompt = f'EEG reading (noisy): "{hint}". Reconstruct the original sentence:'
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False,
                                          return_tensors="pt").to(traj.device)
            prompt_embs = embed_fn(prompt_ids)  # (1, prompt_len, 896)

            combined = torch.cat([prefix[b:b+1], prompt_embs], dim=1)
            out_ids = self.llm.generate(
                inputs_embeds=combined, max_new_tokens=max_tokens,
                do_sample=True, temperature=0.6, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
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
            dst_sd[k] = v.clone()
            n_copied += 1
        else:
            n_skipped += 1

    video_enc.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Transferred {n_copied} tensors, skipped {n_skipped}")
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

    combined = np.concatenate([eeg.T, alpha.T, beta.T, gamma.T], axis=1)
    return combined.astype(np.float32)

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, ctc_vocab):
        import mne
        self.samples = []
        self.ctc_vocab = ctc_vocab
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

        print(f"[Dataset] {len(self.samples)} samples (skipped {skipped} placeholders)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ctc_ids = self.ctc_vocab.encode(s['text'])
        return s['eeg'], s['text'], ctc_ids


def collate_fn(batch, tokenizer):
    eegs, texts, ctc_ids_list = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs))

    # CTC targets: pad to max length
    ctc_lens = torch.tensor([len(c) for c in ctc_ids_list], dtype=torch.long)
    max_ctc  = max(ctc_lens)
    ctc_pad  = torch.zeros(len(ctc_ids_list), max_ctc, dtype=torch.long)
    for i, c in enumerate(ctc_ids_list):
        ctc_pad[i, :len(c)] = torch.tensor(c)

    # LLM tokenization
    enc = tokenizer(list(texts), padding=True, truncation=True,
                    max_length=256, return_tensors="pt")
    return eegs, enc.input_ids, enc.input_ids.clone(), ctc_pad, ctc_lens


# ─── CTC hint prompt builder ─────────────────────────────────────────────────

def build_ctc_hint_batch(model, traj, ctc_vocab, tokenizer, device):
    """
    Run CTC decode on current batch, build tokenized hint prompts.
    Returns: ctc_prompt_ids (B, max_prompt_len), ctc_prompt_mask (B, max_prompt_len)
    """
    with torch.no_grad():
        v_seq = model.encode(traj)
        ctc_logits = model.ctc_head(v_seq).transpose(0, 1)
        ctc_texts = greedy_ctc_decode_subword(ctc_logits, ctc_vocab)

    prompts = []
    for hint in ctc_texts:
        prompt = f'EEG reading (noisy): "{hint}". Reconstruct the original sentence:'
        prompts.append(prompt)

    enc = tokenizer(prompts, padding=True, truncation=True,
                    max_length=128, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)


# ─── Training loop ───────────────────────────────────────────────────────────

ACCUM_STEPS = 8

def _train_phase(model, phase, epochs, train_loader, val_loader,
                 tokenizer, ctc_vocab, ckpt_prefix, mem_bank=None):
    device = next(model.parameters()).device

    if phase == 1:
        print("── Phase 1: Subword CTC (Structural Hardening) ──")
        for name, p in model.named_parameters():
            p.requires_grad = ("llm" not in name and "llm_proj" not in name)
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_train_params = sum(p.numel() for p in trainable)
        print(f"  Phase 1 trainable params: {n_train_params/1e6:.1f}M")
        optimizer = torch.optim.AdamW(trainable, lr=5e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=5e-4,
            steps_per_epoch=len(train_loader), epochs=epochs,
            pct_start=0.1,
        )
    else:
        print("── Phase 2: CTC-Guided Denoising (LM + CTC + InfoNCE) ──")
        # Freeze everything first
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze only semantic pathway
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
                    _, loss_ctc, _, loss_div, _ = model(eeg, c_ids, c_lens)
                    loss = (10.0 * loss_ctc + 0.5 * loss_div) / ACCUM_STEPS
                    loss_lm = loss_lock = torch.tensor(0.0)
                    tok_mean = None
                else:
                    # Build CTC hint prompts from current encoder state
                    ctc_prompt_ids, ctc_prompt_mask = build_ctc_hint_batch(
                        model, eeg, ctc_vocab, tokenizer, device
                    )
                    # Prefix dropout: 50% of batches
                    do_prefix_drop = (torch.rand(1).item() < 0.5)

                    neg_bank = mem_bank.get(device) if mem_bank else None
                    loss_lm, loss_ctc, loss_lock, loss_div, tok_mean = model(
                        eeg, c_ids, c_lens, input_ids, labels, neg_bank=neg_bank,
                        ctc_prompt_ids=ctc_prompt_ids,
                        ctc_prompt_mask=ctc_prompt_mask,
                        prefix_dropout=do_prefix_drop,
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

        if phase == 1 and (tot_ctc / n) < best_ctc:
            best_ctc = tot_ctc / n
            torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_best.pt")
            print(f"  ✓ New best CTC {best_ctc:.3f} → {ckpt_prefix}_phase1_best.pt")

        # Validation every 3 epochs
        if epoch % 3 == 0:
            model.eval()
            cer_scores = []
            wer_scores = []
            print(f"── Val Epoch {epoch} ──")
            with torch.no_grad():
                shown = 0
                for eeg, input_ids, labels, c_ids, c_lens in val_loader:
                    eeg_dev = eeg.to(device)
                    llm_preds, ctc_preds = model.generate(eeg_dev, tokenizer, ctc_vocab)
                    for b in range(eeg.shape[0]):
                        ref = tokenizer.decode(input_ids[b], skip_special_tokens=True)
                        cer = compute_cer(ctc_preds[b], ref.lower())
                        wer = compute_wer(ctc_preds[b], ref.lower())
                        cer_scores.append(cer)
                        wer_scores.append(wer)
                        if len(ref.split()) > 3 or shown < 2:
                            print(f"  REF: '{ref[:100]}'")
                            print(f"  CTC: '{ctc_preds[b][:100]}'  CER={cer:.2f} WER={wer:.2f}")
                            print(f"  GEN: '{llm_preds[b][:100]}'")
                            shown += 1
                        if shown >= 4: break
                    if shown >= 4: break
            if cer_scores:
                print(f"  Mean CER: {np.mean(cer_scores):.3f}  Mean WER: {np.mean(wer_scores):.3f}")
            torch.save(model.state_dict(), f"{ckpt_prefix}_phase{phase}_ep{epoch}.pt")

        # Phase 2: save best by LM loss
        if phase == 2:
            avg_lm = tot_lm / n
            if not hasattr(_train_phase, '_best_lm') or avg_lm < _train_phase._best_lm:
                _train_phase._best_lm = avg_lm
                torch.save(model.state_dict(), f"{ckpt_prefix}_phase2_best.pt")
                print(f"  ✓ New best LM {avg_lm:.3f} → {ckpt_prefix}_phase2_best.pt")


# ─── Modal functions ──────────────────────────────────────────────────────────

GPU_CFG = dict(image=image, gpu="H100", timeout=72000,
               volumes={"/data": data_vol, "/persist": ckpt_vol})

@app.function(**GPU_CFG)
def train_v30(phase: int = 1, epochs: int = 40):
    from transformers import AutoTokenizer
    q_name    = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(q_name)

    ctc_vocab = SubwordCTCVocab(tokenizer, max_vocab=80)

    download_zuco(Path("/data/EEG_Text"), data_vol)
    ds         = EEGDataset("/data/EEG_Text", ctc_vocab)
    n_train    = int(0.9 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])
    mkloader   = lambda d, shuf: torch.utils.data.DataLoader(
        d, batch_size=4, shuffle=shuf, collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    train_loader = mkloader(train_ds, True)
    val_loader   = mkloader(val_ds,   False)

    model = EEG_VLM_V30(q_name, ctc_vocab.vocab_size).to(torch.bfloat16).cuda()

    if phase == 1:
        load_pretrained_videomae_encoder(model.video_enc)
    else:
        ckpt = "/persist/v31_phase1_best.pt"
        if Path(ckpt).exists():
            model.load_state_dict(torch.load(ckpt, map_location="cuda"))
            print(f"Loaded Phase 1 checkpoint: {ckpt}")
        else:
            print("WARNING: Phase 1 checkpoint not found — starting Phase 2 from scratch")

    mem_bank = MemoryBank(size=512, dim=896) if phase == 2 else None
    _train_phase(model, phase=phase, epochs=epochs,
                 train_loader=train_loader, val_loader=val_loader,
                 tokenizer=tokenizer, ctc_vocab=ctc_vocab,
                 ckpt_prefix="/persist/v31", mem_bank=mem_bank)


@app.function(image=image, gpu="H100", timeout=86400,
              volumes={"/data": data_vol, "/persist": ckpt_vol},
              retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=10.0))
def run_pipeline(epochs_p1: int = 40, epochs_p2: int = 40):
    from transformers import AutoTokenizer
    q_name    = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(q_name)

    ctc_vocab = SubwordCTCVocab(tokenizer, max_vocab=300)

    download_zuco(Path("/data/EEG_Text"), data_vol)
    ds         = EEGDataset("/data/EEG_Text", ctc_vocab)
    n_train    = int(0.9 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds) - n_train])
    mkloader   = lambda d, shuf: torch.utils.data.DataLoader(
        d, batch_size=4, shuffle=shuf, collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    train_loader = mkloader(train_ds, True)
    val_loader   = mkloader(val_ds,   False)

    model = EEG_VLM_V30(q_name, ctc_vocab.vocab_size).to(torch.bfloat16).cuda()

    # ── Resume logic: find latest checkpoint to skip completed work ──────────
    import glob as _glob
    p1_ckpts = sorted(_glob.glob("/persist/v30_phase1_ep*.pt"))
    p2_ckpts = sorted(_glob.glob("/persist/v30_phase2_ep*.pt"))
    best_p1  = "/persist/v30_phase1_best.pt"
    final    = "/persist/v30_final.pt"

    if Path(final).exists():
        print("✓ V3.0 already complete — nothing to do.")
        return

    skip_p1 = False
    skip_p2 = False

    if p2_ckpts:
        # Phase 2 was in progress — resume Phase 2
        last_p2 = p2_ckpts[-1]
        last_p2_epoch = int(last_p2.split("_ep")[-1].replace(".pt", ""))
        model.load_state_dict(torch.load(last_p2, map_location="cuda"))
        print(f"✓ Resuming Phase 2 from epoch {last_p2_epoch}: {last_p2}")
        skip_p1 = True
        epochs_p2_remaining = epochs_p2 - last_p2_epoch
        if epochs_p2_remaining <= 0:
            skip_p2 = True
    elif Path(best_p1).exists() and p1_ckpts:
        last_p1 = p1_ckpts[-1]
        last_p1_epoch = int(last_p1.split("_ep")[-1].replace(".pt", ""))
        if last_p1_epoch >= epochs_p1 - 3:
            # Phase 1 close enough to done — load best and move to Phase 2
            model.load_state_dict(torch.load(best_p1, map_location="cuda"))
            print(f"✓ Phase 1 complete (best ckpt loaded). Starting Phase 2.")
            skip_p1 = True
            epochs_p2_remaining = epochs_p2
        else:
            # Phase 1 was interrupted — resume
            model.load_state_dict(torch.load(last_p1, map_location="cuda"))
            print(f"✓ Resuming Phase 1 from epoch {last_p1_epoch}: {last_p1}")
            epochs_p1 = epochs_p1 - last_p1_epoch
    else:
        load_pretrained_videomae_encoder(model.video_enc)
        epochs_p2_remaining = epochs_p2

    print(f"\n{'='*60}")
    print(f" V3.0 PIPELINE {'(RESUMED)' if (skip_p1 or p1_ckpts) else ''}")
    print(f" Subword CTC ({ctc_vocab.vocab_size} tokens) + CTC-Guided Denoising")
    print(f"{'='*60}\n")

    # ── Phase 1 ─────────────────────────────────────────────────────────────
    if not skip_p1:
        _train_phase(model, phase=1, epochs=epochs_p1,
                     train_loader=train_loader, val_loader=val_loader,
                     tokenizer=tokenizer, ctc_vocab=ctc_vocab,
                     ckpt_prefix="/persist/v30")
        if Path(best_p1).exists():
            model.load_state_dict(torch.load(best_p1, map_location="cuda"))
            print(f"\n✓ Loaded best Phase 1 weights for Phase 2\n")

    # ── Phase 2 ─────────────────────────────────────────────────────────────
    if not skip_p2:
        mem_bank = MemoryBank(size=512, dim=896)
        remaining = epochs_p2_remaining if skip_p1 else epochs_p2
        _train_phase(model, phase=2, epochs=remaining,
                     train_loader=train_loader, val_loader=val_loader,
                     tokenizer=tokenizer, ctc_vocab=ctc_vocab,
                     ckpt_prefix="/persist/v30", mem_bank=mem_bank)

    torch.save(model.state_dict(), "/persist/v30_final.pt")
    print("\n✓ Full V3.0 pipeline complete. Saved /persist/v30_final.pt")


@app.local_entrypoint()
def main(mode: str = "pipeline", epochs_p1: int = 40, epochs_p2: int = 40,
         phase: int = 1, epochs: int = 40):
    if mode == "pipeline":
        print(f"Launching V3.0 full pipeline: Phase1={epochs_p1}ep, Phase2={epochs_p2}ep")
        run_pipeline.remote(epochs_p1=epochs_p1, epochs_p2=epochs_p2)
    elif mode == "p1":
        print(f"Launching V3.0 Phase 1 only: {epochs}ep")
        train_v30.remote(phase=1, epochs=epochs)
    elif mode == "p2":
        print(f"Launching V3.0 Phase 2 only: {epochs}ep")
        train_v30.remote(phase=2, epochs=epochs)
    else:
        print(f"Unknown mode '{mode}'. Use: pipeline / p1 / p2")
