"""
EEG-VLM V5 — Proper Benchmark Evaluation
==========================================
Standard metrics matching published EEG-to-text literature:
  1. BLEU-1/2/3/4 (translation quality)
  2. ROUGE-1/L (overlap with reference)
  3. WER (word error rate via jiwer)
  4. Top-K retrieval accuracy (contrastive: EEG embedding vs text embeddings)
  5. Exact match / partial match
  6. Cosine similarity of EEG embeddings to text embeddings (semantic)

Runs on the saved best checkpoint from training.
"""

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from pathlib import Path

app = modal.App("eeg-vlm-v5-benchmark")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v5", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.3.0", "transformers>=4.40", "peft>=0.11",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "sentence-transformers", "accelerate", "rouge-score", "nltk"
    ])
)

# Import architecture + data utils from main script
# (We duplicate the minimal needed code to avoid import complexity on Modal)

# ═══════════════════════════════════════════════════════════════════════════════
# Architecture (copied from eeg_vlm_v5_modal.py)
# ═══════════════════════════════════════════════════════════════════════════════

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

class MultiScaleRasterizer(nn.Module):
    def __init__(self, sizes=(32, 64), n_electrodes=64):
        super().__init__()
        self.sizes = sizes
        for s in sizes:
            angles = torch.linspace(0, 2 * np.pi, n_electrodes)
            pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
            gx, gy = torch.meshgrid(
                torch.linspace(-1, 1, s), torch.linspace(-1, 1, s), indexing='ij'
            )
            px, py = gx.flatten().unsqueeze(1), gy.flatten().unsqueeze(1)
            ex, ey = pos_2d[:, 0].unsqueeze(0), pos_2d[:, 1].unsqueeze(0)
            dist = torch.sqrt((px - ex)**2 + (py - ey)**2)
            w = 1.0 / (dist + 1e-4) ** 2.0
            r = torch.sqrt(px**2 + py**2)
            w[(r > 1.1).squeeze(1), :] = 0.0
            w = w / (w.sum(dim=1, keepdim=True).clamp(min=1e-8))
            self.register_buffer(f"W_{s}", w)

    def rasterize(self, x, size):
        B, T, Ch = x.shape
        W = getattr(self, f"W_{size}")
        imgs = (x.reshape(B*T, Ch) @ W.T.to(x.dtype)).reshape(B, T, 1, size, size)
        return imgs

    def forward(self, x):
        return {s: self.rasterize(x, s) for s in self.sizes}

class FrequencyDecomposer(nn.Module):
    def __init__(self, n_bands=4, kernel_size=15):
        super().__init__()
        self.bands = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
            for _ in range(n_bands)
        ])

    def forward(self, x):
        B, T, Ch = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * Ch, 1, T)
        bands = []
        for band_conv in self.bands:
            filtered = band_conv(x_flat)
            bands.append(filtered.reshape(B, Ch, T).permute(0, 2, 1))
        return torch.stack(bands, dim=1)

class SpatioTemporalVideoEncoder(nn.Module):
    def __init__(self, d_model=768, n_heads=12, n_layers=6, ff_dim=3072, n_temporal_frames=32):
        super().__init__()
        self.d_model = d_model
        self.n_temporal_frames = n_temporal_frames
        self.spatial_32 = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(1024, d_model // 2),
        )
        self.spatial_64 = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.GELU(), nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(2048, d_model // 2),
        )
        self.temporal_pool = nn.AvgPool1d(kernel_size=128 // n_temporal_frames, stride=128 // n_temporal_frames)
        self.pos_embed = nn.Parameter(torch.randn(1, n_temporal_frames, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            batch_first=True, dropout=0.1, activation='gelu', norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, multi_scale_imgs, freq_bands):
        B, T = multi_scale_imgs[32].shape[:2]
        imgs_32 = multi_scale_imgs[32]; imgs_64 = multi_scale_imgs[64]
        band_weights = freq_bands.mean(dim=3).permute(0, 2, 1)
        band_weights = torch.sigmoid(band_weights)
        imgs_32_4ch = imgs_32.expand(-1, -1, 4, -1, -1) * band_weights.unsqueeze(-1).unsqueeze(-1)
        imgs_64_4ch = imgs_64.expand(-1, -1, 4, -1, -1) * band_weights.unsqueeze(-1).unsqueeze(-1)
        feat_32 = self.spatial_32(imgs_32_4ch.reshape(B*T, 4, 32, 32))
        feat_64 = self.spatial_64(imgs_64_4ch.reshape(B*T, 4, 64, 64))
        spatial_feat = torch.cat([feat_32, feat_64], dim=-1).reshape(B, T, self.d_model)
        temporal_feat = self.temporal_pool(spatial_feat.transpose(1, 2)).transpose(1, 2)
        temporal_feat = temporal_feat + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls, temporal_feat], dim=1)
        out = self.transformer(seq)
        return self.ln(out)

class QFormerBridge(nn.Module):
    def __init__(self, d_model=768, n_heads=12, n_queries=64, n_layers=2):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'self_ln': nn.LayerNorm(d_model),
                'cross_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'cross_ln': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)),
                'ffn_ln': nn.LayerNorm(d_model),
            }))

    def forward(self, video_features):
        B = video_features.shape[0]
        q = self.queries.expand(B, -1, -1)
        for layer in self.layers:
            q_normed = layer['self_ln'](q)
            sa_out, _ = layer['self_attn'](q_normed, q_normed, q_normed)
            q = q + sa_out
            q_normed = layer['cross_ln'](q)
            ca_out, _ = layer['cross_attn'](q_normed, video_features, video_features)
            q = q + ca_out
            q_normed = layer['ffn_ln'](q)
            q = q + layer['ffn'](q_normed)
        return q

class SubjectAdversarial(nn.Module):
    def __init__(self, d_model, n_subjects):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, n_subjects)
        )
    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha)
        return self.classifier(x)

class EEG_VLM_V5(nn.Module):
    def __init__(self, n_subjects, n_classes, qwen_name, tokenizer):
        super().__init__()
        d_model = 768
        self.rasterizer = MultiScaleRasterizer(sizes=(32, 64))
        self.freq_decomposer = FrequencyDecomposer(n_bands=4, kernel_size=15)
        self.video_encoder = SpatioTemporalVideoEncoder(d_model=d_model)
        self.bridge = QFormerBridge(d_model=d_model)
        self.llm_proj = nn.Linear(d_model, 3584)
        self.grounding_proj = nn.Linear(d_model, 768)
        self.classifier = nn.Linear(d_model, n_classes)
        self.subject_adv = SubjectAdversarial(d_model, n_subjects)
        self.log_temp = nn.Parameter(torch.tensor(np.log(1/0.07)))

        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        llm_base = AutoModelForCausalLM.from_pretrained(
            qwen_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        llm_base.config.use_cache = False
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]
        )
        self.llm = get_peft_model(llm_base, lora_cfg)
        self._tokenizer = tokenizer
        self._anchor_ids = None

    def encode_eeg(self, traj):
        freq_bands = self.freq_decomposer(traj)
        multi_imgs = self.rasterizer(traj)
        video_feat = self.video_encoder(multi_imgs, freq_bands)
        bridged = self.bridge(video_feat)
        return bridged, video_feat[:, 0]

    @torch.no_grad()
    def generate(self, traj, tokenizer, max_tokens=40, num_beams=5):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            bridged, _ = self.encode_eeg(traj)
            B = traj.shape[0]; prefix = self.llm_proj(bridged)
            llm_dev = self.llm.base_model.model.model.embed_tokens.weight.device
            prefix = prefix.to(llm_dev, dtype=torch.bfloat16)
            try: embed_fn = self.llm.model.model.embed_tokens
            except AttributeError: embed_fn = self.llm.base_model.model.model.embed_tokens
            anchor_text = "Based on the brain activity above, the person was reading: "
            anchor = tokenizer(anchor_text, return_tensors="pt", add_special_tokens=False).input_ids.to(llm_dev)
            anchor_emb = embed_fn(anchor).expand(B, -1, -1)
            combined = torch.cat([prefix, anchor_emb], dim=1)
            gen = self.llm.generate(
                inputs_embeds=combined, max_new_tokens=max_tokens,
                num_beams=num_beams, repetition_penalty=2.0,
                no_repeat_ngram_size=3, early_stopping=True,
                length_penalty=1.0,
            )
            return tokenizer.batch_decode(gen, skip_special_tokens=True)

    @torch.no_grad()
    def get_eeg_embedding(self, traj):
        """Get the grounding projection embedding for retrieval metrics."""
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, cls_token = self.encode_eeg(traj)
            return self.grounding_proj(cls_token).float()

    @torch.no_grad()
    def get_classifier_logits(self, traj):
        """Get classification logits for top-k accuracy."""
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _, cls_token = self.encode_eeg(traj)
            return self.classifier(cls_token).float()


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading (from main script)
# ═══════════════════════════════════════════════════════════════════════════════

def load_mat_any(path):
    import scipy.io as sio; import h5py
    try:
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        def extract_sentences(arr):
            if not isinstance(arr, np.ndarray): arr = np.array([arr])
            for item in arr.flat:
                text = getattr(item, 'content', None)
                if not text or not isinstance(text, str): continue
                eeg = None
                for field in ['rawData', 'mean_EEG', 'processedEEG', 'RawEEG']:
                    val = getattr(item, field, None)
                    if val is not None and isinstance(val, np.ndarray) and val.ndim == 2 and min(val.shape) > 5:
                        eeg = np.array(val, dtype=np.float32); break
                if eeg is not None: yield eeg, text
        for root_key in ['sentenceData', 'snData']:
            root = mat.get(root_key)
            if root is not None: yield from extract_sentences(root); return
        results = mat.get('results')
        if results is not None:
            for sub_key in ['sentenceData', 'snData']:
                sub = getattr(results, sub_key, None)
                if sub is not None: yield from extract_sentences(sub); return
    except NotImplementedError: pass
    except Exception as e: print(f"  scipy warning: {e}")

def normalize_eeg(eeg, target_ch=64, target_t=128):
    if eeg.ndim != 2: return None
    ch, t = eeg.shape
    if ch < t: pass
    else: eeg = eeg.T; ch, t = eeg.shape
    if ch > target_ch: eeg = eeg[:target_ch, :]
    elif ch < target_ch: eeg = np.pad(eeg, ((0, target_ch - ch), (0, 0)))
    if t > target_t: eeg = eeg[:, :target_t]
    elif t < target_t: eeg = np.pad(eeg, ((0, 0), (0, target_t - t)))
    mean = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True)
    eeg = (eeg - mean) / (std + 1e-8)
    return eeg.T.astype(np.float32)

def parse_data(base_path):
    import mne; final_data = []
    for zuco_path in [base_path/"ZuCo_v1", base_path/"ZuCo_v2"]:
        if not zuco_path.exists(): continue
        for p in sorted(zuco_path.rglob("*.mat")):
            count = 0
            for eeg, text in load_mat_any(p):
                normed = normalize_eeg(eeg)
                if normed is not None:
                    final_data.append({'eeg': normed, 'text': text, 'sub': p.stem[:5], 'is_sentence': True})
                    count += 1
            print(f"  Parsed {count} sentences from {p.name}")
    print(f"ZuCo total: {len(final_data)} sentences")
    is_path = base_path / "InnerSpeech"
    is_count = 0
    for p in sorted(is_path.rglob("*-epo.fif")):
        try:
            epochs = mne.read_epochs(str(p), preload=True, verbose=False)
            epochs.resample(128)
            data = epochs.get_data()
            if epochs.metadata is not None and 'condition' in epochs.metadata.columns:
                labels = epochs.metadata['condition'].tolist()
            elif hasattr(epochs, 'event_id') and epochs.event_id:
                id_to_label = {v: k for k, v in epochs.event_id.items()}
                labels = [id_to_label.get(e, f"event_{e}") for e in epochs.events[:, 2]]
            else:
                labels = [f"epoch_{i}" for i in range(len(data))]
            sub = p.stem.split('_')[0] if '_' in p.stem else 'sub01'
            for d, l in zip(data, labels):
                normed = normalize_eeg(d)
                if normed is not None:
                    final_data.append({'eeg': normed, 'text': str(l), 'sub': sub, 'is_sentence': False})
                    is_count += 1
        except: continue
    print(f"InnerSpeech total: {is_count} | TOTAL: {len(final_data)}")
    return final_data


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@app.function(image=image, gpu="H100", volumes={"/data": data_vol, "/persist": ckpt_vol}, timeout=7200, memory=65536)
def benchmark():
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    from jiwer import wer as compute_wer
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    device = torch.device("cuda:0")
    base_path = Path("/data/EEG_Text")

    # ── Load data ──
    all_samples = parse_data(base_path)
    if not all_samples:
        print("FAILURE: No data found."); return

    texts = [s['text'] for s in all_samples]
    eegs = np.array([s['eeg'] for s in all_samples])
    subjects = [s['sub'] for s in all_samples]
    unique_labels = sorted(list(set(texts)))
    label_to_id = {l: i for i, l in enumerate(unique_labels)}
    class_ids = np.array([label_to_id[t] for t in texts])
    unique_subs = sorted(list(set(subjects)))

    # Recreate same split (use fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(texts))
    val_idx = indices[int(0.8*len(indices)):]
    print(f"\nEval set: {len(val_idx)} samples | Unique labels: {len(unique_labels)}")

    # ── Load model ──
    qwen_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(qwen_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = EEG_VLM_V5(
        n_subjects=len(unique_subs), n_classes=len(unique_labels),
        qwen_name=qwen_name, tokenizer=tokenizer
    ).to(device)
    model.llm.to(device, dtype=torch.bfloat16)

    ckpt_path = Path("/persist/eeg_vlm_v5_best.pt")
    if ckpt_path.exists():
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        print("✓ Loaded best checkpoint")
    else:
        print("⚠ No checkpoint found, evaluating random init")

    model.eval()

    # ── Teacher embeddings for retrieval ──
    print("Computing teacher text embeddings...")
    teacher = SentenceTransformer('all-mpnet-base-v2').to(device)
    all_label_embs = F.normalize(
        torch.from_numpy(teacher.encode(unique_labels)).float().to(device), dim=-1
    )
    del teacher; torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════
    # METRIC 1: Classification Top-K Accuracy
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("METRIC 1: Classification Top-K Accuracy (classifier head)")
    print("="*70)

    top1_correct = 0; top5_correct = 0; top10_correct = 0
    n_cls = 0
    for i in range(0, len(val_idx), 32):
        b_idx = val_idx[i:i+32]
        t_eeg = torch.from_numpy(eegs[b_idx]).to(device, dtype=torch.bfloat16)
        logits = model.get_classifier_logits(t_eeg)
        gt = torch.tensor(class_ids[b_idx], device=device)
        _, top10 = logits.topk(10, dim=-1)
        _, top5 = logits.topk(5, dim=-1)
        top1_correct += (logits.argmax(-1) == gt).sum().item()
        top5_correct += (top5 == gt.unsqueeze(1)).any(dim=1).sum().item()
        top10_correct += (top10 == gt.unsqueeze(1)).any(dim=1).sum().item()
        n_cls += len(b_idx)

    print(f"  Top-1 Accuracy: {top1_correct/n_cls:.1%} ({top1_correct}/{n_cls})")
    print(f"  Top-5 Accuracy: {top5_correct/n_cls:.1%} ({top5_correct}/{n_cls})")
    print(f"  Top-10 Accuracy: {top10_correct/n_cls:.1%} ({top10_correct}/{n_cls})")

    # ════════════════════════════════════════════════════════════════════
    # METRIC 2: Contrastive Retrieval (EEG embedding vs text embeddings)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("METRIC 2: Contrastive Retrieval Accuracy (EEG↔Text cosine)")
    print("="*70)

    ret_top1 = 0; ret_top5 = 0; ret_top10 = 0
    n_ret = 0
    for i in range(0, len(val_idx), 32):
        b_idx = val_idx[i:i+32]
        t_eeg = torch.from_numpy(eegs[b_idx]).to(device, dtype=torch.bfloat16)
        eeg_embs = model.get_eeg_embedding(t_eeg)
        eeg_norm = F.normalize(eeg_embs, dim=-1)
        # Cosine similarity against ALL label embeddings
        sim = eeg_norm @ all_label_embs.T  # (B, n_labels)
        gt = torch.tensor(class_ids[b_idx], device=device)
        _, top10 = sim.topk(10, dim=-1)
        _, top5 = sim.topk(5, dim=-1)
        ret_top1 += (sim.argmax(-1) == gt).sum().item()
        ret_top5 += (top5 == gt.unsqueeze(1)).any(dim=1).sum().item()
        ret_top10 += (top10 == gt.unsqueeze(1)).any(dim=1).sum().item()
        n_ret += len(b_idx)

    print(f"  Retrieval Top-1: {ret_top1/n_ret:.1%} ({ret_top1}/{n_ret})")
    print(f"  Retrieval Top-5: {ret_top5/n_ret:.1%} ({ret_top5}/{n_ret})")
    print(f"  Retrieval Top-10: {ret_top10/n_ret:.1%} ({ret_top10}/{n_ret})")

    # ════════════════════════════════════════════════════════════════════
    # METRIC 3: Generative (BLEU, ROUGE, WER) — on subset
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("METRIC 3: Generative Decoding (BLEU / ROUGE / WER)")
    print("="*70)

    N_GEN = min(200, len(val_idx))  # Generate is slow, cap at 200
    smoother = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    all_bleu1 = []; all_bleu2 = []; all_bleu3 = []; all_bleu4 = []
    all_rouge1_f = []; all_rougeL_f = []
    all_wer = []
    exact_match = 0
    word_overlap = []
    predictions = []; references = []

    for i in range(N_GEN):
        idx = val_idx[i]
        t_eeg = torch.from_numpy(eegs[idx:idx+1]).to(device, dtype=torch.bfloat16)
        pred_raw = model.generate(t_eeg, tokenizer, max_tokens=40, num_beams=5)[0]
        # Clean prediction
        pred = pred_raw.strip()
        if "[EEG: ]" in pred:
            pred = pred.split("[EEG: ]")[-1].strip()
        if "[EEG:" in pred:
            pred = pred.split("[EEG:")[-1].strip("] ")

        ref = texts[idx].strip()
        predictions.append(pred)
        references.append(ref)

        # Tokenize
        ref_tokens = ref.lower().split()
        pred_tokens = pred.lower().split()

        if not pred_tokens:
            pred_tokens = ["<empty>"]

        # BLEU
        b1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother)
        b2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
        b3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.34, 0), smoothing_function=smoother)
        b4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
        all_bleu1.append(b1); all_bleu2.append(b2); all_bleu3.append(b3); all_bleu4.append(b4)

        # ROUGE
        scores = scorer.score(ref, pred)
        all_rouge1_f.append(scores['rouge1'].fmeasure)
        all_rougeL_f.append(scores['rougeL'].fmeasure)

        # WER
        try:
            w = compute_wer(ref.lower(), pred.lower())
            all_wer.append(min(w, 2.0))  # cap at 200%
        except:
            all_wer.append(2.0)

        # Exact match
        if pred.lower().strip() == ref.lower().strip():
            exact_match += 1

        # Word overlap
        ref_set = set(ref_tokens)
        pred_set = set(pred_tokens)
        if ref_set:
            overlap = len(ref_set & pred_set) / len(ref_set)
            word_overlap.append(overlap)

    print(f"\n  Generated {N_GEN} predictions")
    print(f"\n  --- Translation Metrics ---")
    print(f"  BLEU-1: {np.mean(all_bleu1)*100:.1f}")
    print(f"  BLEU-2: {np.mean(all_bleu2)*100:.1f}")
    print(f"  BLEU-3: {np.mean(all_bleu3)*100:.1f}")
    print(f"  BLEU-4: {np.mean(all_bleu4)*100:.1f}")
    print(f"\n  --- ROUGE ---")
    print(f"  ROUGE-1 F: {np.mean(all_rouge1_f)*100:.1f}")
    print(f"  ROUGE-L F: {np.mean(all_rougeL_f)*100:.1f}")
    print(f"\n  --- Error Metrics ---")
    print(f"  WER: {np.mean(all_wer)*100:.1f}%")
    print(f"  Exact Match: {exact_match/N_GEN:.1%} ({exact_match}/{N_GEN})")
    print(f"  Word Overlap: {np.mean(word_overlap)*100:.1f}%")

    # ════════════════════════════════════════════════════════════════════
    # METRIC 4: Semantic Similarity (cosine between generated & reference)
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("METRIC 4: Semantic Similarity (sentence embeddings)")
    print("="*70)

    sem_teacher = SentenceTransformer('all-mpnet-base-v2').to(device)
    pred_embs = sem_teacher.encode(predictions, show_progress_bar=False)
    ref_embs = sem_teacher.encode(references, show_progress_bar=False)
    cosine_sims = []
    for p, r in zip(pred_embs, ref_embs):
        cos = np.dot(p, r) / (np.linalg.norm(p) * np.linalg.norm(r) + 1e-8)
        cosine_sims.append(cos)
    del sem_teacher; torch.cuda.empty_cache()

    print(f"  Mean Cosine Similarity: {np.mean(cosine_sims):.3f}")
    print(f"  Median Cosine Similarity: {np.median(cosine_sims):.3f}")
    print(f"  >0.5 similarity: {sum(1 for c in cosine_sims if c > 0.5)/len(cosine_sims):.1%}")
    print(f"  >0.7 similarity: {sum(1 for c in cosine_sims if c > 0.7)/len(cosine_sims):.1%}")

    # ════════════════════════════════════════════════════════════════════
    # Sample Predictions
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS (first 20)")
    print("="*70)
    for i in range(min(20, N_GEN)):
        cos = cosine_sims[i]
        print(f"\n  [{i+1}] REF: {references[i][:100]}")
        print(f"       GEN: {predictions[i][:100]}")
        print(f"       BLEU-1={all_bleu1[i]:.2f} | ROUGE-1={all_rouge1_f[i]:.2f} | WER={all_wer[i]:.1%} | CosSim={cos:.2f}")

    # ════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"  Dataset:        {len(all_samples)} total | {len(val_idx)} eval")
    print(f"  Unique labels:  {len(unique_labels)}")
    print(f"  Subjects:       {len(unique_subs)}")
    print(f"  Architecture:   EEG-VLM V5 (Topo-Raster → TimeSformer → Q-Former → Qwen2.5-7B)")
    print(f"")
    print(f"  ┌─────────────────────────────────────────┐")
    print(f"  │  Classification                         │")
    print(f"  │    Top-1:  {top1_correct/n_cls:>6.1%}                       │")
    print(f"  │    Top-5:  {top5_correct/n_cls:>6.1%}                       │")
    print(f"  │    Top-10: {top10_correct/n_cls:>6.1%}                       │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Retrieval (EEG↔Text cosine)            │")
    print(f"  │    Top-1:  {ret_top1/n_ret:>6.1%}                       │")
    print(f"  │    Top-5:  {ret_top5/n_ret:>6.1%}                       │")
    print(f"  │    Top-10: {ret_top10/n_ret:>6.1%}                       │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Generative (N={N_GEN})                    │")
    print(f"  │    BLEU-1:    {np.mean(all_bleu1)*100:>5.1f}                    │")
    print(f"  │    BLEU-4:    {np.mean(all_bleu4)*100:>5.1f}                    │")
    print(f"  │    ROUGE-1 F: {np.mean(all_rouge1_f)*100:>5.1f}                    │")
    print(f"  │    ROUGE-L F: {np.mean(all_rougeL_f)*100:>5.1f}                    │")
    print(f"  │    WER:       {np.mean(all_wer)*100:>5.1f}%                   │")
    print(f"  │    Exact:     {exact_match/N_GEN:>6.1%}                    │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  Semantic Similarity                    │")
    print(f"  │    Mean Cosine: {np.mean(cosine_sims):>5.3f}                  │")
    print(f"  │    >0.7 sim:    {sum(1 for c in cosine_sims if c > 0.7)/len(cosine_sims):>5.1%}                  │")
    print(f"  └─────────────────────────────────────────┘")

@app.local_entrypoint()
def main():
    benchmark.remote()
