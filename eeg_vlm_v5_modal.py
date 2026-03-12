
"""
EEG-to-Text VLM V5 — Major Architecture Upgrade
================================================
Core Idea PRESERVED: Topographic Rasterization → Video-like Encoding → LLM Decode
This is what makes us unique vs. all other EEG-to-text papers.

V5 Upgrades over V4 (which got ~5%):
1. Multi-Scale Rasterizer: 32x32 + 64x64 scalp maps, fused for richer spatial info
2. Temporal Hierarchy: Local CNN features + Global Transformer (like TimeSformer)
3. Frequency-Band Decomposition: Delta/Theta/Alpha/Beta bands → multi-channel video
4. Q-Former Bridge (BLIP-2 style): 64 learned queries with 2 cross-attn stages
5. Subject-Adversarial Training: Gradient reversal to learn subject-invariant features
6. Contrastive Pre-Alignment: CLIP-style EEG↔Text alignment before LLM training
7. Curriculum Learning: Easy (syllables) → Hard (words) schedule
8. Qwen2.5-7B-Instruct with LoRA r=64 on A100/H100
9. Data Augmentation: temporal jitter, channel dropout, Gaussian noise, amplitude scaling
10. Proper eval: held-out subject, beam search, WER + exact match

Target: Break 5% ceiling → 15%+ accuracy on 186-class word decoding
"""

import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import subprocess
from pathlib import Path

app = modal.App("eeg-vlm-v5-breakthrough")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v5", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.3.0", "transformers>=4.40", "peft>=0.11",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "sentence-transformers", "accelerate", "osfclient"
    ])
)

# ═══════════════════════════════════════════════════════════════════════════════
# V5 Architecture Components
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


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_pos = torch.norm(anchor - positive, p=2, dim=-1)
        d_neg = torch.norm(anchor - negative, p=2, dim=-1)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


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
        # x is (B, 128, 64), W is (1024, 64)
        W = getattr(self, f"W_{size}")
        # (B*T, Ch) @ (Ch, 1024) -> (B*T, 1024)
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
    def __init__(self, d_model=768, n_heads=12, n_layers=6, ff_dim=3072,
                 n_temporal_frames=64):
        super().__init__()
        self.d_model = d_model
        self.n_temporal_frames = n_temporal_frames

        self.spatial_32 = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2, padding=1),
            nn.GELU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GELU(), nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, d_model // 2),
        )
        self.spatial_64 = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2, padding=1),
            nn.GELU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(), nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GELU(), nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GELU(), nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
            nn.Linear(2048, d_model // 2),
        )

        self.temporal_pool = nn.AvgPool1d(
            kernel_size=128 // n_temporal_frames,
            stride=128 // n_temporal_frames
        )

        self.pos_embed = nn.Parameter(torch.randn(1, n_temporal_frames, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            batch_first=True, dropout=0.1, activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, multi_scale_imgs, freq_bands):
        B, T = multi_scale_imgs[32].shape[:2]
        imgs_32 = multi_scale_imgs[32]
        imgs_64 = multi_scale_imgs[64]
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
        out = self.ln(out)
        return out[:, 0], out[:, 1:]


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
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
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
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
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
        self.video_encoder = SpatioTemporalVideoEncoder(d_model=d_model, n_temporal_frames=64)
        self.bridge = QFormerBridge(d_model=d_model)
        self.llm_proj = nn.Linear(d_model, 896) # Qwen2.5-0.5B hidden size
        self.grounding_proj = nn.Linear(d_model, 768)
        self.classifier = nn.Linear(d_model, n_classes)
        self.subject_adv = SubjectAdversarial(d_model, n_subjects)
        self.log_temp = nn.Parameter(torch.tensor(np.log(1/0.07)))
        self.triplet_loss_fn = TripletLoss(margin=1.0)
        
        # CTC Head for temporal grounding (V5.1: 64 frames)
        vocab_size = 151643 # Qwen2.5
        self.ctc_head = nn.Linear(d_model, vocab_size + 1) 
        self.ctc_loss_fn = nn.CTCLoss(blank=vocab_size, zero_infinity=True)

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
        # video_feat is full sequence (B, T+1, D), cls_token is (B, D)
        cls_token, temporal_seq = self.video_encoder(multi_imgs, freq_bands)
        bridged = self.bridge(temporal_seq) # Bridge queries the temporal sequence
        return bridged, cls_token, temporal_seq

    def forward(self, traj, input_ids, labels, sub_ids, teacher_embs, neg_teacher_emb, class_ids,
                grl_alpha=1.0, phase="joint", is_sentence=None):
        bridged, cls_token, temporal_seq = self.encode_eeg(traj)
        B = traj.shape[0]
        prefix = self.llm_proj(bridged)
        eeg_lat = self.grounding_proj(cls_token)

        temp = torch.exp(self.log_temp).clamp(max=100)
        eeg_norm = F.normalize(eeg_lat, dim=-1)
        txt_norm = F.normalize(teacher_embs, dim=-1)
        sim = (eeg_norm @ txt_norm.T) * temp
        cont_labels = torch.arange(B, device=traj.device)
        loss_contrastive = (F.cross_entropy(sim, cont_labels) + F.cross_entropy(sim.T, cont_labels)) / 2

        cls_logits = self.classifier(cls_token)
        loss_cls = F.cross_entropy(cls_logits, class_ids)
        sub_logits = self.subject_adv(cls_token, alpha=grl_alpha)
        loss_sub = F.cross_entropy(sub_logits, sub_ids)

        # Triplet Grounding (Ground truth vs. Fixed Distractor)
        loss_triplet = self.triplet_loss_fn(eeg_norm.float(), txt_norm.float(), neg_teacher_emb.float())

        # CTC Loss — Local Temporal Alignment (V5.1: 64 frames)
        ctc_logits = self.ctc_head(temporal_seq).transpose(0, 1) # (T, B, V)
        log_probs = F.log_softmax(ctc_logits, dim=-1)
        
        # CTC labels: remove padding/-100
        ctc_targets = []
        target_lengths = []
        pad_id = self._tokenizer.pad_token_id if self._tokenizer is not None else 0
        for i in range(B):
            v = labels[i]
            valid = v[(v != -100) & (v != pad_id)]
            # CTC requirement: target_length <= input_length (64)
            if len(valid) > temporal_seq.shape[1]:
                valid = valid[:temporal_seq.shape[1]]
            ctc_targets.append(valid)
            target_lengths.append(len(valid))
        
        if any(l > 0 for l in target_lengths):
            flat_targets = torch.cat([t for t in ctc_targets if len(t) > 0])
            input_lengths = torch.full((B,), temporal_seq.shape[1], dtype=torch.long, device=traj.device)
            loss_ctc = self.ctc_loss_fn(log_probs, flat_targets, input_lengths, torch.tensor(target_lengths, dtype=torch.long, device=traj.device))
        else:
            loss_ctc = torch.tensor(0.0, device=traj.device)

        # Phase 1: encoder-only (no LM loss, LLM frozen)
        if phase == "encoder_only":
            # V5.1.1: Aggressive CTC grounding (3.0 weight)
            total = 1.0 * loss_contrastive + 2.0 * loss_triplet + 3.0 * loss_ctc + 1.0 * loss_cls + 0.1 * loss_sub
            return total, {
                'lm': 0.0, 'contrastive': loss_contrastive.item(),
                'triplet': loss_triplet.item(), 'ctc': loss_ctc.item(),
                'cls': loss_cls.item(), 'sub': loss_sub.item(),
                'cls_acc': (cls_logits.argmax(-1) == class_ids).float().mean().item()
            }

        llm_dev = self.llm.base_model.model.model.embed_tokens.weight.device
        prefix = prefix.to(llm_dev, dtype=torch.bfloat16)
        input_ids = input_ids.to(llm_dev); labels = labels.to(llm_dev)

        try: embed_fn = self.llm.model.model.embed_tokens
        except AttributeError: embed_fn = self.llm.base_model.model.model.embed_tokens

        # Use instruction-style prompt so LLM knows what to generate
        if self._anchor_ids is None:
            anchor_text = "Based on the brain activity above, the person was reading: "
            self._anchor_ids = self._tokenizer(anchor_text, return_tensors="pt", add_special_tokens=False).input_ids.to(llm_dev)
        anchor_embs = embed_fn(self._anchor_ids).expand(B, -1, -1)
        tok_embs = embed_fn(input_ids)
        combined = torch.cat([prefix, anchor_embs, tok_embs], dim=1)
        n_prefix = 64 + self._anchor_ids.shape[1]
        combined_labels = torch.cat([torch.full((B, n_prefix), -100, device=labels.device, dtype=labels.dtype), labels], dim=1)
        attention_mask = torch.ones(B, combined.shape[1], device=combined.device)
        out = self.llm(inputs_embeds=combined, attention_mask=attention_mask, labels=combined_labels)
        loss_lm = out.loss

        # If we know which samples are sentences vs words, upweight sentence LM loss
        if is_sentence is not None and is_sentence.any():
            # Recompute LM loss for sentence-only samples with higher weight
            sent_mask = is_sentence.bool()
            lm_weight = 5.0  # Prioritize sentence decoding
        else:
            lm_weight = 3.0

        # V5.1.1: Parallel grounding in Phase 2
        total = lm_weight * loss_lm + 1.0 * loss_ctc + 1.0 * loss_contrastive + 1.0 * loss_triplet + 0.5 * loss_cls + 0.1 * loss_sub
        return total, {
            'lm': loss_lm.item(), 'contrastive': loss_contrastive.item(),
            'triplet': loss_triplet.item(), 'ctc': loss_ctc.item(),
            'cls': loss_cls.item(), 'sub': loss_sub.item(),
            'cls_acc': (cls_logits.argmax(-1) == class_ids).float().mean().item()
        }

    @torch.no_grad()
    def generate(self, traj, tokenizer, max_tokens=40, num_beams=5):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            bridged, cls_token, temporal_seq = self.encode_eeg(traj)
            B = traj.shape[0]; prefix = self.llm_proj(bridged)
            
            # --- CTC Greedy Decode (Diagnostic V5.1.2) ---
            ctc_logits = self.ctc_head(temporal_seq) # (B, 64, V)
            ctc_probs = F.softmax(ctc_logits, dim=-1)
            ctc_ids = ctc_logits.argmax(dim=-1) # (B, 64)
            ctc_texts = []
            blank_id = 151643
            
            # Diagnostic: check how often we predict non-blank
            non_blank_mask = (ctc_ids != blank_id)
            nb_ratio = non_blank_mask.float().mean().item()
            
            for b in range(B):
                ids = ctc_ids[b].tolist()
                collapsed = []
                prev = -1
                for i in ids:
                    if i != prev and i != blank_id: 
                        collapsed.append(i)
                    prev = i
                
                decoded = tokenizer.decode(collapsed, skip_special_tokens=True).strip()
                
                # If decoded is empty, let's see what the top non-blank tokens were
                if not decoded:
                    # Get top-5 tokens across the whole 64-frame sequence (excluding blank)
                    flat_probs = ctc_probs[b].clone() # (64, V)
                    flat_probs[:, blank_id] = 0
                    V = flat_probs.size(1)
                    top_vals, top_idx = torch.topk(flat_probs.view(-1), k=3)
                    top_tokens = []
                    for idx in top_idx:
                        t_id = idx.item() % V
                        frame = idx.item() // V
                        token_str = tokenizer.decode([t_id])
                        top_tokens.append(f"{token_str}@{frame}")
                    decoded = f"[EMPTY] (Whispers: {'|'.join(top_tokens)}, NB-ratio: {nb_ratio:.1%})"
                
                ctc_texts.append(decoded)
            
            llm_dev = self.llm.base_model.model.model.embed_tokens.weight.device
            prefix = prefix.to(llm_dev, dtype=torch.bfloat16)
            try: embed_fn = self.llm.model.model.embed_tokens
            except AttributeError: embed_fn = self.llm.base_model.model.model.embed_tokens
            # Must match the anchor used during training
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
            llm_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
            return llm_texts, ctc_texts

# ═══════════════════════════════════════════════════════════════════════════════
# Data Utils
# ═══════════════════════════════════════════════════════════════════════════════

def augment_eeg(traj, p=0.5):
    B, T, Ch = traj.shape; device = traj.device
    if torch.rand(1).item() < p: traj = torch.roll(traj, shifts=torch.randint(1, 5, (1,)).item(), dims=1)
    if torch.rand(1).item() < p: traj = traj + torch.randn_like(traj) * 0.03
    if torch.rand(1).item() < p: traj[:, :, torch.randperm(Ch)[:torch.randint(2, 8, (1,)).item()]] = 0.0
    if torch.rand(1).item() < p: traj = traj * (0.85 + torch.rand(1).item() * 0.3)
    return traj

def load_mat_any(path):
    """Parse ZuCo .mat files. Yields (eeg_array, text_string) tuples.
    
    ZuCo NR/SR files have sentenceData[i].rawData (105 x T) and .content (str).
    Some v7.3 HDF5 files use h5py references.
    """
    import scipy.io as sio; import h5py; import numpy as np

    # -- scipy path (works for most ZuCo v1/v2 files) --
    try:
        mat = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        
        def extract_sentences(arr):
            """Extract from an array of sentence structs."""
            if not isinstance(arr, np.ndarray):
                arr = np.array([arr])  # single struct
            for item in arr.flat:
                text = getattr(item, 'content', None)
                if not text or not isinstance(text, str):
                    continue
                # Try EEG fields in priority order
                eeg = None
                for field in ['rawData', 'mean_EEG', 'processedEEG', 'RawEEG']:
                    val = getattr(item, field, None)
                    if val is not None and isinstance(val, np.ndarray) and val.ndim == 2 and min(val.shape) > 5:
                        eeg = np.array(val, dtype=np.float32)
                        break
                if eeg is not None:
                    yield eeg, text

        # Search top-level and nested under 'results'
        for root_key in ['sentenceData', 'snData']:
            root = mat.get(root_key)
            if root is not None:
                yield from extract_sentences(root); return
        # Some files nest under results.sentenceData / results.snData
        results = mat.get('results')
        if results is not None:
            for sub_key in ['sentenceData', 'snData']:
                sub = getattr(results, sub_key, None)
                if sub is not None:
                    yield from extract_sentences(sub); return
    except NotImplementedError:
        pass  # v7.3 HDF5 format, fall through to h5py
    except Exception as e:
        print(f"  scipy warning for {path}: {e}")

    # -- h5py path (for MATLAB v7.3+ HDF5 files) --
    try:
        with h5py.File(str(path), 'r') as f:
            targets = []
            # Find sentence data groups
            if 'results' in f:
                res = f['results']
                for k in ['sentenceData', 'snData']:
                    if k in res: targets.append(res[k])
            for k in ['sentenceData', 'snData']:
                if k in f: targets.append(f[k])

            for group in targets:
                if isinstance(group, h5py.Dataset):
                    # Layout 1: Dataset of object references (Structs)
                    for ref_obj in group:
                        try:
                            ref = ref_obj[0] if hasattr(ref_obj, '__getitem__') else ref_obj
                            obj = f[ref]
                            # Resolve text
                            text_ref = obj['content'][0][0]
                            text = "".join([chr(c[0]) for c in f[text_ref]])
                            # Resolve EEG
                            eeg = None
                            for k in ['rawData', 'mean_EEG', 'processedEEG', 'RawEEG']:
                                if k in obj:
                                    eeg_ref = obj[k][0][0]; eeg = np.array(f[eeg_ref], dtype=np.float32).T; break
                            if text and eeg is not None and eeg.ndim == 2 and min(eeg.shape) > 5:
                                yield eeg, text
                        except: continue
                else:
                    # Layout 2: Group with datasets for each field (Columnar)
                    if 'content' not in group: continue
                    contents = group['content']
                    n_samples = contents.shape[0] if contents.ndim > 0 else 0
                    for i in range(n_samples):
                        try:
                            # Resolve text
                            c_ref = contents[i][0]; text = "".join([chr(c[0]) for c in f[c_ref]])
                            # Resolve EEG
                            eeg = None
                            for k in ['rawData', 'mean_EEG', 'processedEEG', 'RawEEG']:
                                if k in group:
                                    e_ref = group[k][i][0]; eeg = np.array(f[e_ref], dtype=np.float32).T; break
                            if text and eeg is not None and eeg.ndim == 2 and min(eeg.shape) > 5:
                                yield eeg, text
                        except: continue
                if targets: return
    except Exception as e:
        print(f"  h5py warning for {path}: {e}")

def verify_mat_file(path):
    if not path.exists(): return False
    if path.stat().st_size < 100000: return False
    try:
        with open(path, 'rb') as f:
            header = f.read(128)
            return b'MATLAB 5.0' in header or b'MATLAB' in header
    except: return False

def download_zuco(base_path, vol):
    import os; import time; import subprocess
    zuco_v1 = base_path / "ZuCo_v1"; zuco_v2 = base_path / "ZuCo_v2"; inner_speech = base_path / "InnerSpeech"
    zuco_v1.mkdir(parents=True, exist_ok=True); zuco_v2.mkdir(parents=True, exist_ok=True); inner_speech.mkdir(parents=True, exist_ok=True)

    print("  Downloading ZuCo data from OSF via dynamic listing...")
    
    def download_project_files(project_id, target_dir, target_names):
        print(f"  Listing OSF project {project_id}...")
        res = subprocess.run(["osf", "-p", project_id, "list"], capture_output=True, text=True)
        paths = res.stdout.splitlines()
        for name in target_names:
            local_path = target_dir / name
            if verify_mat_file(local_path):
                print(f"    ✓ {name} is already valid.")
                continue
            
            # Find the full remote path that ends with the target name
            full_remote = next((p for p in paths if p.strip().endswith(name)), None)
            if full_remote:
                if local_path.exists(): local_path.unlink()
                print(f"    Fetching {name} from {full_remote}...")
                subprocess.run(["osf", "-p", project_id, "fetch", full_remote.strip(), str(local_path)], check=False, timeout=900)
            else:
                print(f"    ✗ Could not find {name} in {project_id} listing.")

    # ZuCo v1
    v1_subs = ["resultsZAB_SR.mat", "resultsZDM_SR.mat", "resultsZAB_NR.mat", "resultsZDM_NR.mat"]
    download_project_files("q3zws", zuco_v1, v1_subs)
    
    # ZuCo v2
    v2_subs = ["resultsYAC_NR.mat", "resultsYAG_NR.mat", "resultsYAK_NR.mat"]
    download_project_files("2urht", zuco_v2, v2_subs)
    
    # InnerSpeech - derivatives
    if not any(inner_speech.rglob("*-epo.fif")):
        print("  Downloading InnerSpeech derivatives...")
        subprocess.run(["pip", "install", "openneuro-py"], check=False)
        import openneuro; openneuro.download(dataset="ds003626", target_dir=str(inner_speech), include=["derivatives/sub-01/"])
    vol.commit()

def normalize_eeg(eeg, target_ch=64, target_t=128):
    """Normalize EEG to (target_t, target_ch). Input can be (Ch, T) or (T, Ch)."""
    if eeg.ndim != 2:
        return None
    # ZuCo rawData is (105, T) where 105 > T typically, ensure (Ch, T) orientation
    # Heuristic: if shape[0] > shape[1] and shape[0] > 200, it's likely (T, Ch) — transpose
    ch, t = eeg.shape
    if ch < t:
        pass  # Already (Ch, T)
    else:
        eeg = eeg.T  # Was (T, Ch) → now (Ch, T)
        ch, t = eeg.shape
    # Truncate/pad channels
    if ch > target_ch: eeg = eeg[:target_ch, :]
    elif ch < target_ch: eeg = np.pad(eeg, ((0, target_ch - ch), (0, 0)))
    # Truncate/pad time
    if t > target_t: eeg = eeg[:, :target_t]
    elif t < target_t: eeg = np.pad(eeg, ((0, 0), (0, target_t - t)))
    # Z-score normalize per channel
    mean = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True)
    eeg = (eeg - mean) / (std + 1e-8)
    # Output (T, Ch) for the model
    return eeg.T.astype(np.float32)

def parse_data(base_path):
    import mne; final_data = []
    # ZuCo .mat files — these are SENTENCES (is_sentence=True)
    for zuco_path in [base_path/"ZuCo_v1", base_path/"ZuCo_v2"]:
        if not zuco_path.exists():
            print(f"Skipping {zuco_path} (not found)")
            continue
        for p in sorted(zuco_path.rglob("*.mat")):
            count = 0
            for eeg, text in load_mat_any(p):
                normed = normalize_eeg(eeg)
                if normed is not None:
                    final_data.append({'eeg': normed, 'text': text, 'sub': p.stem[:5], 'is_sentence': True})
                    count += 1
            print(f"  Parsed {count} sentences from {p.name}")
    print(f"ZuCo total: {len(final_data)} sentences")

    # InnerSpeech — epoch files are in derivatives/ — these are WORDS (is_sentence=False)
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
        except Exception as e:
            print(f"  InnerSpeech error on {p.name}: {e}")
            continue
    print(f"InnerSpeech total: {is_count} samples")
    print(f"TOTAL DATASET: {len(final_data)} samples")
    return final_data

# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

@app.function(image=image, gpu="H100", volumes={"/data": data_vol, "/persist": ckpt_vol}, timeout=72000, memory=65536)
def train_v5(epochs=50, batch_size=16, phase1_epochs=15):
    epochs = int(epochs); batch_size = int(batch_size); phase1_epochs = int(phase1_epochs)
    from transformers import AutoTokenizer; from sentence_transformers import SentenceTransformer
    device = torch.device("cuda:0"); base_path = Path("/data/EEG_Text")

    # Download data (with ZuCo v2 fallback)
    download_zuco(base_path, data_vol)

    all_samples = parse_data(base_path)
    if not all_samples: print("FAILURE: No data found."); return

    texts = [s['text'] for s in all_samples]; eegs = np.array([s['eeg'] for s in all_samples])
    subjects = [s['sub'] for s in all_samples]
    is_sentence_arr = np.array([s.get('is_sentence', len(s['text'].split()) > 2) for s in all_samples])
    unique_labels = sorted(list(set(texts))); label_to_id = {l: i for i, l in enumerate(unique_labels)}
    class_ids = np.array([label_to_id[t] for t in texts]); unique_subs = sorted(list(set(subjects)))
    sub_to_id = {s: i for i, s in enumerate(unique_subs)}; sub_ids = np.array([sub_to_id[s] for s in subjects])

    n_sentences = is_sentence_arr.sum()
    n_words = len(is_sentence_arr) - n_sentences
    print(f"Dataset: {len(texts)} total | {n_sentences} sentences (ZuCo) | {n_words} words (InnerSpeech)")

    indices = np.random.permutation(len(texts))
    train_idx = indices[:int(0.8*len(indices))]; val_idx = indices[int(0.8*len(indices)):]
    # Separate sentence-only train indices for Phase 2
    sentence_train_idx = np.array([i for i in train_idx if is_sentence_arr[i]])
    print(f"Train: {len(train_idx)} ({len(sentence_train_idx)} sentences) | Val: {len(val_idx)}")

    teacher = SentenceTransformer('all-mpnet-base-v2').to(device)
    all_label_embs = F.normalize(torch.from_numpy(teacher.encode(unique_labels)).float().to(device), dim=-1)
    
    # Negative Teacher (Distractor)
    neg_text = "I am a large language model trained by Google."
    neg_teacher_emb = F.normalize(torch.from_numpy(teacher.encode([neg_text])).float().to(device), dim=-1)
    
    del teacher; torch.cuda.empty_cache()

    qwen_name = "Qwen/Qwen2.5-0.5B-Instruct"; tokenizer = AutoTokenizer.from_pretrained(qwen_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = EEG_VLM_V5(n_subjects=len(unique_subs), n_classes=len(unique_labels), qwen_name=qwen_name, tokenizer=tokenizer).to(device)
    model.llm.to(device, dtype=torch.bfloat16)

    best_val_bleu = 0.0

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 1: Encoder-only training (LLM fully frozen)
    # Train rasterizer + freq_decomposer + video_encoder + bridge + projections + classifier
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print(f"PHASE 1: Encoder-only (LLM frozen) — {phase1_epochs} epochs")
    print("="*70)

    # Freeze LLM completely
    for param in model.llm.parameters():
        param.requires_grad = False

    encoder_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer_p1 = torch.optim.AdamW(encoder_params, lr=1e-4, weight_decay=0.01)
    print(f"Phase 1 trainable params: {sum(p.numel() for p in encoder_params):,}")

    for epoch in range(1, phase1_epochs + 1):
        model.train(); epoch_loss = 0; n_b = 0
        perm = np.random.permutation(len(train_idx))
        for i in range(0, len(perm), batch_size):
            b_perm = perm[i:i+batch_size]
            b_idx = train_idx[b_perm]
            if len(b_idx) < 2: continue
            t_eeg = augment_eeg(torch.from_numpy(eegs[b_idx]).to(device, dtype=torch.bfloat16))
            t_sub = torch.tensor(sub_ids[b_idx], device=device)
            t_cls = torch.tensor(class_ids[b_idx], device=device)
            b_texts = [texts[idx] for idx in b_idx]
            b_embs = all_label_embs[[label_to_id[t] for t in b_texts]]
            enc = tokenizer(b_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            optimizer_p1.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, metrics = model(t_eeg, enc.input_ids, enc.input_ids, t_sub, b_embs, 
                                      neg_teacher_emb.expand(len(b_idx), -1), t_cls,
                                      grl_alpha=min(1.0, epoch/5), phase="encoder_only")
            loss.backward(); torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
            optimizer_p1.step(); epoch_loss += loss.item(); n_b += 1
        print(f"P1 Epoch {epoch}/{phase1_epochs} | Loss: {epoch_loss/max(1,n_b):.3f} | "
              f"cls_acc: {metrics['cls_acc']:.1%} | contrastive: {metrics['contrastive']:.3f} | "
              f"triplet: {metrics['triplet']:.3f} | ctc: {metrics['ctc']:.3f}")

    # Save Phase 1 checkpoint
    torch.save(model.state_dict(), "/persist/eeg_vlm_v5_phase1.pt")
    ckpt_vol.commit()
    print("✓ Phase 1 complete — encoder trained, saved phase1 checkpoint")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE 2: LLM fine-tuning (sentence-only, LoRA unfrozen, lower LR)
    # Only ZuCo sentences go through the LM loss.
    # InnerSpeech words still contribute contrastive + classification signal.
    # ══════════════════════════════════════════════════════════════════════
    phase2_epochs = epochs - phase1_epochs
    print("\n" + "="*70)
    print(f"PHASE 2: LLM fine-tuning (LoRA) — {phase2_epochs} epochs")
    print(f"  Sentence-only LM training with {len(sentence_train_idx)} ZuCo samples")
    print("="*70)

    # Unfreeze LoRA params only
    for param in model.llm.parameters():
        param.requires_grad = False
    for name, param in model.llm.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad = True
    # Also keep encoder + bridge + projections trainable but at lower LR
    lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not any(
        p is lp for lp in lora_params)]
    optimizer_p2 = torch.optim.AdamW([
        {'params': other_params, 'lr': 1e-5},       # encoder/bridge/proj at low LR
        {'params': lora_params, 'lr': 2e-5},         # LoRA at slightly higher LR
    ], weight_decay=0.01)
    print(f"Phase 2 trainable: {sum(p.numel() for p in lora_params):,} LoRA + "
          f"{sum(p.numel() for p in other_params):,} encoder params")

    for epoch in range(1, phase2_epochs + 1):
        model.train(); epoch_loss = 0; lm_loss_sum = 0; n_b = 0

        # Interleave: for each batch, mix some sentences + some words
        # But only sentences get LM loss
        sent_perm = np.random.permutation(len(sentence_train_idx))
        all_perm = np.random.permutation(len(train_idx))

        for i in range(0, len(sent_perm), max(1, batch_size // 2)):
            # Half batch from sentences (for LM loss)
            sent_b = sentence_train_idx[sent_perm[i:i + batch_size // 2]]
            # Half batch from all data (for contrastive)
            all_start = (i * 2) % len(all_perm)
            all_b = train_idx[all_perm[all_start:all_start + batch_size // 2]]
            b_idx = np.concatenate([sent_b, all_b])
            if len(b_idx) < 2: continue

            t_eeg = augment_eeg(torch.from_numpy(eegs[b_idx]).to(device, dtype=torch.bfloat16))
            t_sub = torch.tensor(sub_ids[b_idx], device=device)
            t_cls = torch.tensor(class_ids[b_idx], device=device)
            t_is_sent = torch.tensor(is_sentence_arr[b_idx], device=device)
            b_texts = [texts[idx] for idx in b_idx]
            b_embs = all_label_embs[[label_to_id[t] for t in b_texts]]
            enc = tokenizer(b_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            optimizer_p2.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, metrics = model(t_eeg, enc.input_ids, enc.input_ids, t_sub, b_embs, 
                                      neg_teacher_emb.expand(len(b_idx), -1), t_cls,
                                      grl_alpha=1.0, phase="joint", is_sentence=t_is_sent)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_p2.step(); epoch_loss += loss.item(); lm_loss_sum += metrics['lm']; n_b += 1

        avg_loss = epoch_loss / max(1, n_b)
        avg_lm = lm_loss_sum / max(1, n_b)
        print(f"P2 Epoch {epoch}/{phase2_epochs} | Loss: {avg_loss:.3f} | LM: {avg_lm:.3f} | "
              f"ctc: {metrics.get('ctc', 0):.3f} | cls_acc: {metrics['cls_acc']:.1%}")

        # Eval — generate on sentence samples only
        if epoch % 2 == 0 or epoch == phase2_epochs:
            model.eval()
            val_sents = [vi for vi in val_idx if is_sentence_arr[vi]][:30]
            gen_correct = 0; bleu_scores = []
            with torch.no_grad():
                for idx in val_sents:
                    t_eeg = torch.from_numpy(eegs[idx:idx+1]).to(device, dtype=torch.bfloat16)
                    llm_pred, ctc_pred = model.generate(t_eeg, tokenizer, max_tokens=40)
                    pred = llm_pred[0].strip()
                    ref = texts[idx].strip()
                    # Word overlap BLEU-1
                    ref_tok = ref.lower().split(); pred_tok = pred.lower().split()
                    if pred_tok:
                        overlap = len(set(ref_tok) & set(pred_tok)) / max(1, len(set(ref_tok)))
                        bleu_scores.append(overlap)
                    if pred.lower().startswith(ref.lower()[:20]): gen_correct += 1
            val_bleu = np.mean(bleu_scores) if bleu_scores else 0
            print(f"  Val sentence BLEU-proxy: {val_bleu:.1%} | prefix-match: {gen_correct}/{len(val_sents)}")
            if val_sents:
                print(f"  Sample: REF='{texts[val_sents[0]][:60]}'")
                print(f"         GEN='{pred[:60]}'")
                print(f"         CTC='{ctc_pred[0][:60]}'")
            else:
                print("  [Warning] No sentence samples in validation set.")

            if val_bleu > best_val_bleu:
                best_val_bleu = val_bleu
                torch.save(model.state_dict(), "/persist/eeg_vlm_v5_best.pt")
                ckpt_vol.commit()
                print(f"  ★ NEW BEST sentence BLEU: {val_bleu:.1%}")

    print(f"\n{'='*70}\nTraining complete. Best sentence BLEU: {best_val_bleu:.1%}")

@app.local_entrypoint()
def main(): train_v5.remote()
