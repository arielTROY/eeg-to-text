"""
eval_v1400.py — Evaluation for V1400 Stage 2 (Q-Former + T5-base)
==================================================================
Metrics:
  - Content Recall (CR): word overlap / reference words  [PRIMARY]
  - WER, CER (edit distance)
  - BLEU-1/2, ROUGE-1/L
  - Diversity (unique preds / total)
  - Semantic cosine similarity (EEG embedding vs ground-truth SBERT)
  - Contrastive R@1/R@5/R@10 retrieval (EEG → sentence pool)

Usage:
  python eval_v1400.py --ckpt ./ckpts/v1400/s2_best.pt
  python eval_v1400.py --ckpt ./ckpts/v1400/s2_best.pt --n-samples 200
  python eval_v1400.py --ckpt ./ckpts/v1400/s2_ep30_*.pt --compare-v1200
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Import architecture from eeg_vlm_v1400
from eeg_vlm_v1400 import (
    EEGV1400_S2, load_or_build_cache, build_labels,
    EEGTextDataset, collate_fn,
    compute_wer, compute_cer, cons_recall,
    DATA_DIR, CKPT_DIR,
)

# ── Metrics ───────────────────────────────────────────────────────────────────

def bleu_n(pred, ref, n):
    """BLEU-n for a single pair."""
    from collections import Counter
    p_tokens = pred.lower().split()
    r_tokens = ref.lower().split()
    if len(p_tokens) < n or len(r_tokens) < n:
        return 0.0
    p_ngrams = Counter(tuple(p_tokens[i:i+n]) for i in range(len(p_tokens) - n + 1))
    r_ngrams = Counter(tuple(r_tokens[i:i+n]) for i in range(len(r_tokens) - n + 1))
    matches  = sum((p_ngrams & r_ngrams).values())
    total    = sum(p_ngrams.values())
    return matches / total if total > 0 else 0.0


def rouge_1_f(pred, ref):
    p = set(pred.lower().split())
    r = set(ref.lower().split())
    if not p or not r:
        return 0.0
    inter = len(p & r)
    prec  = inter / len(p)
    rec   = inter / len(r)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def rouge_l_f(pred, ref):
    """LCS-based ROUGE-L."""
    p = pred.lower().split()
    r = ref.lower().split()
    if not p or not r:
        return 0.0
    m, n = len(r), len(p)
    # DP LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if r[i-1] == p[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs  = dp[m][n]
    prec = lcs / n if n > 0 else 0.0
    rec  = lcs / m if m > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def retrieval_at_k(eeg_embs, text_embs, k_vals=(1, 5, 10)):
    """
    Top-K retrieval: for each EEG embedding, find the closest text embedding.
    Returns hit rates R@k for each k in k_vals.
    eeg_embs:  (N, D)
    text_embs: (N, D)
    """
    eeg_embs  = F.normalize(torch.tensor(eeg_embs, dtype=torch.float32), dim=-1)
    text_embs = F.normalize(torch.tensor(text_embs, dtype=torch.float32), dim=-1)
    sims = eeg_embs @ text_embs.T   # (N, N)
    results = {}
    for k in k_vals:
        topk  = sims.topk(k, dim=1).indices   # (N, k)
        gt    = torch.arange(len(eeg_embs)).unsqueeze(1)  # (N, 1)
        hits  = (topk == gt).any(dim=1).float().mean().item()
        results[f"R@{k}"] = hits
    return results


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)

    cache = load_or_build_cache(data_dir)
    eegs         = cache["eegs"]
    texts        = cache["texts"]
    subject_ids  = cache["subject_ids"]
    sentence_ids = cache["sentence_ids"]
    N = len(texts)

    sbert_cache = ckpt_dir / "sbert_embs.pt"
    if sbert_cache.exists():
        sbert_np = torch.load(sbert_cache, weights_only=False)
    else:
        print("Computing SBERT embeddings...")
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        sbert_np    = sbert_model.encode(texts, batch_size=64, show_progress_bar=True,
                                         convert_to_numpy=True)
        torch.save(sbert_np, sbert_cache)

    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    input_ids_list, attn_mask_list = [], []
    for text in texts:
        enc = tokenizer(text, max_length=80, truncation=True,
                        padding=False, return_tensors='pt')
        input_ids_list.append(enc.input_ids[0])
        attn_mask_list.append(enc.attention_mask[0])

    sentiments, lengths = build_labels(texts, tokenizer)

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(N)
    split = int(0.85 * N)
    val_idx = idx[split:].tolist()

    val_ds = EEGTextDataset(
        eegs, texts, sbert_np, input_ids_list, attn_mask_list,
        sentiments, lengths, subject_ids, sentence_ids, val_idx, augment=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    print(f"Val set: {len(val_ds)} samples")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt_path = args.ckpt
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stage = ckpt.get("stage", 2)
    print(f"Checkpoint: epoch={ckpt.get('epoch','?')} stage={stage} "
          f"div={ckpt.get('diversity','?'):.3f} cr={ckpt.get('cr','?'):.4f} "
          f"wer={ckpt.get('wer','?'):.3f}")

    model = EEGV1400_S2().to(device)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.freeze_videomae()
    model = model.to(torch.bfloat16)
    model.eval()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable/1e6:.1f}M  Frozen: {frozen/1e6:.1f}M")

    # ── SBERT model for semantic similarity ──────────────────────────────────
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # ── Run generation ────────────────────────────────────────────────────────
    all_preds, all_refs = [], []
    all_eeg_embs, all_text_embs = [], []
    n_gen = args.n_samples or len(val_ds)
    n_done = 0

    print(f"\nGenerating predictions (n={n_gen})...")
    for batch in val_loader:
        if n_done >= n_gen:
            break
        eegs_b, texts_b, sberts_b = batch[0], batch[1], batch[2]
        eegs_b = eegs_b.to(device)

        with torch.no_grad():
            preds = model.generate(eegs_b, tokenizer, max_new_tokens=80, diverse=True)
            # Get EEG embeddings for retrieval metric
            v_seq  = model.encode(eegs_b.to(
                next(p for p in model.parameters() if p.requires_grad).dtype))
            eeg_q  = model.qformer(v_seq)
            eeg_mean = eeg_q.mean(1)  # (B, 768)
            q      = F.normalize(model.sem_proj(eeg_mean), dim=-1).cpu().float()

        text_embs_b = F.normalize(
            model.txt_proj(sberts_b.to(q.device, q.dtype)), dim=-1).detach().cpu().float()

        for pred, ref, eq, tq in zip(preds, texts_b, q, text_embs_b):
            if n_done >= n_gen:
                break
            all_preds.append(pred)
            all_refs.append(ref)
            all_eeg_embs.append(eq.numpy())
            all_text_embs.append(tq.numpy())
            n_done += 1

    print(f"Generated {len(all_preds)} predictions.")

    # ── Compute metrics ───────────────────────────────────────────────────────
    wers    = [compute_wer(p.lower(), r.lower()) for p, r in zip(all_preds, all_refs)]
    cers    = [compute_cer(p.lower(), r.lower()) for p, r in zip(all_preds, all_refs)]
    bleu1   = [bleu_n(p, r, 1) for p, r in zip(all_preds, all_refs)]
    bleu2   = [bleu_n(p, r, 2) for p, r in zip(all_preds, all_refs)]
    rouge1  = [rouge_1_f(p, r) for p, r in zip(all_preds, all_refs)]
    rougel  = [rouge_l_f(p, r) for p, r in zip(all_preds, all_refs)]

    mean_wer   = float(np.mean(wers))
    mean_cer   = float(np.mean(cers))
    mean_bleu1 = float(np.mean(bleu1))
    mean_bleu2 = float(np.mean(bleu2))
    mean_r1    = float(np.mean(rouge1))
    mean_rl    = float(np.mean(rougel))
    diversity  = len(set(all_preds)) / max(1, len(all_preds))
    cr         = cons_recall(all_preds, all_refs)

    # Retrieval metrics
    retrieval = retrieval_at_k(
        np.stack(all_eeg_embs), np.stack(all_text_embs), k_vals=(1, 5, 10))

    # Semantic similarity via SBERT
    print("Computing semantic similarity...")
    pred_embs = sbert_model.encode(all_preds, batch_size=64, show_progress_bar=False,
                                   convert_to_numpy=True)
    ref_embs  = sbert_model.encode(all_refs,  batch_size=64, show_progress_bar=False,
                                   convert_to_numpy=True)
    pred_embs = pred_embs / (np.linalg.norm(pred_embs, axis=1, keepdims=True) + 1e-8)
    ref_embs  = ref_embs  / (np.linalg.norm(ref_embs,  axis=1, keepdims=True) + 1e-8)
    cos_sims  = (pred_embs * ref_embs).sum(axis=1)
    mean_cos  = float(np.mean(cos_sims))
    median_cos = float(np.median(cos_sims))

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  V1400 Evaluation Results  (n={len(all_preds)})")
    print("=" * 60)
    print(f"\n  Primary metric:")
    print(f"    Content Recall (CR)     : {cr:.4f}  [{cr*100:.1f}%]")
    print(f"    V1200 baseline CR       : 0.1120  [11.2%]")
    delta = (cr - 0.1120) / 0.1120 * 100
    print(f"    Δ vs V1200              : {delta:+.1f}%")
    print(f"\n  Text quality:")
    print(f"    WER                     : {mean_wer:.3f}  (V1200: 1.183)")
    print(f"    CER                     : {mean_cer:.3f}")
    print(f"    BLEU-1                  : {mean_bleu1:.4f}")
    print(f"    BLEU-2                  : {mean_bleu2:.4f}")
    print(f"    ROUGE-1 F               : {mean_r1:.4f}")
    print(f"    ROUGE-L F               : {mean_rl:.4f}")
    print(f"\n  Diversity (must stay 100%):")
    print(f"    Diversity               : {diversity:.3f}  [{diversity*100:.1f}%]")
    print(f"\n  Semantic similarity:")
    print(f"    Mean cosine sim         : {mean_cos:.4f}")
    print(f"    Median cosine sim       : {median_cos:.4f}")
    print(f"\n  Contrastive retrieval (EEG emb ↔ text pool):")
    for k, v in retrieval.items():
        print(f"    {k:<8}               : {v:.4f}  [{v*100:.1f}%]")

    # ── Sample outputs ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Sample Predictions")
    print("=" * 60)
    for i, (p, r) in enumerate(zip(all_preds[:10], all_refs[:10])):
        w = compute_wer(p.lower(), r.lower())
        cr_i = len(set(p.lower().split()) & set(r.lower().split())) / max(1, len(r.split()))
        cos_i = float(cos_sims[i])
        print(f"\n  [{i+1:2d}] CR={cr_i:.2f}  WER={w:.2f}  cos={cos_i:.3f}")
        print(f"    REF : {r}")
        print(f"    PRED: {p}")

    print("\n" + "=" * 60)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  V1400 CR={cr*100:.1f}%  WER={mean_wer:.3f}  Div={diversity*100:.0f}%  "
          f"cos={mean_cos:.3f}")
    print("=" * 60)

    return {
        "cr": cr, "wer": mean_wer, "cer": mean_cer,
        "bleu1": mean_bleu1, "bleu2": mean_bleu2,
        "rouge1": mean_r1, "rougeL": mean_rl,
        "diversity": diversity, "cos_sim": mean_cos,
        **retrieval,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate V1400 EEG-to-Text model")
    parser.add_argument("--ckpt",      type=str, required=True,
                        help="Path to Stage 2 checkpoint (.pt file)")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit generation to first N val samples (default: all)")
    parser.add_argument("--data-dir",  type=str, default=str(DATA_DIR))
    parser.add_argument("--ckpt-dir",  type=str, default=str(CKPT_DIR))
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
