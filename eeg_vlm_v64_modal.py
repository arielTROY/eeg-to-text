import glob
from pathlib import Path

import modal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

app = modal.App("eeg-vlm-v64")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v64", create_if_missing=True)
v61_vol = modal.Volume.from_name("bt-checkpoints-v61", create_if_missing=False)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)


def _download_models():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, VideoMAEModel

    VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")
    AutoTokenizer.from_pretrained("google/byt5-small")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "numpy",
        "scipy",
        "jiwer",
        "einops",
        "h5py",
        "mne",
        "pandas",
        "accelerate",
        "osfclient",
        "editdistance",
        "sentencepiece",
        "openneuro-py",
    ])
    .run_function(_download_models)
)

# V6.4: multi-beam real-hint byte-level denoiser with constrained reranking.

CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz0123456789.,!?\'\" ()"
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE = len(CHAR_VOCAB)

NUM_EEG_BANDS = 6
CLASS_VOCAB_SIZE = 6
VOWELS = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")


def greedy_ctc_decode(logits):
    ids = logits.argmax(dim=-1)
    results = []
    for b in range(ids.shape[1]):
        seq = ids[:, b].tolist()
        out = []
        prev = -1
        for i in seq:
            if i != prev and i != 0:
                out.append(ID_TO_CHAR.get(i, ""))
            prev = i
        results.append("".join(out))
    return results


def beam_ctc_candidates(logits, beam_width=5, topk=3):
    log_probs = F.log_softmax(logits, dim=-1)
    t_steps, batch, vocab = log_probs.shape
    results = []
    for b in range(batch):
        beams = [([], -1, 0.0)]
        for t in range(t_steps):
            new_beams = {}
            frame_lp = log_probs[t, b]
            top_vals, top_ids = frame_lp.topk(min(beam_width * 2, vocab))
            for seq, last_tok, score in beams:
                for val, tok_id in zip(top_vals.tolist(), top_ids.tolist()):
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
            beams = sorted(new_beams.values(), key=lambda x: -x[2])[:beam_width]
        top_beams = []
        for seq, _, score in beams[:topk]:
            text = "".join(ID_TO_CHAR.get(i, "") for i in seq).strip() or "..."
            top_beams.append((text, float(score)))
        while len(top_beams) < topk:
            top_beams.append(top_beams[-1] if top_beams else ("...", 0.0))
        results.append(top_beams)
    return results


def beam_ctc_decode(logits, beam_width=5):
    return [beams[0][0] for beams in beam_ctc_candidates(logits, beam_width=beam_width, topk=1)]


def compute_cer(pred, ref):
    import editdistance

    if not ref:
        return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, ref) / len(ref), 2.0)


def compute_wer(pred, ref):
    import editdistance

    pred_w = pred.strip().split()
    ref_w = ref.strip().split()
    if not ref_w:
        return 0.0 if not pred_w else 1.0
    return min(editdistance.eval(pred_w, ref_w) / len(ref_w), 2.0)


def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with open(path, "rb") as f:
            return b"MATLAB" in f.read(128)
    except Exception:
        return False


def download_zuco(base_path, vol):
    import subprocess

    zuco_v1 = base_path / "ZuCo_v1"
    zuco_v2 = base_path / "ZuCo_v2"
    inner_sp = base_path / "InnerSpeech"
    for d in [zuco_v1, zuco_v2, inner_sp]:
        d.mkdir(parents=True, exist_ok=True)

    def fetch(project_id, target_dir, names):
        missing = [name for name in names if not verify_mat_file(target_dir / name)]
        if not missing:
            print(f"  Using cached {target_dir.name} files")
            return

        res = subprocess.run(["osf", "-p", project_id, "list"], capture_output=True, text=True, timeout=120)
        paths = res.stdout.splitlines()
        for name in names:
            local = target_dir / name
            if verify_mat_file(local):
                continue
            remote = next((p for p in paths if p.strip().endswith(name)), None)
            if remote:
                print(f"  Fetching {name}...")
                subprocess.run(["osf", "-p", project_id, "fetch", remote.strip(), str(local)], timeout=900)

    fetch("q3zws", zuco_v1, ["resultsZAB_SR.mat", "resultsZDM_SR.mat", "resultsZAB_NR.mat", "resultsZDM_NR.mat"])
    fetch("2urht", zuco_v2, ["resultsYAC_NR.mat", "resultsYAG_NR.mat", "resultsYAK_NR.mat"])

    if not any(inner_sp.rglob("*-epo.fif")):
        import openneuro

        print("  Fetching InnerSpeech ds003626 derivatives/sub-01 ...")
        openneuro.download(dataset="ds003626", target_dir=str(inner_sp), include=["derivatives/sub-01/"])
    vol.commit()


def load_mat_any(path):
    import h5py
    import scipy.io as sio

    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        sentences = data["sentenceData"]
        if not isinstance(sentences, (np.ndarray, list)):
            sentences = [sentences]
        for s in sentences:
            if hasattr(s, "rawData") and hasattr(s, "content") and isinstance(s.rawData, np.ndarray):
                yield s.rawData, str(s.content)
    except Exception:
        try:
            with h5py.File(path, "r") as f:
                if "sentenceData" in f:
                    for key in f["sentenceData"].keys():
                        s = f["sentenceData"][key]
                        if "rawData" in s and "content" in s:
                            yield np.array(s["rawData"]).T, "hdf5_content_placeholder"
        except Exception:
            return


def normalize_eeg(eeg, target_ch=64, target_t=1024):
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 2:
        return None
    ch, t = eeg.shape
    if ch > t:
        eeg = eeg.T
        ch, t = eeg.shape
    if ch > target_ch:
        eeg = eeg[:target_ch, :]
    elif ch < target_ch:
        eeg = np.pad(eeg, ((0, target_ch - ch), (0, 0)))
    if t > target_t:
        eeg = eeg[:, :target_t]
    elif t < target_t:
        eeg = np.pad(eeg, ((0, 0), (0, target_t - t)))

    mu = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True).clip(min=1e-6)
    eeg = (eeg - mu) / std

    from scipy.signal import butter, filtfilt

    def band_env(lo, hi, data):
        try:
            b, a = butter(4, [lo / 64.0, hi / 64.0], btype="bandpass")
            env = np.abs(filtfilt(b, a, data, axis=1))
            return np.convolve(env.flatten(), np.ones(5) / 5, mode="same").reshape(env.shape)
        except Exception:
            return np.zeros_like(data)

    alpha = band_env(8, 13, eeg)
    beta = band_env(13, 30, eeg)
    gamma = band_env(30, 100, eeg)
    delta = np.diff(eeg, axis=1, prepend=eeg[:, :1])
    gamma_delta = np.diff(gamma, axis=1, prepend=gamma[:, :1])

    def rezscore(x):
        mu = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True).clip(min=1e-6)
        return (x - mu) / std

    delta = rezscore(delta)
    gamma_delta = rezscore(gamma_delta)
    combined = np.concatenate([eeg.T, alpha.T, beta.T, gamma.T, delta.T, gamma_delta.T], axis=1)
    return combined.astype(np.float32)


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, base_path):
        self.samples = []
        p = Path(base_path)
        skipped = 0
        for zp in [p / "ZuCo_v1", p / "ZuCo_v2"]:
            for mat in zp.rglob("*.mat"):
                for raw, text in load_mat_any(mat):
                    if not text or "placeholder" in text.lower():
                        skipped += 1
                        continue
                    normed = normalize_eeg(raw)
                    if normed is not None:
                        self.samples.append({"eeg": normed, "text": text})
        print(f"[Dataset] total={len(self.samples)} skipped={skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample["eeg"], sample["text"]


def collate_eeg_batch(batch):
    eegs, texts = zip(*batch)
    return torch.from_numpy(np.stack(eegs)), list(texts)


class MultiScaleRasterizer(nn.Module):
    def __init__(self, size=64, n_electrodes=64, n_bands=NUM_EEG_BANDS):
        super().__init__()
        self.n_bands = n_bands
        self.n_electrodes = n_electrodes
        angles = torch.linspace(0, 2 * np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
        px = gx.flatten().unsqueeze(1)
        py = gy.flatten().unsqueeze(1)
        ex = pos_2d[:, 0].unsqueeze(0)
        ey = pos_2d[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex) ** 2 + (py - ey) ** 2)
        w = 1.0 / (dist + 1e-4) ** 2.0
        r = torch.sqrt(px**2 + py**2)
        w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)

    def forward(self, x):
        bsz, t_steps, _ = x.shape
        x_split = x.reshape(bsz * t_steps, self.n_bands, self.n_electrodes)
        proj = x_split @ self.W.T.to(device=x.device, dtype=x.dtype)
        return proj.reshape(bsz, t_steps, self.n_bands, 64, 64)


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels=NUM_EEG_BANDS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        bsz, t_steps, channels, height, width = imgs.shape
        out = self.conv(imgs.view(bsz * t_steps, channels, height, width))
        return out.view(bsz, t_steps, 3, height, width)


class CrossAttentionBridge(nn.Module):
    def __init__(self, in_ch=768, out_ch=1024, n_latents=128):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, in_ch))
        self.mha = nn.MultiheadAttention(in_ch, num_heads=16, batch_first=True)
        self.ln_k = nn.LayerNorm(in_ch)
        self.ln_q = nn.LayerNorm(in_ch)
        self.proj = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        bsz = x.shape[0]
        q = self.latents.unsqueeze(0).expand(bsz, -1, -1)
        x, _ = self.mha(self.ln_q(q), self.ln_k(x), x)
        return self.proj(x)


class EEGCTCEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel

        v_cfg = VideoMAEConfig(
            num_channels=3,
            image_size=64,
            patch_size=16,
            num_frames=1024,
            tubelet_size=4,
            hidden_size=768,
        )
        self.rasterizer = MultiScaleRasterizer(n_bands=NUM_EEG_BANDS)
        self.ch_adapt = ChannelAdapter(NUM_EEG_BANDS)
        self.video_enc = VideoMAEModel(v_cfg)
        self.bridge = CrossAttentionBridge(768, 1024, n_latents=128)
        self.ctc_head = nn.Linear(768, VOCAB_SIZE)
        self.class_ctc_head = nn.Linear(768, CLASS_VOCAB_SIZE)

    def encode(self, traj):
        imgs = self.rasterizer(traj)
        imgs = self.ch_adapt(imgs)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(dim=2)

    @torch.no_grad()
    def decode_ctc(self, traj, use_beam=True, return_beams=False, topk=3):
        traj = traj.to(next(self.parameters()).dtype)
        v_seq = self.encode(traj)
        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)
        if return_beams:
            return beam_ctc_candidates(ctc_logits, beam_width=max(5, topk * 2), topk=topk)
        return beam_ctc_decode(ctc_logits, beam_width=5) if use_beam else greedy_ctc_decode(ctc_logits)


class ByT5Denoiser(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, max_new_tokens=96, num_beams=6, num_return_sequences=4):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=4,
            repetition_penalty=1.20,
            length_penalty=0.55,
            early_stopping=True,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )


def load_v61_phase1_encoder(model, ckpt_path="/v61/v61_phase1_best.pt"):
    if not Path(ckpt_path).exists():
        print(f"V6.1 Phase 1 checkpoint not found at {ckpt_path}")
        return False
    print(f"Loading V6.1 Phase 1 encoder from {ckpt_path}...")
    src_sd = torch.load(ckpt_path, map_location="cpu")
    dst_sd = model.state_dict()
    prefixes = ("rasterizer.", "video_enc.", "ch_adapt.", "ctc_head.", "bridge.", "class_ctc_head.")
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
    print(f"  Loaded {n_copied} encoder tensors, skipped {n_skipped}")
    return n_copied > 0


def ctc_like_corrupt_text(text, severity=1.0):
    out = []
    for ch in text.lower():
        if ch not in CHAR_TO_ID:
            continue
        r = np.random.random()
        if ch in VOWELS:
            if r < 0.08 * severity:
                continue
            out.append(ch)
            if np.random.random() < 0.04 * severity:
                out.append(ch)
        elif ch in CONSONANTS:
            if r < 0.45 * severity:
                continue
            if r < 0.60 * severity:
                out.append(np.random.choice(list(CONSONANTS)))
            else:
                out.append(ch)
        elif ch == " ":
            out.append(" " if r > 0.20 * severity else "  ")
        elif ch in ".!?":
            if r > 0.50 * severity:
                out.append(".")
        else:
            if r > 0.35 * severity:
                out.append(ch)
    hint = " ".join("".join(out).split())
    return hint if hint else "..."


def format_denoiser_source(beams, scores=None):
    parts = []
    for idx, beam in enumerate(beams[:3], start=1):
        clean = (beam or "...").strip() or "..."
        if scores is not None and idx - 1 < len(scores):
            delta = scores[idx - 1] - scores[0]
            parts.append(f"<b{idx}> {clean} </b{idx}> <s{idx}> {delta:.2f} </s{idx}>")
        else:
            parts.append(f"<b{idx}> {clean} </b{idx}>")
    return " ".join(parts)


class DenoiserPairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, repeats=4, synth_prob=0.30):
        self.pairs = [p for p in pairs if p["text"].strip()]
        self.repeats = repeats
        self.synth_prob = synth_prob

    def __len__(self):
        return len(self.pairs) * self.repeats

    def __getitem__(self, idx):
        pair = self.pairs[idx % len(self.pairs)]
        beams = list(pair["beams"])
        scores = list(pair.get("scores", [0.0, -0.5, -1.0]))
        if np.random.random() < self.synth_prob:
            beams = [
                ctc_like_corrupt_text(pair["text"], severity=np.random.uniform(0.75, 1.05)),
                ctc_like_corrupt_text(pair["text"], severity=np.random.uniform(0.95, 1.25)),
                ctc_like_corrupt_text(pair["text"], severity=np.random.uniform(1.15, 1.45)),
            ]
            scores = [0.0, -0.4, -0.8]
        return {
            "source": format_denoiser_source(beams, scores),
            "target": pair["text"],
            "hint": pair["hint"],
            "beams": beams,
        }


def collate_denoiser(batch, tokenizer, max_source_len=192, max_target_len=192):
    sources = [item["source"] for item in batch]
    targets = [item["target"] for item in batch]
    hints = [item["hint"] for item in batch]
    beams = [item["beams"] for item in batch]
    enc = tokenizer(sources, padding=True, truncation=True, max_length=max_source_len, return_tensors="pt")
    target_enc = tokenizer(text_target=targets, padding=True, truncation=True, max_length=max_target_len, return_tensors="pt")
    labels = target_enc.input_ids
    labels[labels == tokenizer.pad_token_id] = -100
    return enc.input_ids, enc.attention_mask, labels, targets, hints, beams


def precompute_real_pairs(ctc_model, subset, split_name):
    device = next(ctc_model.parameters()).device
    loader = torch.utils.data.DataLoader(subset, batch_size=4, shuffle=False, collate_fn=collate_eeg_batch)
    pairs = []
    for step, (eegs, texts) in enumerate(loader, start=1):
        beam_groups = ctc_model.decode_ctc(
            eegs.to(device).to(torch.bfloat16),
            use_beam=True,
            return_beams=True,
            topk=3,
        )
        for beams, text in zip(beam_groups, texts):
            beam_texts = [(b[0].strip() or "...") for b in beams]
            beam_scores = [float(b[1]) for b in beams]
            pairs.append({
                "hint": beam_texts[0],
                "beams": beam_texts,
                "scores": beam_scores,
                "text": text,
            })
        if step % 50 == 0:
            print(f"[Pairs:{split_name}] processed {step}/{len(loader)} batches")
    return pairs


def load_or_build_pairs(ctc_model, train_subset, val_subset):
    train_cache = Path("/persist/v64_train_pairs.pt")
    val_cache = Path("/persist/v64_val_pairs.pt")
    if train_cache.exists() and val_cache.exists():
        print("Loaded cached real CTC pairs")
        return torch.load(train_cache), torch.load(val_cache)

    print("Building real CTC hint pairs from V6.1 Phase 1 checkpoint...")
    train_pairs = precompute_real_pairs(ctc_model, train_subset, "train")
    val_pairs = precompute_real_pairs(ctc_model, val_subset, "val")
    torch.save(train_pairs, train_cache)
    torch.save(val_pairs, val_cache)
    ckpt_vol.commit()
    print(f"Saved {len(train_pairs)} train pairs and {len(val_pairs)} val pairs")
    return train_pairs, val_pairs


def clean_generated(text):
    return " ".join(text.strip().split())


def repeated_ngram_penalty(text, n=3):
    toks = text.split()
    if len(toks) < n:
        return 0.0
    seen = set()
    repeats = 0
    for i in range(len(toks) - n + 1):
        gram = tuple(toks[i:i + n])
        if gram in seen:
            repeats += 1
        seen.add(gram)
    return repeats / max(len(toks) - n + 1, 1)


def rerank_generated_candidates(candidates, candidate_scores, beam_group):
    beam_group = [(b or "...").lower() for b in beam_group if (b or "").strip()]
    primary = beam_group[0] if beam_group else "..."
    best_text = clean_generated(candidates[0].lower()) if candidates else primary
    best_score = -1e9
    for cand, seq_score in zip(candidates, candidate_scores):
        pred = clean_generated(cand.lower())
        if not pred:
            continue
        hint_distance = min(compute_cer(pred, beam) for beam in beam_group) if beam_group else 1.0
        length_pen = abs(len(pred) - len(primary)) / max(len(primary), 1)
        repeat_pen = repeated_ngram_penalty(pred, n=3)
        score = float(seq_score) - 1.10 * hint_distance - 0.35 * length_pen - 0.60 * repeat_pen
        if score > best_score:
            best_score = score
            best_text = pred
    return best_text


def evaluate_denoiser(model, tokenizer, val_pairs, epoch):
    device = next(model.parameters()).device
    val_loader = torch.utils.data.DataLoader(
        val_pairs,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_denoiser(
            [{
                "source": format_denoiser_source(p["beams"], p.get("scores")),
                "target": p["text"],
                "hint": p["hint"],
                "beams": p["beams"],
            } for p in b],
            tokenizer,
        ),
    )
    raw_cer, raw_wer, gen_cer, gen_wer = [], [], [], []
    print(f"Val Epoch {epoch}")
    shown = 0
    for input_ids, attn_mask, labels, targets, hints, beam_groups in val_loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        max_new_tokens = min(128, max(28, int(max(len(h) for h in hints) * 1.4 + 8)))
        gen = model.generate(input_ids, attn_mask, max_new_tokens=max_new_tokens, num_beams=6, num_return_sequences=4)
        preds = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)
        seq_scores = gen.sequences_scores.tolist() if gen.sequences_scores is not None else [0.0] * len(preds)
        group_size = 4
        for idx, (hint, ref, beams) in enumerate(zip(hints, targets, beam_groups)):
            ref_lower = ref.lower()
            cand_texts = preds[idx * group_size:(idx + 1) * group_size]
            cand_scores = seq_scores[idx * group_size:(idx + 1) * group_size]
            pred_clean = rerank_generated_candidates(cand_texts, cand_scores, beams)
            raw_cer.append(compute_cer(hint.lower(), ref_lower))
            raw_wer.append(compute_wer(hint.lower(), ref_lower))
            gen_cer.append(compute_cer(pred_clean, ref_lower))
            gen_wer.append(compute_wer(pred_clean, ref_lower))
            if shown < 4:
                print(f"  REF: '{ref[:100]}'")
                print(f"  CTC: '{hint[:100]}'  CER={raw_cer[-1]:.2f} WER={raw_wer[-1]:.2f}")
                print(f"  GEN: '{pred_clean[:100]}'  CER={gen_cer[-1]:.2f} WER={gen_wer[-1]:.2f}")
                shown += 1
    metrics = {
        "raw_cer": float(np.mean(raw_cer)),
        "raw_wer": float(np.mean(raw_wer)),
        "gen_cer": float(np.mean(gen_cer)),
        "gen_wer": float(np.mean(gen_wer)),
    }
    print(f"  Raw CTC Mean CER: {metrics['raw_cer']:.3f}  Mean WER: {metrics['raw_wer']:.3f}")
    print(f"  GEN Mean CER: {metrics['gen_cer']:.3f}  Mean WER: {metrics['gen_wer']:.3f}")
    return metrics


def train_denoiser(model, tokenizer, train_pairs, val_pairs, ckpt_prefix, epochs, start_epoch=0):
    device = next(model.parameters()).device
    train_ds = DenoiserPairDataset(train_pairs, repeats=6, synth_prob=0.30)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda b: collate_denoiser(b, tokenizer),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_metrics_path = Path(f"{ckpt_prefix}_best_metrics.pt")
    best_gen_wer = torch.load(best_metrics_path)["gen_wer"] if best_metrics_path.exists() else float("inf")

    for local_epoch in range(1, epochs + 1):
        epoch = start_epoch + local_epoch
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            input_ids, attn_mask, labels, _, _, _ = batch
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            labels = labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids, attn_mask, labels)
                loss = out.loss / 4
            loss.backward()
            total_loss += out.loss.item()
            if step % 4 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if len(train_loader) % 4 != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch:3d} | LM:{avg_loss:.3f} | LR0:{optimizer.param_groups[0]['lr']:.2e}")

        metrics = evaluate_denoiser(model, tokenizer, val_pairs, epoch)
        torch.save(model.state_dict(), f"{ckpt_prefix}_ep{epoch}.pt")
        torch.save(metrics, f"{ckpt_prefix}_ep{epoch}_metrics.pt")
        ckpt_vol.commit()

        if metrics["gen_wer"] < best_gen_wer:
            best_gen_wer = metrics["gen_wer"]
            torch.save(model.state_dict(), f"{ckpt_prefix}_best.pt")
            torch.save(metrics, best_metrics_path)
            ckpt_vol.commit()
            print(f"  New best GEN WER {best_gen_wer:.3f}")


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={"/data": data_vol, "/persist": ckpt_vol, "/v61": v61_vol},
    retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=10.0),
)
def run_pipeline(epochs: int = 12):
    from transformers import AutoTokenizer

    model_name = "google/byt5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    download_zuco(Path("/data/EEG_Text"), data_vol)

    ds_zuco = EEGDataset("/data/EEG_Text")
    n_train = int(0.9 * len(ds_zuco))
    indices = list(range(len(ds_zuco)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_subset = torch.utils.data.Subset(ds_zuco, indices[:n_train])
    val_subset = torch.utils.data.Subset(ds_zuco, indices[n_train:])

    ctc_model = EEGCTCEncoder().to(torch.bfloat16).cuda()
    if not load_v61_phase1_encoder(ctc_model):
        raise RuntimeError("V6.1 Phase 1 checkpoint is required for V6.4.")
    ctc_model.eval()
    train_pairs, val_pairs = load_or_build_pairs(ctc_model, train_subset, val_subset)
    del ctc_model
    torch.cuda.empty_cache()

    denoiser = ByT5Denoiser(model_name).to(torch.bfloat16).cuda()
    ckpt_prefix = "/persist/v64_denoiser"
    final_path = Path("/persist/v64_final.pt")
    epoch_ckpts = sorted(glob.glob("/persist/v64_denoiser_ep*.pt"))
    start_epoch = 0

    if final_path.exists():
        print("V6.4 already complete.")
        return

    if epoch_ckpts:
        latest = epoch_ckpts[-1]
        start_epoch = int(latest.split("_ep")[-1].replace(".pt", ""))
        denoiser.load_state_dict(torch.load(latest, map_location="cuda"))
        print(f"Resuming V6.4 denoiser from epoch {start_epoch}")

    print("\n" + "=" * 60)
    print(" V6.4: multi-beam ByT5 denoiser on top of V6.1 Phase 1")
    print(f" {'(RESUMED)' if start_epoch > 0 else 'FRESH START'}")
    print("=" * 60 + "\n")

    remaining_epochs = max(epochs - start_epoch, 0)
    if remaining_epochs > 0:
        train_denoiser(denoiser, tokenizer, train_pairs, val_pairs, ckpt_prefix, remaining_epochs, start_epoch)

    best_path = Path("/persist/v64_denoiser_best.pt")
    if best_path.exists():
        denoiser.load_state_dict(torch.load(best_path, map_location="cuda"))
    torch.save(denoiser.state_dict(), final_path)
    ckpt_vol.commit()
    print("\nV6.4 complete. Saved /persist/v64_final.pt")


@app.local_entrypoint()
def main(mode: str = "pipeline", epochs: int = 12):
    if mode == "pipeline":
        print(f"Launching V6.4: denoiser_epochs={epochs}")
        run_pipeline.remote(epochs=epochs)
    else:
        print(f"Unknown mode '{mode}'. Use: pipeline")
