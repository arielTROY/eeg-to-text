
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

app = modal.App("eeg-vlm-v11-ctc")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v61", create_if_missing=True)
v20_vol  = modal.Volume.from_name("bt-checkpoints-v20", create_if_missing=False)
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
        "editdistance", "sentencepiece", "openneuro-py",
    ])
    .run_function(_download_models)
)

# ═══════════════════════════════════════════════════════════════════════════════
# V11: CTC-first branch.
#
# Goal:
#   Break the current vowel-skeleton ceiling by improving Phase 1 directly,
#   before asking a denoiser to hallucinate missing lexical information.
#
# What this branch changes:
#   [1] Keeps the proven V20 warm-start and gamma/motion preprocessing.
#   [2] Expands the EEG video into 8 channels by adding spatial-Laplacian
#       views of raw and gamma to emphasize local consonant-like transients.
#   [3] Uses a 2-view shared VideoMAE encoder: full scalp + center zoom.
#   [4] Trains with char CTC + articulatory class CTC + boundary loss +
#       text-contrastive loss.
#   [5] Evaluates diverse top-3 beams, not only a single best path.
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Vocabulary ───────────────────────────────────────────────────────────────

CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz0123456789.,!?'\" ()"
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR  = {i: c for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE  = len(CHAR_VOCAB)

VOWELS     = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
STOPS      = set("bcdgkptq")
FRICATIVES = set("fhjsvxz")
SONORANTS  = set("lmnrwy")

CLASS_BLANK = 0
CLASS_VOWEL = 1
CLASS_STOP = 2
CLASS_FRICATIVE = 3
CLASS_SONORANT = 4
CLASS_DIGIT = 5
CLASS_OTHER = 6
CLASS_SPACE = 7
CLASS_VOCAB_SIZE = 8

NUM_EEG_BANDS = 8
PHASE1_MIX_EPOCHS = 6
HASH_DIM = 512
BOUNDARY_STEPS = 256
BOUNDARY_SIGMA = 2.2

def text_to_char_ids(text):
    return [CHAR_TO_ID[c] for c in text.lower() if c in CHAR_TO_ID]


def char_to_class_id(ch):
    if ch == " ":
        return CLASS_SPACE
    if ch in VOWELS:
        return CLASS_VOWEL
    if ch in STOPS:
        return CLASS_STOP
    if ch in FRICATIVES:
        return CLASS_FRICATIVE
    if ch in SONORANTS:
        return CLASS_SONORANT
    if ch in CONSONANTS:
        return CLASS_OTHER
    if ch.isdigit():
        return CLASS_DIGIT
    return CLASS_OTHER


def text_to_class_ids(text):
    class_ids = []
    prev = None
    for c in text.lower():
        if c not in CHAR_TO_ID:
            continue
        cls = char_to_class_id(c)
        if cls != prev:
            class_ids.append(cls)
            prev = cls
    return class_ids


def normalize_text(text):
    text = text.lower()
    return "".join(c if c in CHAR_TO_ID and c != "_" else " " for c in text)


def _stable_hash_ngram(ngram):
    h = 2166136261
    for b in ngram.encode("utf-8"):
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return h % HASH_DIM


def text_to_hash_features(text):
    norm = f" {normalize_text(text)} "
    vec = np.zeros(HASH_DIM, dtype=np.float32)
    if len(norm) < 3:
        vec[0] = 1.0
        return vec
    for i in range(len(norm) - 2):
        ng = norm[i:i + 3]
        vec[_stable_hash_ngram(ng)] += 1.0
    norm_val = np.linalg.norm(vec)
    if norm_val > 0:
        vec /= norm_val
    return vec


def text_to_boundary_targets(text, steps=BOUNDARY_STEPS):
    chars = [c for c in text.lower() if c in CHAR_TO_ID and c != " "]
    target = np.zeros(steps, dtype=np.float32)
    if not chars:
        return target
    centers = np.linspace(0, steps - 1, len(chars) + 2, dtype=np.float32)[1:-1]
    grid = np.arange(steps, dtype=np.float32)
    for center, ch in zip(centers, chars):
        amp = 1.0 if ch in CONSONANTS else (0.75 if ch in VOWELS else 0.55)
        target = np.maximum(target, amp * np.exp(-0.5 * ((grid - center) / BOUNDARY_SIGMA) ** 2))
    return target

def greedy_ctc_decode(logits):
    """logits: (T, B, Vocab) → list[str]"""
    ids = logits.argmax(dim=-1)
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

def beam_ctc_decode(logits, beam_width=5):
    """logits: (T, B, Vocab) → list[str] using beam search."""
    return [beams[0][0] if beams else "" for beams in beam_ctc_decode_nbest(
        logits, beam_width=beam_width, nbest=1)]


def beam_ctc_decode_nbest(logits, beam_width=10, nbest=3):
    """logits: (T, B, Vocab) → list[list[(text, norm_score)]]."""
    log_probs = F.log_softmax(logits, dim=-1)
    T, B, V = log_probs.shape
    results = []
    for b in range(B):
        beams = [([], -1, 0.0)]
        for t in range(T):
            new_beams = {}
            frame_lp = log_probs[t, b]
            topk_vals, topk_ids = frame_lp.topk(min(beam_width * 2, V))
            for seq, last_tok, score in beams:
                for val, tok_id in zip(topk_vals.tolist(), topk_ids.tolist()):
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
            beams = sorted(new_beams.values(), key=lambda x: -x[2])[:max(beam_width, nbest)]
        beams = beams[:nbest]
        if not beams:
            results.append([("", 1.0)])
            continue
        scores = torch.tensor([beam[2] for beam in beams], dtype=torch.float32)
        norm_scores = torch.softmax(scores - scores.max(), dim=0).tolist()
        decoded = []
        for (seq, _, _), score in zip(beams, norm_scores):
            text = ''.join(ID_TO_CHAR.get(i, '') for i in seq)
            decoded.append((text, float(score)))
        results.append(decoded)
    return results


def compute_cer(pred, ref):
    import editdistance
    if not ref: return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, ref) / len(ref), 2.0)

def compute_wer(pred, ref):
    import editdistance
    pred_w = pred.strip().split()
    ref_w  = ref.strip().split()
    if not ref_w: return 0.0 if not pred_w else 1.0
    return min(editdistance.eval(pred_w, ref_w) / len(ref_w), 2.0)


# ─── EEG Augmentation ─────────────────────────────────────────────────────────

def augment_eeg(eeg_np, strength='moderate'):
    eeg = eeg_np.copy()
    T, C = eeg.shape
    if strength == 'moderate':
        jitter, amp_lo, amp_hi, noise_std, ch_drop_p, mask_spans = (64, 0.85, 1.15, 0.05, 0.10, 2)
    else:
        jitter, amp_lo, amp_hi, noise_std, ch_drop_p, mask_spans = (16, 0.95, 1.05, 0.02, 0.05, 1)

    shift = np.random.randint(-jitter, jitter + 1)
    if shift != 0:
        eeg = np.roll(eeg, shift, axis=0)
        if shift > 0: eeg[:shift] = 0
        else: eeg[shift:] = 0

    scales = np.random.uniform(amp_lo, amp_hi, size=(1, C)).astype(np.float32)
    eeg = eeg * scales
    eeg = eeg + np.random.randn(*eeg.shape).astype(np.float32) * noise_std

    n_drop = int(C * ch_drop_p)
    if n_drop > 0:
        drop_idx = np.random.choice(C, n_drop, replace=False)
        eeg[:, drop_idx] = 0

    for _ in range(mask_spans):
        span_len = np.random.randint(16, 33)
        start = np.random.randint(0, max(1, T - span_len))
        eeg[start:start+span_len] = 0

    return eeg


# ─── Memory Bank ──────────────────────────────────────────────────────────────

class MemoryBank:
    def __init__(self, size=512, dim=896):
        self.size = size
        self.bank = torch.zeros(size, dim)
        self.ptr  = 0
        self.n    = 0

    def enqueue(self, emb):
        emb = emb.detach().cpu().float()
        B = emb.shape[0]
        idx = torch.arange(self.ptr, self.ptr + B) % self.size
        self.bank[idx] = emb
        self.ptr = (self.ptr + B) % self.size
        self.n = min(self.n + B, self.size)

    def get(self, device):
        if self.n < 2: return None
        return self.bank[:self.n].to(device)


# ─── Modules ──────────────────────────────────────────────────────────────────

class MultiScaleRasterizer(nn.Module):
    def __init__(self, size=64, n_electrodes=64, n_bands=NUM_EEG_BANDS):
        super().__init__()
        self.n_bands = n_bands
        self.n_electrodes = n_electrodes
        angles = torch.linspace(0, 2 * np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
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
        x_split = x.reshape(B * T, self.n_bands, self.n_electrodes)
        proj = x_split @ self.W.T.to(device=x.device, dtype=x.dtype)
        return proj.reshape(B, T, self.n_bands, 64, 64)


class ChannelAdapter(nn.Module):
    def __init__(self, in_channels=NUM_EEG_BANDS):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        nn.init.kaiming_uniform_(self.conv.weight, a=1)
        nn.init.zeros_(self.conv.bias)

    def forward(self, imgs):
        B, T, C, H, W = imgs.shape
        out = self.conv(imgs.view(B * T, C, H, W))
        return out.view(B, T, 3, H, W)


class CrossAttentionBridge(nn.Module):
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


# ─── Model ────────────────────────────────────────────────────────────────────

class EEG_CTC_V11(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel

        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=1024, tubelet_size=4, hidden_size=768)
        self.rasterizer = MultiScaleRasterizer(n_bands=NUM_EEG_BANDS)
        self.ch_adapt = ChannelAdapter(NUM_EEG_BANDS)
        self.video_enc = VideoMAEModel(v_cfg)
        self.view_fuse = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.LayerNorm(768),
            nn.GELU(),
        )
        self.ctc_head = nn.Linear(768, VOCAB_SIZE)
        self.class_ctc_head = nn.Linear(768, CLASS_VOCAB_SIZE)
        self.boundary_head = nn.Linear(768, 1)
        self.contrast_proj = nn.Sequential(
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, HASH_DIM),
        )
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        self.class_ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        self.boundary_loss_fn = nn.BCEWithLogitsLoss()

    def _encode_view(self, imgs):
        B, T, _, _, _ = imgs.shape
        imgs = self.ch_adapt(imgs)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        return v_out.reshape(B, 256, 16, 768).mean(dim=2)

    def _center_zoom(self, imgs):
        B, T, C, _, _ = imgs.shape
        crop = imgs[:, :, :, 8:56, 8:56]
        crop = F.interpolate(
            crop.reshape(B * T, C, 48, 48),
            size=(64, 64),
            mode="bilinear",
            align_corners=False,
        )
        return crop.reshape(B, T, C, 64, 64)

    def encode(self, traj):
        imgs = self.rasterizer(traj)
        full_seq = self._encode_view(imgs)
        zoom_seq = self._encode_view(self._center_zoom(imgs))
        return self.view_fuse(torch.cat([full_seq, zoom_seq], dim=-1))

    def forward(self, traj, char_ids, char_lens, class_ids, class_lens, text_hash, boundary_targets):
        device = traj.device
        B = traj.shape[0]
        v_seq = self.encode(traj)

        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)
        input_lens = torch.full((B,), 256, dtype=torch.long, device=device)
        loss_ctc = self.ctc_loss_fn(
            F.log_softmax(ctc_logits, dim=-1), char_ids, input_lens, char_lens)

        class_logits = self.class_ctc_head(v_seq).transpose(0, 1)
        loss_class_ctc = self.class_ctc_loss_fn(
            F.log_softmax(class_logits, dim=-1), class_ids, input_lens, class_lens)

        boundary_logits = self.boundary_head(v_seq).squeeze(-1)
        loss_boundary = self.boundary_loss_fn(boundary_logits, boundary_targets)

        probs = F.softmax(ctc_logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        ent = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        loss_div = -ent.mean()

        eeg_text = F.normalize(self.contrast_proj(v_seq.mean(dim=1)), dim=-1)
        target_text = F.normalize(text_hash, dim=-1)
        sims = eeg_text @ target_text.T / 0.07
        loss_contrast = F.cross_entropy(sims, torch.arange(B, device=device))

        return {
            "ctc_logits": ctc_logits,
            "loss_ctc": loss_ctc,
            "loss_class_ctc": loss_class_ctc,
            "loss_boundary": loss_boundary,
            "loss_contrast": loss_contrast,
            "loss_div": loss_div,
        }

    @torch.no_grad()
    def decode_ctc(self, traj, beam_width=10, nbest=3):
        traj = traj.to(next(self.parameters()).dtype)
        v_seq = self.encode(traj)
        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)
        return beam_ctc_decode_nbest(ctc_logits, beam_width=beam_width, nbest=nbest)


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
            dst_sd[k] = v.clone(); n_copied += 1
        else:
            n_skipped += 1
    video_enc.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Transferred {n_copied} tensors, skipped {n_skipped} "
          f"(patch emb + pos emb — re-init from scratch)")
    del src; torch.cuda.empty_cache()


def load_v20_phase1_warmstart(model, ckpt_path="/v20/v20_phase1_best.pt"):
    """Load the proven V20 stack, expanding the old 4→3 adapter into 8→3."""
    if not Path(ckpt_path).exists():
        print(f"V20 Phase 1 checkpoint not found at {ckpt_path}")
        return False

    print(f"Loading V20 Phase 1 warm-start weights from {ckpt_path}...")
    src_sd = torch.load(ckpt_path, map_location="cpu")
    dst_sd = model.state_dict()
    prefixes = ("rasterizer.", "video_enc.", "ch_adapt.", "ctc_head.")
    n_copied = n_skipped = 0
    for k, v in src_sd.items():
        if not k.startswith(prefixes):
            continue
        if k == "ch_adapt.conv.weight" and k in dst_sd:
            dst = dst_sd[k]
            if v.shape[0] == dst.shape[0] and v.shape[1] == 4 and dst.shape[1] == 8:
                expanded = torch.zeros_like(dst)
                expanded[:, :4] = v
                expanded[:, 4] = v[:, 0]
                expanded[:, 5] = v[:, 3]
                expanded[:, 6] = v[:, 0]
                expanded[:, 7] = v[:, 3]
                dst_sd[k] = expanded
                n_copied += 1
                continue
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v.clone()
            n_copied += 1
        else:
            n_skipped += 1
    model.load_state_dict(dst_sd, strict=False)
    print(f"  ✓ Loaded {n_copied} V20 Phase 1 tensors, skipped {n_skipped}")
    return n_copied > 0


# ─── Data loading ─────────────────────────────────────────────────────────────

def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000: return False
    try:
        with open(path, 'rb') as f: return b'MATLAB' in f.read(128)
    except: return False

def download_zuco(base_path, vol):
    zuco_v1 = base_path / "ZuCo_v1"
    zuco_v2 = base_path / "ZuCo_v2"
    inner_sp = base_path / "InnerSpeech"
    for d in [zuco_v1, zuco_v2, inner_sp]:
        d.mkdir(parents=True, exist_ok=True)

    def fetch(project_id, target_dir, names):
        res   = __import__('subprocess').run(
            ["osf", "-p", project_id, "list"], capture_output=True, text=True)
        paths = res.stdout.splitlines()
        for name in names:
            local = target_dir / name
            if not verify_mat_file(local):
                remote = next((p for p in paths if p.strip().endswith(name)), None)
                if remote:
                    print(f"  Fetching {name}...")
                    __import__('subprocess').run(
                        ["osf", "-p", project_id, "fetch", remote.strip(), str(local)],
                        timeout=900)

    fetch("q3zws", zuco_v1, ["resultsZAB_SR.mat","resultsZDM_SR.mat",
                               "resultsZAB_NR.mat","resultsZDM_NR.mat"])
    fetch("2urht", zuco_v2, ["resultsYAC_NR.mat","resultsYAG_NR.mat","resultsYAK_NR.mat"])

    if not any(inner_sp.rglob("*-epo.fif")):
        import openneuro
        print("  Fetching InnerSpeech ds003626 derivatives/sub-01 ...")
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
        except: pass

def normalize_eeg(eeg, target_ch=64, target_t=1024):
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 2: return None
    ch, t = eeg.shape
    if ch > t: eeg = eeg.T; ch, t = eeg.shape
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
    delta = np.diff(eeg, axis=1, prepend=eeg[:, :1])
    gamma_delta = np.diff(gamma, axis=1, prepend=gamma[:, :1])
    lap_raw = eeg - np.roll(eeg, 1, axis=0)
    lap_gamma = gamma - np.roll(gamma, 1, axis=0)
    lap_raw[0] = 0
    lap_gamma[0] = 0

    def rezscore(x):
        mu  = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True).clip(min=1e-6)
        return (x - mu) / std

    delta = rezscore(delta)
    gamma_delta = rezscore(gamma_delta)
    lap_raw = rezscore(lap_raw)
    lap_gamma = rezscore(lap_gamma)

    combined = np.concatenate(
        [eeg.T, alpha.T, beta.T, gamma.T, delta.T, gamma_delta.T, lap_raw.T, lap_gamma.T], axis=1
    )  # (1024, 512)
    return combined.astype(np.float32)


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, augment='none',
                 include_zuco=True, include_innerspeech=False):
        self.samples = []
        self.augment = augment
        p = Path(base_path)
        skipped = 0
        zuco_count = 0
        inner_count = 0

        if include_zuco:
            for zp in [p/"ZuCo_v1", p/"ZuCo_v2"]:
                for mat in zp.rglob("*.mat"):
                    for raw, text in load_mat_any(mat):
                        if not text or "placeholder" in text.lower():
                            skipped += 1
                            continue
                        normed = normalize_eeg(raw)
                        if normed is not None:
                            self.samples.append({'eeg': normed, 'text': text})
                            zuco_count += 1

        if include_innerspeech:
            import mne
            for fif in (p/"InnerSpeech").rglob("*-epo.fif"):
                try:
                    epochs = mne.read_epochs(str(fif), preload=True, verbose=False)
                    epochs.resample(128)
                    data = epochs.get_data()
                    labels = (
                        epochs.metadata['condition'].tolist()
                        if epochs.metadata is not None and 'condition' in epochs.metadata.columns
                        else [f"word_{i}" for i in range(len(data))]
                    )
                    for d, label in zip(data, labels):
                        normed = normalize_eeg(d)
                        if normed is not None:
                            self.samples.append({'eeg': normed, 'text': str(label)})
                            inner_count += 1
                except:
                    continue

        print(
            f"[Dataset] total={len(self.samples)} zuco={zuco_count} inner={inner_count} "
            f"skipped={skipped} augment={augment}"
        )

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        eeg = s['eeg']
        if self.augment != 'none':
            eeg = augment_eeg(eeg, strength=self.augment)
        char_ids = text_to_char_ids(s['text'])
        class_ids = text_to_class_ids(s['text'])
        text_hash = text_to_hash_features(s['text'])
        boundary = text_to_boundary_targets(s['text'])
        return eeg, s['text'], char_ids, class_ids, text_hash, boundary


def collate_fn(batch):
    eegs, texts, char_ids_list, class_ids_list, hash_list, boundary_list = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs))
    char_lens = torch.tensor([len(c) for c in char_ids_list], dtype=torch.long)
    max_chars = max(char_lens) if max(char_lens) > 0 else 1
    char_pad  = torch.zeros(len(char_ids_list), max_chars, dtype=torch.long)
    for i, c in enumerate(char_ids_list):
        if len(c) > 0:
            char_pad[i, :len(c)] = torch.tensor(c)
    class_lens = torch.tensor([len(c) for c in class_ids_list], dtype=torch.long)
    max_classes = max(class_lens) if max(class_lens) > 0 else 1
    class_pad = torch.zeros(len(class_ids_list), max_classes, dtype=torch.long)
    for i, c in enumerate(class_ids_list):
        if len(c) > 0:
            class_pad[i, :len(c)] = torch.tensor(c)
    text_hash = torch.from_numpy(np.stack(hash_list))
    boundary = torch.from_numpy(np.stack(boundary_list))
    return eegs, list(texts), char_pad, char_lens, class_pad, class_lens, text_hash, boundary


# ─── Training loop ────────────────────────────────────────────────────────────

ACCUM_STEPS = 12

def make_phase1_optimizer(model, steps_per_epoch, epochs):
    groups = []

    def add_group(params, lr, name):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append({"name": name, "params": params, "lr": lr, "weight_decay": 0.01})

    add_group(model.ch_adapt.parameters(), 4e-4, "adapter")
    add_group(model.ctc_head.parameters(), 2e-4, "ctc")
    add_group(model.class_ctc_head.parameters(), 2e-4, "class_ctc")
    add_group(model.boundary_head.parameters(), 2e-4, "boundary")
    add_group(model.contrast_proj.parameters(), 2e-4, "contrast")
    add_group(model.view_fuse.parameters(), 1.5e-4, "view_fuse")
    add_group(model.video_enc.parameters(), 5e-5, "encoder")

    seen = {id(p) for g in groups for p in g["params"]}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
    add_group(other, 5e-5, "other")

    printable = ", ".join(
        f"{g['name']}:{sum(p.numel() for p in g['params'])/1e6:.1f}M@{g['lr']:.0e}"
        for g in groups
    )
    optimizer = torch.optim.AdamW([
        {"params": g["params"], "lr": g["lr"], "weight_decay": g["weight_decay"]}
        for g in groups
    ])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in groups],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,
    )
    return optimizer, scheduler, printable

def _beam_diversity(texts):
    if len(texts) < 2:
        return 0.0
    scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            scores.append(compute_cer(texts[i], texts[j]))
    return float(np.mean(scores)) if scores else 0.0


def _validate_phase1(model, val_loader, epoch, ckpt_prefix, best_val_cer):
    device = next(model.parameters()).device
    model.eval()
    cer_scores, wer_scores = [], []
    oracle_cer_scores, oracle_wer_scores, diversity_scores = [], [], []
    vowel_correct = vowel_total = cons_correct = cons_total = 0
    print(f"── Val Epoch {epoch} ──")
    with torch.no_grad():
        shown = 0
        for batch in val_loader:
            eeg_v, texts_v, _, _, _, _, _, _ = batch
            beam_groups = model.decode_ctc(eeg_v.to(device).to(torch.bfloat16), beam_width=10, nbest=3)
            for beams, ref in zip(beam_groups, texts_v):
                ref_lower = ref.lower()
                beam_texts = [(b[0] or "").strip() for b in beams]
                top = beam_texts[0]
                oracle = min(beam_texts, key=lambda x: (compute_wer(x, ref_lower), compute_cer(x, ref_lower)))
                cer_scores.append(compute_cer(top, ref_lower))
                wer_scores.append(compute_wer(top, ref_lower))
                oracle_cer_scores.append(compute_cer(oracle, ref_lower))
                oracle_wer_scores.append(compute_wer(oracle, ref_lower))
                diversity_scores.append(_beam_diversity(beam_texts))
                pred_chars = set(top)
                ref_chars = set(ref_lower)
                for c in ref_chars:
                    if c in VOWELS:
                        vowel_total += 1
                        if c in pred_chars:
                            vowel_correct += 1
                    elif c in CONSONANTS:
                        cons_total += 1
                        if c in pred_chars:
                            cons_correct += 1
                if shown < 4:
                    print(f"  REF: '{ref[:100]}'")
                    print(f"  TOP1: '{beam_texts[0][:100]}'  CER={cer_scores[-1]:.2f} WER={wer_scores[-1]:.2f}")
                    if len(beam_texts) > 1:
                        print(f"  TOP2: '{beam_texts[1][:100]}'")
                    if len(beam_texts) > 2:
                        print(f"  TOP3: '{beam_texts[2][:100]}'")
                    shown += 1
    metrics = {
        "cer": float(np.mean(cer_scores)),
        "wer": float(np.mean(wer_scores)),
        "oracle_cer": float(np.mean(oracle_cer_scores)),
        "oracle_wer": float(np.mean(oracle_wer_scores)),
        "beam_diversity": float(np.mean(diversity_scores)),
        "vowel_recall": float(vowel_correct / max(vowel_total, 1)),
        "cons_recall": float(cons_correct / max(cons_total, 1)),
    }
    print(f"  Mean CER: {metrics['cer']:.3f}  Mean WER: {metrics['wer']:.3f}")
    print(f"  Oracle CER: {metrics['oracle_cer']:.3f}  Oracle WER: {metrics['oracle_wer']:.3f}")
    print(f"  Beam diversity: {metrics['beam_diversity']:.3f}")
    print(f"  Vowel recall: {metrics['vowel_recall']:.2f} ({vowel_correct}/{vowel_total})  "
          f"Consonant recall: {metrics['cons_recall']:.2f} ({cons_correct}/{cons_total})")
    torch.save(metrics, f"{ckpt_prefix}_phase1_ep{epoch}_metrics.pt")
    if metrics["cer"] < best_val_cer:
        best_val_cer = metrics["cer"]
        torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_best.pt")
        torch.save(metrics, f"{ckpt_prefix}_phase1_best_metrics.pt")
        print(f"  ✓ New best val CER {best_val_cer:.3f}")
    ckpt_vol.commit()
    return best_val_cer


def _train_phase1(model, epochs, train_loader, val_loader, ckpt_prefix, epoch_offset=0,
                  best_train_ctc=float("inf"), best_val_cer=float("inf")):
    device = next(model.parameters()).device
    print("── Phase 1: multi-view CTC + boundary + contrastive supervision ──")
    for p in model.parameters():
        p.requires_grad = True
    optimizer, scheduler, phase1_groups = make_phase1_optimizer(model, len(train_loader), epochs)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"  Phase 1 trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    print(f"  Phase 1 groups: {phase1_groups}")

    for local_epoch in range(1, epochs + 1):
        epoch = epoch_offset + local_epoch
        model.train()
        tot_ctc = tot_class = tot_boundary = tot_contrast = tot_div = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            eeg, texts, c_ids, c_lens, class_ids, class_lens, text_hash, boundary_targets = batch
            eeg = eeg.to(device).to(torch.bfloat16)
            c_ids = c_ids.to(device)
            c_lens = c_lens.to(device)
            class_ids = class_ids.to(device)
            class_lens = class_lens.to(device)
            text_hash = text_hash.to(device)
            boundary_targets = boundary_targets.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = model(eeg, c_ids, c_lens, class_ids, class_lens, text_hash, boundary_targets)
                loss = (
                    10.0 * out["loss_ctc"] +
                    3.0 * out["loss_class_ctc"] +
                    2.0 * out["loss_boundary"] +
                    1.5 * out["loss_contrast"] +
                    0.5 * out["loss_div"]
                ) / ACCUM_STEPS

            loss.backward()
            tot_ctc += out["loss_ctc"].item()
            tot_class += out["loss_class_ctc"].item()
            tot_boundary += out["loss_boundary"].item()
            tot_contrast += out["loss_contrast"].item()
            tot_div += out["loss_div"].item()

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if len(train_loader) % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        n = len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        avg_ctc = tot_ctc / max(n, 1)
        print(f"Epoch {epoch:3d} | CTC:{avg_ctc:.3f} ACTC:{tot_class/n:.3f} "
              f"BND:{tot_boundary/n:.3f} CON:{tot_contrast/n:.3f} Div:{tot_div/n:.4f} | LR0:{lr:.2e}")

        if avg_ctc < best_train_ctc:
            best_train_ctc = avg_ctc
            print(f"  ✓ New best train CTC {best_train_ctc:.3f}")

        if epoch % 3 == 0:
            torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_ep{epoch}.pt")
            best_val_cer = _validate_phase1(model, val_loader, epoch, ckpt_prefix, best_val_cer)

    return best_train_ctc, best_val_cer


def run_phase1_curriculum(model, epochs_p1, train_loader_mix, train_loader_pure,
                          val_loader, ckpt_prefix, start_epoch=0):
    best_train_ctc = float("inf")
    best_val_cer = float("inf")
    mix_epochs = min(PHASE1_MIX_EPOCHS, epochs_p1)
    current_epoch = start_epoch

    if current_epoch < mix_epochs and train_loader_mix is not None:
        stage_epochs = mix_epochs - current_epoch
        print(f"[Phase1] Stage A: mixed lexical alignment for {stage_epochs} epochs")
        best_train_ctc, best_val_cer = _train_phase1(
            model, epochs=stage_epochs, train_loader=train_loader_mix, val_loader=val_loader,
            ckpt_prefix=ckpt_prefix, epoch_offset=current_epoch,
            best_train_ctc=best_train_ctc, best_val_cer=best_val_cer)
        current_epoch = mix_epochs

    if current_epoch < epochs_p1:
        stage_epochs = epochs_p1 - current_epoch
        print(f"[Phase1] Stage B: pure ZuCo consolidation for {stage_epochs} epochs")
        best_train_ctc, best_val_cer = _train_phase1(
            model, epochs=stage_epochs, train_loader=train_loader_pure, val_loader=val_loader,
            ckpt_prefix=ckpt_prefix, epoch_offset=current_epoch,
            best_train_ctc=best_train_ctc, best_val_cer=best_val_cer)

    return best_train_ctc, best_val_cer


# ─── Modal functions ──────────────────────────────────────────────────────────

@app.function(image=image, gpu="H100", timeout=86400,
              volumes={"/data": data_vol, "/persist": ckpt_vol, "/v20": v20_vol},
              retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=10.0))
def run_pipeline(epochs_p1: int = 24):
    download_zuco(Path("/data/EEG_Text"), data_vol)

    ds_zuco = EEGDataset(
        "/data/EEG_Text", augment='none',
        include_zuco=True, include_innerspeech=False)
    n_train  = int(0.9 * len(ds_zuco))
    indices  = list(range(len(ds_zuco)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    train_zuco = torch.utils.data.Subset(ds_zuco, train_idx)
    val_ds     = torch.utils.data.Subset(ds_zuco, val_idx)

    ds_inner = EEGDataset(
        "/data/EEG_Text", augment='none',
        include_zuco=False, include_innerspeech=True)
    train_loader_mix = None
    if len(ds_inner) > 0:
        inner_cap = min(len(ds_inner), len(train_idx))
        inner_idx = np.random.permutation(len(ds_inner))[:inner_cap].tolist()
        train_inner = torch.utils.data.Subset(ds_inner, inner_idx)
        train_mix = torch.utils.data.ConcatDataset([train_zuco, train_inner])
        print(f"[Phase1] Using balanced InnerSpeech subset: {inner_cap} train words")
    else:
        train_mix = train_zuco
        print("[Phase1] InnerSpeech not found; training on ZuCo only")

    mkloader = lambda d, shuf: torch.utils.data.DataLoader(
        d, batch_size=2, shuffle=shuf, collate_fn=collate_fn)
    train_loader_pure = mkloader(train_zuco, True)
    val_loader        = mkloader(val_ds, False)
    if len(ds_inner) > 0:
        train_loader_mix = mkloader(train_mix, True)

    model = EEG_CTC_V11().to(torch.bfloat16).cuda()

    # ── Resume logic ──────────────────────────────────────────────────────────
    import glob as _glob
    p1_ckpts  = sorted(_glob.glob("/persist/v11_phase1_ep*.pt"))
    best_p1   = "/persist/v11_phase1_best.pt"
    final     = "/persist/v11_final.pt"
    p1_start_epoch = 0

    if Path(final).exists():
        print("✓ V11 already complete.")
        return

    if Path(best_p1).exists() and p1_ckpts:
        last_p1       = p1_ckpts[-1]
        last_p1_epoch = int(last_p1.split("_ep")[-1].replace(".pt", ""))
        model.load_state_dict(torch.load(last_p1, map_location="cuda"), strict=False)
        print(f"✓ Resuming Phase 1 from epoch {last_p1_epoch}")
        p1_start_epoch = last_p1_epoch
    else:
        if not load_v20_phase1_warmstart(model):
            load_pretrained_videomae_encoder(model.video_enc)

    print(f"\n{'='*60}")
    print(f" V11: multi-view CTC + boundary + contrastive Phase 1")
    print(f" {'(RESUMED)' if p1_ckpts else 'FRESH START'}")
    print(f"{'='*60}\n")

    run_phase1_curriculum(
        model, epochs_p1=epochs_p1,
        train_loader_mix=train_loader_mix,
        train_loader_pure=train_loader_pure,
        val_loader=val_loader,
        ckpt_prefix="/persist/v11",
        start_epoch=p1_start_epoch,
    )

    best_v11 = Path("/persist/v11_phase1_best.pt")
    if best_v11.exists():
        model.load_state_dict(torch.load(best_v11, map_location="cuda"), strict=False)
    torch.save(model.state_dict(), final)
    ckpt_vol.commit()
    print("\n✓ V11 Phase 1 complete. Saved /persist/v11_final.pt")


@app.local_entrypoint()
def main(mode: str = "pipeline", epochs_p1: int = 24):
    if mode == "pipeline":
        print(f"Launching V11: Phase1={epochs_p1}ep")
        run_pipeline.remote(epochs_p1=epochs_p1)
    else:
        print(f"Unknown mode '{mode}'. Use: pipeline")
