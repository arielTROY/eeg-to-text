
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

app = modal.App("eeg-vlm-v12-ctc")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v12", create_if_missing=True)
v20_vol  = modal.Volume.from_name("bt-checkpoints-v20", create_if_missing=False)
v50_vol  = modal.Volume.from_name("bt-checkpoints-v50", create_if_missing=True)
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
# V12: Conservative CTC-first branch.
#
# What failed:
#   [1] V11 added too many new degrees of freedom at once and erased alignment.
#   [2] Validation collapsed to punctuation despite better train losses.
#
# What worked:
#   [1] V20 warm-start.
#   [2] V50 signal path: 6-band gamma-heavy preprocessing with motion channels.
#   [3] InnerSpeech lexical curriculum in early Phase 1.
#
# This branch keeps the working V50/V20 path intact and adds only one new
# auxiliary signal: articulatory-class CTC. It also freezes the encoder during
# Stage A so the transferred alignment is preserved before low-LR unfreezing.
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Vocabulary ───────────────────────────────────────────────────────────────

CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz0123456789.,!?'\" ()"
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR  = {i: c for i, c in enumerate(CHAR_VOCAB)}
VOCAB_SIZE  = len(CHAR_VOCAB)

VOWELS     = set("aeiou")
CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
STOPS      = set("bcdgkpt")
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

NUM_EEG_BANDS    = 6
PHASE1_MIX_EPOCHS = 6

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

class EEG_CTC_V12(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel

        v_cfg = VideoMAEConfig(
            num_channels=3, image_size=64, patch_size=16,
            num_frames=1024, tubelet_size=4, hidden_size=768)
        self.rasterizer = MultiScaleRasterizer(n_bands=NUM_EEG_BANDS)
        self.ch_adapt = ChannelAdapter(NUM_EEG_BANDS)
        self.video_enc = VideoMAEModel(v_cfg)
        self.ctc_head = nn.Linear(768, VOCAB_SIZE)
        self.class_ctc_head = nn.Linear(768, CLASS_VOCAB_SIZE)
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        self.class_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    def encode(self, traj):
        imgs = self.rasterizer(traj)
        imgs = self.ch_adapt(imgs)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        return v_out.reshape(traj.shape[0], 256, 16, 768).mean(dim=2)

    def forward(self, traj, char_ids, char_lens, class_ids, class_lens):
        B = traj.shape[0]
        device = traj.device

        v_seq = self.encode(traj)
        ctc_logits = self.ctc_head(v_seq).transpose(0, 1)
        class_logits = self.class_ctc_head(v_seq).transpose(0, 1)
        input_lens = torch.full((B,), 256, dtype=torch.long, device=device)

        loss_ctc = self.ctc_loss_fn(
            F.log_softmax(ctc_logits, dim=-1), char_ids, input_lens, char_lens)
        loss_class = self.class_loss_fn(
            F.log_softmax(class_logits, dim=-1), class_ids, input_lens, class_lens)

        probs = F.softmax(ctc_logits, dim=-1)
        mean_probs = probs.mean(dim=0)
        ent = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        loss_div = -ent.mean()

        return {
            "loss_ctc": loss_ctc,
            "loss_class": loss_class,
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


def load_v50_phase1_warmstart(model, ckpt_path="/v50/v50_phase1_best.pt"):
    if not Path(ckpt_path).exists():
        print(f"V50 Phase 1 checkpoint not found at {ckpt_path}")
        return False

    print(f"Loading V50 Phase 1 warm-start weights from {ckpt_path}...")
    src_sd = torch.load(ckpt_path, map_location="cpu")
    dst_sd = model.state_dict()
    prefixes = ("rasterizer.", "video_enc.", "ch_adapt.", "ctc_head.")
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
    print(f"  ✓ Loaded {n_copied} V50 Phase 1 tensors, skipped {n_skipped}")
    return n_copied > 0


def load_v20_phase1_warmstart(model, ckpt_path="/v20/v20_phase1_best.pt"):
    """Load the proven V20 stack, expanding the old 4→3 adapter into 6→3."""
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
            if v.shape[0] == dst.shape[0] and v.shape[1] == 4 and dst.shape[1] == 6:
                expanded = torch.zeros_like(dst)
                expanded[:, :4] = v
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

    def rezscore(x):
        mu  = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True).clip(min=1e-6)
        return (x - mu) / std

    delta = rezscore(delta)
    gamma_delta = rezscore(gamma_delta)

    combined = np.concatenate(
        [eeg.T, alpha.T, beta.T, gamma.T, delta.T, gamma_delta.T], axis=1
    )  # (1024, 384)
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
        return eeg, s['text'], char_ids, class_ids


def collate_fn(batch):
    eegs, texts, char_ids_list, class_ids_list = zip(*batch)
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
    return eegs, list(texts), char_pad, char_lens, class_pad, class_lens


# ─── Hint builders ────────────────────────────────────────────────────────────

def build_ctc_hint_batch(model, traj, tokenizer, device):
    with torch.no_grad():
        v_seq      = model.encode(traj)
        ctc_logits = model.ctc_head(v_seq).transpose(0, 1)
        ctc_texts  = beam_ctc_decode(ctc_logits, beam_width=5)
    prompts = []
    for hint in ctc_texts:
        messages = [
            {"role": "system", "content": "Recover the exact original sentence from noisy EEG-derived text. Be conservative: copy words from the hint when possible. Do not add unsupported facts. Return only the sentence."},
            {"role": "user", "content": f'Noisy EEG reading: "{hint}"'},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def build_teacher_hint_batch(texts, tokenizer, device, corruption_rate=0.3):
    prompts = []
    for text in texts:
        words     = text.split()
        corrupted = []
        for w in words:
            r = np.random.random()
            if r < corruption_rate:                pass
            elif r < corruption_rate * 1.3:        corrupted.append('...')
            else:                                  corrupted.append(w)
        hint = ' '.join(corrupted) if corrupted else '...'
        messages = [
            {"role": "system", "content": "Recover the exact original sentence from noisy EEG-derived text. Be conservative: copy words from the hint when possible. Do not add unsupported facts. Return only the sentence."},
            {"role": "user", "content": f'Noisy EEG reading: "{hint}"'},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    return enc.input_ids.to(device), enc.attention_mask.to(device)


# ─── Training loop ────────────────────────────────────────────────────────────

ACCUM_STEPS = 8


def set_phase1_trainable(model, freeze_encoder):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.ch_adapt.parameters():
        p.requires_grad = True
    for p in model.ctc_head.parameters():
        p.requires_grad = True
    for p in model.class_ctc_head.parameters():
        p.requires_grad = True
    if not freeze_encoder:
        for p in model.video_enc.parameters():
            p.requires_grad = True


def make_phase1_optimizer(model, steps_per_epoch, epochs, freeze_encoder):
    set_phase1_trainable(model, freeze_encoder)
    groups = []

    def add_group(params, lr, name):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append({"name": name, "params": params, "lr": lr, "weight_decay": 0.01})

    add_group(model.ch_adapt.parameters(), 2e-4 if freeze_encoder else 3e-4, "adapter")
    add_group(model.ctc_head.parameters(), 8e-5 if freeze_encoder else 1e-4, "ctc")
    add_group(model.class_ctc_head.parameters(), 6e-5 if freeze_encoder else 8e-5, "actc")
    if not freeze_encoder:
        add_group(model.video_enc.parameters(), 3e-5, "encoder")

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
            eeg_v, texts_v, _, _, _, _ = batch
            beam_groups = model.decode_ctc(eeg_v.to(device).to(torch.bfloat16), beam_width=10, nbest=3)
            for beams, ref in zip(beam_groups, texts_v):
                ref_lower = ref.lower()
                beam_texts = [(b[0] or "").strip() for b in beams]
                top = beam_texts[0] if beam_texts else ""
                oracle = min(beam_texts, key=lambda x: (compute_wer(x, ref_lower), compute_cer(x, ref_lower))) if beam_texts else ""
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
                if len(ref.split()) > 3 or shown < 2:
                    print(f"  REF: '{ref[:100]}'")
                    print(f"  TOP1: '{top[:100]}'  CER={cer_scores[-1]:.2f} WER={wer_scores[-1]:.2f}")
                    if len(beam_texts) > 1:
                        print(f"  TOP2: '{beam_texts[1][:100]}'")
                    if len(beam_texts) > 2:
                        print(f"  TOP3: '{beam_texts[2][:100]}'")
                    shown += 1
                if shown >= 4:
                    break
            if shown >= 4:
                break

    mean_cer = float(np.mean(cer_scores)) if cer_scores else float("inf")
    if cer_scores:
        v_acc = vowel_correct / max(vowel_total, 1)
        c_acc = cons_correct / max(cons_total, 1)
        print(f"  Mean CER: {mean_cer:.3f}  Mean WER: {np.mean(wer_scores):.3f}")
        print(f"  Oracle CER: {np.mean(oracle_cer_scores):.3f}  Oracle WER: {np.mean(oracle_wer_scores):.3f}")
        print(f"  Beam diversity: {np.mean(diversity_scores):.3f}")
        print(f"  Vowel recall: {v_acc:.2f} ({vowel_correct}/{vowel_total})  "
              f"Consonant recall: {c_acc:.2f} ({cons_correct}/{cons_total})")

    torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_ep{epoch}.pt")
    ckpt_vol.commit()
    if mean_cer < best_val_cer:
        best_val_cer = mean_cer
        torch.save(model.state_dict(), f"{ckpt_prefix}_phase1_best.pt")
        print(f"  ✓ New best val CER {best_val_cer:.3f}")
    return best_val_cer


def _train_phase1_stage(model, epochs, train_loader, val_loader, ckpt_prefix,
                        epoch_offset=0, best_train_ctc=float("inf"),
                        best_val_cer=float("inf"), freeze_encoder=False,
                        stage_name=""):
    device = next(model.parameters()).device
    optimizer, scheduler, groups = make_phase1_optimizer(
        model, len(train_loader), epochs, freeze_encoder=freeze_encoder)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"── Phase 1: {stage_name} ──")
    print(f"  Phase 1 trainable params: {sum(p.numel() for p in trainable)/1e6:.1f}M")
    print(f"  Phase 1 groups: {groups}")

    for local_epoch in range(1, epochs + 1):
        epoch = epoch_offset + local_epoch
        model.train()
        tot_ctc = tot_class = tot_div = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            eeg, _, c_ids, c_lens, class_ids, class_lens = batch
            eeg = eeg.to(device).to(torch.bfloat16)
            c_ids = c_ids.to(device)
            c_lens = c_lens.to(device)
            class_ids = class_ids.to(device)
            class_lens = class_lens.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(eeg, c_ids, c_lens, class_ids, class_lens)
                loss = (10.0 * out["loss_ctc"] + 0.8 * out["loss_class"] + 0.5 * out["loss_div"]) / ACCUM_STEPS

            loss.backward()
            tot_ctc += out["loss_ctc"].item()
            tot_class += out["loss_class"].item()
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
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | CTC:{tot_ctc/n:.3f} ACTC:{tot_class/n:.3f} Div:{tot_div/n:.4f} | LR0:{lr:.2e}")

        if (tot_ctc / n) < best_train_ctc:
            best_train_ctc = tot_ctc / n
            print(f"  ✓ New best train CTC {best_train_ctc:.3f}")

        if epoch % 3 == 0:
            best_val_cer = _validate_phase1(model, val_loader, epoch, ckpt_prefix, best_val_cer)

    return best_train_ctc, best_val_cer


def run_phase1_curriculum(model, epochs_p1, train_loader_mix, train_loader_pure,
                          val_loader, ckpt_prefix, start_epoch=0,
                          best_train_ctc=float("inf"), best_val_cer=float("inf")):
    mix_epochs = min(PHASE1_MIX_EPOCHS, epochs_p1)
    current_epoch = start_epoch

    if current_epoch < mix_epochs and train_loader_mix is not None:
        stage_epochs = mix_epochs - current_epoch
        print(f"[Phase1] Stage A: frozen-encoder lexical alignment for {stage_epochs} epochs")
        best_train_ctc, best_val_cer = _train_phase1_stage(
            model, epochs=stage_epochs, train_loader=train_loader_mix,
            val_loader=val_loader, ckpt_prefix=ckpt_prefix,
            epoch_offset=current_epoch, best_train_ctc=best_train_ctc,
            best_val_cer=best_val_cer, freeze_encoder=True,
            stage_name="frozen V50/V20 alignment + ACTC")
        current_epoch = mix_epochs

    if current_epoch < epochs_p1:
        stage_epochs = epochs_p1 - current_epoch
        print(f"[Phase1] Stage B: low-LR ZuCo consolidation for {stage_epochs} epochs")
        best_train_ctc, best_val_cer = _train_phase1_stage(
            model, epochs=stage_epochs, train_loader=train_loader_pure,
            val_loader=val_loader, ckpt_prefix=ckpt_prefix,
            epoch_offset=current_epoch, best_train_ctc=best_train_ctc,
            best_val_cer=best_val_cer, freeze_encoder=False,
            stage_name="unfrozen low-LR consolidation")

    return best_train_ctc, best_val_cer


# ─── Modal functions ──────────────────────────────────────────────────────────

@app.function(image=image, gpu="H100", timeout=86400,
              volumes={"/data": data_vol, "/persist": ckpt_vol, "/v20": v20_vol, "/v50": v50_vol},
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
        d, batch_size=4, shuffle=shuf, collate_fn=collate_fn)
    train_loader_pure = mkloader(train_zuco, True)
    val_loader        = mkloader(val_ds, False)
    if len(ds_inner) > 0:
        train_loader_mix = mkloader(train_mix, True)

    model = EEG_CTC_V12().to(torch.bfloat16).cuda()

    # ── Resume logic ──────────────────────────────────────────────────────────
    import glob as _glob
    p1_ckpts = sorted(_glob.glob("/persist/v12_phase1_ep*.pt"))
    best_p1 = "/persist/v12_phase1_best.pt"
    final = "/persist/v12_final.pt"
    p1_start_epoch = 0
    best_val_cer = float("inf")
    best_train_ctc = float("inf")

    if Path(final).exists():
        print("✓ V12 already complete.")
        return

    if Path(best_p1).exists() and p1_ckpts:
        last_p1       = p1_ckpts[-1]
        last_p1_epoch = int(last_p1.split("_ep")[-1].replace(".pt", ""))
        model.load_state_dict(torch.load(last_p1, map_location="cuda"))
        print(f"✓ Resuming Phase 1 from epoch {last_p1_epoch}")
        p1_start_epoch = last_p1_epoch
    else:
        if not load_v50_phase1_warmstart(model):
            if not load_v20_phase1_warmstart(model):
                load_pretrained_videomae_encoder(model.video_enc)

    print(f"\n{'='*60}")
    print(" V12: conservative CTC-first branch")
    print(" warm-start + ACTC + staged unfreeze")
    print(f" {'(RESUMED)' if p1_ckpts else 'FRESH START'}")
    print(f"{'='*60}\n")

    best_train_ctc, best_val_cer = run_phase1_curriculum(
        model, epochs_p1=epochs_p1,
        train_loader_mix=train_loader_mix,
        train_loader_pure=train_loader_pure,
        val_loader=val_loader,
        ckpt_prefix="/persist/v12",
        start_epoch=p1_start_epoch,
        best_train_ctc=best_train_ctc,
        best_val_cer=best_val_cer,
    )

    if Path(best_p1).exists():
        model.load_state_dict(torch.load(best_p1, map_location="cuda"))

    torch.save(model.state_dict(), "/persist/v12_final.pt")
    ckpt_vol.commit()
    print("\n✓ V12 complete. Saved /persist/v12_final.pt")


@app.local_entrypoint()
def main(mode: str = "pipeline", epochs_p1: int = 24):
    if mode == "pipeline":
        print(f"Launching V12: Phase1={epochs_p1}ep")
        run_pipeline.remote(epochs_p1=epochs_p1)
    else:
        print(f"Unknown mode '{mode}'. Use: pipeline")
