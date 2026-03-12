import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from pathlib import Path


app = modal.App("eeg-vlm-v400-video2text-reset")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v400", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)


def _download_models():
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from torchvision.models import resnet18, ResNet18_Weights

    AutoTokenizer.from_pretrained("google/byt5-small")
    T5ForConditionalGeneration.from_pretrained("google/byt5-small")
    resnet18(weights=ResNet18_Weights.DEFAULT)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "transformers>=4.45.0",
        "numpy",
        "scipy",
        "jiwer",
        "h5py",
        "mne",
        "pandas",
        "accelerate",
        "osfclient",
        "editdistance",
        "openneuro-py",
    ])
    .run_function(_download_models)
)

# V400: full architectural reset.
# Keep only the EEG-as-video concept.
# New stack:
#   1. exact topographic EEG video
#   2. pretrained image encoder on each frame
#   3. temporal transformer over frame sequence
#   4. Perceiver-style latent resampler
#   5. byte-level T5 decoder
#
# No CTC, no Whisper bridge, no retrieval teacher, no old checkpoints.

TOKENIZER = None
NUM_EEG_BANDS = 3
FRAME_COUNT = 128
GRID_SIZE = 64
LATENT_QUERIES = 32
MAX_TARGET_LEN = 96
ACCUM_STEPS = 4


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_cer(pred, ref):
    import editdistance
    if not ref:
        return 0.0 if not pred else 1.0
    return min(editdistance.eval(pred, ref) / max(len(ref), 1), 2.0)


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
    zuco_v1 = base_path / "ZuCo_v1"
    zuco_v2 = base_path / "ZuCo_v2"
    for d in (zuco_v1, zuco_v2):
        d.mkdir(parents=True, exist_ok=True)

    expected = {
        zuco_v1 / "resultsZAB_SR.mat",
        zuco_v1 / "resultsZDM_SR.mat",
        zuco_v1 / "resultsZAB_NR.mat",
        zuco_v1 / "resultsZDM_NR.mat",
        zuco_v2 / "resultsYAC_NR.mat",
        zuco_v2 / "resultsYAG_NR.mat",
        zuco_v2 / "resultsYAK_NR.mat",
    }
    if all(verify_mat_file(path) for path in expected):
        print(f"[ZuCo] cache ready: {len(expected)}/{len(expected)} .mat files")
        return

    def fetch(project_id, target_dir, names):
        res = __import__("subprocess").run(
            ["osf", "-p", project_id, "list"], capture_output=True, text=True
        )
        paths = res.stdout.splitlines()
        for name in names:
            local = target_dir / name
            if verify_mat_file(local):
                continue
            remote = next((p for p in paths if p.strip().endswith(name)), None)
            if remote:
                print(f"  Fetching {name}...")
                __import__("subprocess").run(
                    ["osf", "-p", project_id, "fetch", remote.strip(), str(local)],
                    timeout=900,
                )

    fetch("q3zws", zuco_v1, ["resultsZAB_SR.mat", "resultsZDM_SR.mat", "resultsZAB_NR.mat", "resultsZDM_NR.mat"])
    fetch("2urht", zuco_v2, ["resultsYAC_NR.mat", "resultsYAG_NR.mat", "resultsYAK_NR.mat"])
    vol.commit()


def load_mat_any(path):
    import scipy.io as sio
    import h5py

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
            pass


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

    theta = band_env(4, 8, eeg)
    alpha = band_env(8, 13, eeg)
    gamma = band_env(30, 80, eeg)
    combined = np.concatenate([theta.T, alpha.T, gamma.T], axis=1)
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
        s = self.samples[idx]
        return s["eeg"], normalize_text(s["text"])


def collate_fn(batch):
    global TOKENIZER
    eegs, texts = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs))
    tok = TOKENIZER(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_TARGET_LEN,
        return_tensors="pt",
    )
    labels = tok.input_ids.clone()
    labels[labels == TOKENIZER.pad_token_id] = -100
    lengths = (tok.input_ids != TOKENIZER.pad_token_id).sum(dim=1)
    return eegs, list(texts), labels, lengths


class ExactTopographicRasterizer(nn.Module):
    def __init__(self, grid_size=GRID_SIZE, frame_count=FRAME_COUNT, n_electrodes=64, n_bands=NUM_EEG_BANDS):
        super().__init__()
        self.grid_size = grid_size
        self.frame_count = frame_count
        self.n_electrodes = n_electrodes
        self.n_bands = n_bands
        import mne

        montage = mne.channels.make_standard_montage("standard_1020")
        ch_names = montage.ch_names[:n_electrodes]
        xyz = np.stack([montage.get_positions()["ch_pos"][name] for name in ch_names], axis=0)
        pos_2d = xyz[:, :2].astype(np.float32)
        pos_2d -= pos_2d.mean(axis=0, keepdims=True)
        pos_2d /= np.abs(pos_2d).max() + 1e-6
        pos = torch.from_numpy(pos_2d)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(1, -1, grid_size),
            torch.linspace(-1, 1, grid_size),
            indexing="ij",
        )
        px = grid_x.flatten().unsqueeze(1)
        py = grid_y.flatten().unsqueeze(1)
        ex = pos[:, 0].unsqueeze(0)
        ey = pos[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex) ** 2 + (py - ey) ** 2)
        w = 1.0 / (dist + 1e-4) ** 2.0
        r = torch.sqrt(px ** 2 + py ** 2)
        w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("W", w)

    def forward(self, traj):
        bsz, timesteps, _ = traj.shape
        x = traj.reshape(bsz, timesteps, self.n_bands, self.n_electrodes)
        stride = timesteps // self.frame_count
        x = x[:, : self.frame_count * stride].reshape(
            bsz, self.frame_count, stride, self.n_bands, self.n_electrodes
        ).mean(dim=2)
        x = x.reshape(bsz * self.frame_count, self.n_bands, self.n_electrodes)
        maps = x @ self.W.T.to(device=traj.device, dtype=traj.dtype)
        maps = maps.reshape(bsz, self.frame_count, self.n_bands, self.grid_size, self.grid_size)
        maps = maps.clamp(min=0.0)
        denom = maps.amax(dim=(-1, -2), keepdim=True).clamp(min=1e-6)
        return maps / denom


class FrameEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.out_dim = 512

    def forward(self, frames):
        feats = self.stem(frames).flatten(1)
        return feats


class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_queries=LATENT_QUERIES, num_layers=2, num_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                "attn": nn.MultiheadAttention(dim, num_heads, batch_first=True),
                "norm1": nn.LayerNorm(dim),
                "ff": nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                ),
            }))

    def forward(self, seq):
        q = self.queries.expand(seq.shape[0], -1, -1)
        for layer in self.layers:
            attn_out, _ = layer["attn"](q, seq, seq, need_weights=False)
            q = layer["norm1"](q + attn_out)
            q = q + layer["ff"](q)
        return q


class EEGVideo2Text(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import T5ForConditionalGeneration

        self.rasterizer = ExactTopographicRasterizer()
        self.frame_encoder = FrameEncoder()
        self.frame_proj = nn.Linear(self.frame_encoder.out_dim, 768)
        self.frame_pos = nn.Parameter(torch.randn(1, FRAME_COUNT, 768) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.resampler = PerceiverResampler(768, num_queries=LATENT_QUERIES, num_layers=2, num_heads=8)
        self.text_model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
        self.enc_proj = nn.Linear(768, self.text_model.config.d_model)
        self.length_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 1),
        )

        if hasattr(self.text_model, "encoder"):
            for p in self.text_model.encoder.parameters():
                p.requires_grad = False

    def encode_video(self, traj):
        frames = self.rasterizer(traj)
        bsz, frames_n, ch, h, w = frames.shape
        flat = frames.reshape(bsz * frames_n, ch, h, w)
        feats = self.frame_encoder(flat)
        seq = self.frame_proj(feats).reshape(bsz, frames_n, 768)
        seq = seq + self.frame_pos[:, :frames_n]
        seq = self.temporal(seq)
        latents = self.resampler(seq)
        return latents, seq

    def forward(self, traj, labels, target_lens):
        from transformers.modeling_outputs import BaseModelOutput

        latents, seq = self.encode_video(traj)
        enc = self.enc_proj(latents)
        enc_out = BaseModelOutput(last_hidden_state=enc)
        out = self.text_model(encoder_outputs=enc_out, labels=labels)
        pred_lens = F.softplus(self.length_head(seq.mean(dim=1)).squeeze(-1)) + 1.0
        loss_len = F.l1_loss(torch.log1p(pred_lens), torch.log1p(target_lens.float()))
        return {
            "loss_text": out.loss,
            "loss_len": loss_len,
        }

    @torch.no_grad()
    def generate(self, traj, max_new_tokens=80):
        from transformers.modeling_outputs import BaseModelOutput

        latents, _ = self.encode_video(traj.to(next(self.parameters()).dtype))
        enc = self.enc_proj(latents)
        enc_out = BaseModelOutput(last_hidden_state=enc)
        gen_ids = self.text_model.generate(
            encoder_outputs=enc_out,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            early_stopping=True,
        )
        return [normalize_text(t) for t in TOKENIZER.batch_decode(gen_ids, skip_special_tokens=True)]


def set_trainable(model, freeze_frames):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.frame_proj.parameters():
        p.requires_grad = True
    for p in model.temporal.parameters():
        p.requires_grad = True
    for p in model.resampler.parameters():
        p.requires_grad = True
    for p in model.enc_proj.parameters():
        p.requires_grad = True
    for p in model.length_head.parameters():
        p.requires_grad = True
    for p in model.text_model.decoder.parameters():
        p.requires_grad = True
    for p in model.text_model.lm_head.parameters():
        p.requires_grad = True
    for p in model.text_model.shared.parameters():
        p.requires_grad = True
    if not freeze_frames:
        for p in model.frame_encoder.stem[-2:].parameters():
            p.requires_grad = True


def make_optimizer(model, steps_per_epoch, epochs, freeze_frames):
    set_trainable(model, freeze_frames=freeze_frames)
    groups = []

    def add(params, lr):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": 0.01})

    add(model.frame_proj.parameters(), 2e-4)
    add(model.temporal.parameters(), 2e-4 if freeze_frames else 1.5e-4)
    add(model.resampler.parameters(), 2e-4)
    add(model.enc_proj.parameters(), 1.5e-4)
    add(model.length_head.parameters(), 8e-5)
    add(model.text_model.decoder.parameters(), 8e-5)
    add(model.text_model.lm_head.parameters(), 8e-5)
    add(model.text_model.shared.parameters(), 4e-5)
    if not freeze_frames:
        add(model.frame_encoder.stem[-2:].parameters(), 8e-6)

    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in groups],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.2,
    )
    return optimizer, scheduler


def validate(model, val_loader, epoch, ckpt_prefix, best_wer):
    device = next(model.parameters()).device
    model.eval()
    cer_scores, wer_scores = [], []
    print(f"── Val Epoch {epoch} ──")
    with torch.no_grad():
        shown = 0
        for eegs, texts, _, _ in val_loader:
            preds = model.generate(eegs.to(device).to(torch.bfloat16))
            for pred, ref in zip(preds, texts):
                cer = compute_cer(pred, ref)
                wer = compute_wer(pred, ref)
                cer_scores.append(cer)
                wer_scores.append(wer)
                if shown < 4:
                    print(f"  REF: '{ref[:120]}'")
                    print(f"  GEN: '{pred[:120]}'  CER={cer:.2f} WER={wer:.2f}")
                    shown += 1
            if shown >= 4:
                break

    mean_cer = float(np.mean(cer_scores)) if cer_scores else float("inf")
    mean_wer = float(np.mean(wer_scores)) if wer_scores else float("inf")
    print(f"  Mean CER: {mean_cer:.3f}  Mean WER: {mean_wer:.3f}")
    torch.save(model.state_dict(), f"{ckpt_prefix}_ep{epoch}.pt")
    ckpt_vol.commit()
    if mean_wer < best_wer:
        best_wer = mean_wer
        torch.save(model.state_dict(), f"{ckpt_prefix}_best.pt")
        ckpt_vol.commit()
        print(f"  ✓ New best checkpoint: WER={best_wer:.3f}")
    return best_wer


def train_stage(model, epochs, train_loader, val_loader, ckpt_prefix, start_epoch, best_wer, freeze_frames, stage_name):
    device = next(model.parameters()).device
    optimizer, scheduler = make_optimizer(model, len(train_loader), epochs, freeze_frames)
    print(f"── {stage_name} ──")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Trainable params: {trainable:.1f}M")

    for local_epoch in range(1, epochs + 1):
        epoch = start_epoch + local_epoch
        model.train()
        tot_text = tot_len = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            eegs, _, labels, lengths = batch
            eegs = eegs.to(device).to(torch.bfloat16)
            labels = labels.to(device)
            lengths = lengths.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(eegs, labels, lengths)
                loss = (out["loss_text"] + 0.25 * out["loss_len"]) / ACCUM_STEPS
            loss.backward()
            tot_text += out["loss_text"].item()
            tot_len += out["loss_len"].item()
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
        print(f"Epoch {epoch:3d} | TXT:{tot_text/n:.3f} LEN:{tot_len/n:.3f} | LR0:{lr:.2e}")
        best_wer = validate(model, val_loader, epoch, ckpt_prefix, best_wer)

    return best_wer


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={"/data": data_vol, "/persist": ckpt_vol},
    retries=modal.Retries(max_retries=3, backoff_coefficient=1.0, initial_delay=10.0),
)
def run_pipeline(epochs_p1: int = 2):
    from transformers import AutoTokenizer

    download_zuco(Path("/data/EEG_Text"), data_vol)
    ds = EEGDataset("/data/EEG_Text")
    n_train = int(0.9 * len(ds))
    indices = list(range(len(ds)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    global TOKENIZER
    TOKENIZER = AutoTokenizer.from_pretrained("google/byt5-small")

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    mkloader = lambda d, shuf: torch.utils.data.DataLoader(d, batch_size=4, shuffle=shuf, collate_fn=collate_fn)
    train_loader = mkloader(train_ds, True)
    val_loader = mkloader(val_ds, False)

    model = EEGVideo2Text().to(torch.bfloat16).cuda()

    print("\n============================================================")
    print(" V400: exact EEG-video -> ResNet frame encoder -> temporal transformer")
    print("       -> Perceiver latent resampler -> ByT5 byte decoder")
    print(" FRESH START")
    print("============================================================\n")

    best_wer = float("inf")
    stage_a = min(1, epochs_p1)
    if stage_a > 0:
        best_wer = train_stage(
            model, stage_a, train_loader, val_loader,
            ckpt_prefix="/persist/v400",
            start_epoch=0,
            best_wer=best_wer,
            freeze_frames=True,
            stage_name="Stage A: frozen frame encoder",
        )
    if epochs_p1 > stage_a:
        best_wer = train_stage(
            model, epochs_p1 - stage_a, train_loader, val_loader,
            ckpt_prefix="/persist/v400",
            start_epoch=stage_a,
            best_wer=best_wer,
            freeze_frames=False,
            stage_name="Stage B: unfreeze late frame blocks",
        )

    best_path = Path("/persist/v400_best.pt")
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cuda"))

    torch.save(model.state_dict(), "/persist/v400_final.pt")
    ckpt_vol.commit()
    print("\n✓ V400 complete. Saved /persist/v400_final.pt")


@app.local_entrypoint()
def main(epochs_p1: int = 2):
    print(f"Launching V400: Phase1={epochs_p1}ep")
    run_pipeline.remote(epochs_p1=epochs_p1)
