import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from pathlib import Path


app = modal.App("eeg-vlm-v401-audio-reset")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v401", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)


def _download_models():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0",
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

# V401: fresh audio-style reset.
# EEG is treated as a temporal modality, not a rasterized image.
# Stack:
#   1. 192-dim EEG band timeline
#   2. learned temporal conv frontend
#   3. temporal transformer
#   4. projection to pseudo-mel features
#   5. Whisper-tiny text decoder
#
# No old checkpoints, no CTC main path, no teacher cache, no retrieval.

WHISPER_PROCESSOR = None
WHISPER_PROMPT_IDS = None
MAX_TARGET_LEN = 96
ACCUM_STEPS = 4
TIME_STEPS = 1024
EEG_DIM = 192
LATENT_DIM = 256


def get_whisper_processor():
    global WHISPER_PROCESSOR, WHISPER_PROMPT_IDS
    if WHISPER_PROCESSOR is None:
        from transformers import WhisperProcessor

        WHISPER_PROCESSOR = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
        WHISPER_PROMPT_IDS = WHISPER_PROCESSOR.get_decoder_prompt_ids(language="english", task="transcribe")
    return WHISPER_PROCESSOR


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


def normalize_eeg(eeg, target_ch=64, target_t=TIME_STEPS):
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
                        self.samples.append({"eeg": normed, "text": normalize_text(text)})
        print(f"[Dataset] total={len(self.samples)} skipped={skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["eeg"], s["text"]


def collate_fn(batch):
    processor = get_whisper_processor()
    eegs, texts = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs))
    tok = processor.tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=MAX_TARGET_LEN,
        return_tensors="pt",
    )
    labels = tok.input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    lengths = (tok.input_ids != processor.tokenizer.pad_token_id).sum(dim=1)
    return eegs, list(texts), labels, lengths


class EEGAudioWhisper(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import WhisperForConditionalGeneration

        self.frontend = nn.Sequential(
            nn.Conv1d(EEG_DIM, LATENT_DIM, kernel_size=7, padding=3),
            nn.GroupNorm(16, LATENT_DIM),
            nn.GELU(),
            nn.Conv1d(LATENT_DIM, LATENT_DIM, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, LATENT_DIM),
            nn.GELU(),
            nn.Conv1d(LATENT_DIM, LATENT_DIM, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(16, LATENT_DIM),
            nn.GELU(),
        )
        self.time_steps = TIME_STEPS // 4
        self.pos = nn.Parameter(torch.randn(1, self.time_steps, LATENT_DIM) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=LATENT_DIM,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.to_mel = nn.Sequential(
            nn.LayerNorm(LATENT_DIM),
            nn.Linear(LATENT_DIM, 80),
        )
        self.whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        self.length_head = nn.Sequential(
            nn.LayerNorm(LATENT_DIM),
            nn.Linear(LATENT_DIM, 1),
        )

    def encode_audio_like(self, traj):
        x = traj.transpose(1, 2)
        x = self.frontend(x).transpose(1, 2)
        x = self.temporal(x + self.pos[:, : x.shape[1]])
        mel = self.to_mel(x).transpose(1, 2)
        mel = F.interpolate(mel, size=3000, mode="linear", align_corners=False)
        return mel, x

    def forward(self, traj, labels, target_lens):
        mel, latent = self.encode_audio_like(traj)
        out = self.whisper(input_features=mel, labels=labels)
        pred_lens = F.softplus(self.length_head(latent.mean(dim=1)).squeeze(-1)) + 1.0
        loss_len = F.l1_loss(torch.log1p(pred_lens), torch.log1p(target_lens.float()))
        return {"loss_text": out.loss, "loss_len": loss_len}

    @torch.no_grad()
    def generate(self, traj, max_new_tokens=80):
        processor = get_whisper_processor()
        mel, _ = self.encode_audio_like(traj.to(next(self.parameters()).dtype))
        gen_ids = self.whisper.generate(
            input_features=mel,
            forced_decoder_ids=WHISPER_PROMPT_IDS,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            early_stopping=True,
        )
        return [normalize_text(t) for t in processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)]


def set_trainable(model, freeze_whisper_body):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.frontend.parameters():
        p.requires_grad = True
    for p in model.temporal.parameters():
        p.requires_grad = True
    for p in model.to_mel.parameters():
        p.requires_grad = True
    for p in model.length_head.parameters():
        p.requires_grad = True
    for p in model.whisper.proj_out.parameters():
        p.requires_grad = True
    for p in model.whisper.model.decoder.layers[-2:].parameters():
        p.requires_grad = True
    if not freeze_whisper_body:
        for p in model.whisper.model.encoder.layers[-2:].parameters():
            p.requires_grad = True
        for p in model.whisper.model.decoder.layers[:-2].parameters():
            p.requires_grad = True


def make_optimizer(model, steps_per_epoch, epochs, freeze_whisper_body):
    set_trainable(model, freeze_whisper_body)
    groups = []

    def add(params, lr):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": 0.01})

    add(model.frontend.parameters(), 3e-4)
    add(model.temporal.parameters(), 2e-4)
    add(model.to_mel.parameters(), 2e-4)
    add(model.length_head.parameters(), 8e-5)
    add(model.whisper.proj_out.parameters(), 6e-5)
    add(model.whisper.model.decoder.layers[-2:].parameters(), 5e-5)
    if not freeze_whisper_body:
        add(model.whisper.model.encoder.layers[-2:].parameters(), 1.5e-5)
        add(model.whisper.model.decoder.layers[:-2].parameters(), 1.2e-5)

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


def train_stage(model, epochs, train_loader, val_loader, ckpt_prefix, start_epoch, best_wer, freeze_whisper_body, stage_name):
    device = next(model.parameters()).device
    optimizer, scheduler = make_optimizer(model, len(train_loader), epochs, freeze_whisper_body)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"── {stage_name} ──")
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
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=10.0),
)
def run_pipeline(epochs_p1: int = 2):
    download_zuco(Path("/data/EEG_Text"), data_vol)
    get_whisper_processor()

    ds = EEGDataset("/data/EEG_Text")
    n_train = int(0.9 * len(ds))
    indices = list(range(len(ds)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    mkloader = lambda d, shuf: torch.utils.data.DataLoader(d, batch_size=4, shuffle=shuf, collate_fn=collate_fn)
    train_loader = mkloader(train_ds, True)
    val_loader = mkloader(val_ds, False)

    model = EEGAudioWhisper().to(torch.bfloat16).cuda()

    print("\n============================================================")
    print(" V401: EEG timeline -> conv audio frontend -> temporal transformer")
    print("       -> pseudo-mel -> Whisper-tiny decoder")
    print(" FRESH START")
    print("============================================================\n")

    best_wer = float("inf")
    stage_a = min(1, epochs_p1)
    if stage_a > 0:
        best_wer = train_stage(
            model, stage_a, train_loader, val_loader,
            ckpt_prefix="/persist/v401",
            start_epoch=0,
            best_wer=best_wer,
            freeze_whisper_body=True,
            stage_name="Stage A: frontend + late decoder only",
        )
    if epochs_p1 > stage_a:
        best_wer = train_stage(
            model, epochs_p1 - stage_a, train_loader, val_loader,
            ckpt_prefix="/persist/v401",
            start_epoch=stage_a,
            best_wer=best_wer,
            freeze_whisper_body=False,
            stage_name="Stage B: unfreeze late encoder/decoder blocks",
        )

    best_path = Path("/persist/v401_best.pt")
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cuda"))
    torch.save(model.state_dict(), "/persist/v401_final.pt")
    ckpt_vol.commit()
    print("\n✓ V401 complete. Saved /persist/v401_final.pt")


@app.local_entrypoint()
def main(epochs_p1: int = 2):
    print(f"Launching V401: Phase1={epochs_p1}ep")
    run_pipeline.remote(epochs_p1=epochs_p1)
