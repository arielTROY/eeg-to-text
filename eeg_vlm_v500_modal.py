import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
from pathlib import Path


app = modal.App("eeg-vlm-v500-closedset-retrieval")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v500", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)


def _download_models():
    from transformers import WhisperForConditionalGeneration
    WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0",
        "transformers>=4.45.0",
        "numpy",
        "scipy",
        "h5py",
        "mne",
        "pandas",
        "accelerate",
        "osfclient",
        "editdistance",
        "openneuro-py",
        "sentencepiece",
    ])
    .run_function(_download_models)
)

# V500: closed-set retrieval branch.
# Purpose: maximize exact sentence matches on the dataset and test whether the
# EEG modality encoder contains sentence identity signal at all.

SP_MODEL = None
WP_VOCAB_SIZE = 0
TIME_STEPS = 1024
EEG_DIM = 192
LATENT_DIM = 256
EMB_DIM = 256
ACCUM_STEPS = 4


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with open(path, "rb") as f:
            return b"MATLAB" in f.read(128)
    except Exception:
        return False


def build_sentencepiece_model(texts, model_prefix="/persist/v500_wordpiece", vocab_size=256):
    import sentencepiece as spm

    text_path = f"{model_prefix}.txt"
    model_path = f"{model_prefix}.model"
    if not Path(model_path).exists():
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        with open(text_path, "w", encoding="utf-8") as f:
            for text in texts:
                norm = normalize_text(text).strip()
                if norm:
                    f.write(norm + "\n")
        spm.SentencePieceTrainer.train(
            input=text_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            bos_id=-1,
            eos_id=-1,
            pad_id=0,
            unk_id=1,
            input_sentence_size=100000,
            shuffle_input_sentence=True,
            split_by_whitespace=False,
        )
    return spm.SentencePieceProcessor(model_file=model_path)


def text_to_wp_ids(text):
    return SP_MODEL.encode(normalize_text(text), out_type=int) if SP_MODEL is not None else []


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
        res = __import__("subprocess").run(["osf", "-p", project_id, "list"], capture_output=True, text=True)
        paths = res.stdout.splitlines()
        for name in names:
            local = target_dir / name
            if verify_mat_file(local):
                continue
            remote = next((p for p in paths if p.strip().endswith(name)), None)
            if remote:
                print(f"  Fetching {name}...")
                __import__("subprocess").run(["osf", "-p", project_id, "fetch", remote.strip(), str(local)], timeout=900)

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
    return np.concatenate([theta.T, alpha.T, gamma.T], axis=1).astype(np.float32)


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
        return s["eeg"], s["text"], text_to_wp_ids(s["text"])


def collate_fn(batch):
    eegs, texts, wp_ids_list = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs))
    lengths = torch.tensor([len(x) for x in wp_ids_list], dtype=torch.long)
    max_len = max(lengths) if max(lengths) > 0 else 1
    wp_pad = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, ids in enumerate(wp_ids_list):
        if ids:
            wp_pad[i, :len(ids)] = torch.tensor(ids)
    return eegs, list(texts), wp_pad, lengths


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(WP_VOCAB_SIZE, EMB_DIM, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, 256, EMB_DIM) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=EMB_DIM,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.proj = nn.Sequential(
            nn.LayerNorm(EMB_DIM),
            nn.Linear(EMB_DIM, EMB_DIM),
        )

    def forward(self, wp_ids, lengths):
        x = self.embed(wp_ids)
        x = x + self.pos[:, : x.shape[1]]
        x = self.encoder(x)
        mask = (wp_ids != 0).float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)


class EEGEncoder(nn.Module):
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
        self.to_mel = nn.Sequential(nn.LayerNorm(LATENT_DIM), nn.Linear(LATENT_DIM, 80))
        whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
        self.whisper_encoder = whisper.model.encoder
        self.attn_pool = nn.MultiheadAttention(self.whisper_encoder.config.d_model, 8, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, self.whisper_encoder.config.d_model) * 0.02)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.whisper_encoder.config.d_model),
            nn.Linear(self.whisper_encoder.config.d_model, EMB_DIM),
        )

    def forward(self, traj):
        x = traj.transpose(1, 2)
        x = self.frontend(x).transpose(1, 2)
        x = self.temporal(x + self.pos[:, : x.shape[1]])
        mel = self.to_mel(x).transpose(1, 2)
        mel = F.interpolate(mel, size=3000, mode="linear", align_corners=False)
        states = self.whisper_encoder(input_features=mel).last_hidden_state
        q = self.query.expand(states.shape[0], -1, -1)
        pooled, _ = self.attn_pool(q, states, states, need_weights=False)
        pooled = pooled.squeeze(1)
        return F.normalize(self.proj(pooled), dim=-1)


class EEGSentenceRetrieval(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg_encoder = EEGEncoder()
        self.text_encoder = TextEncoder()
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def forward(self, eegs, wp_ids, lengths):
        eeg_emb = self.eeg_encoder(eegs)
        text_emb = self.text_encoder(wp_ids, lengths)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * eeg_emb @ text_emb.T
        labels = torch.arange(eegs.shape[0], device=eegs.device)
        loss_eeg = F.cross_entropy(logits, labels)
        loss_txt = F.cross_entropy(logits.T, labels)
        return {"loss": 0.5 * (loss_eeg + loss_txt), "eeg_emb": eeg_emb, "text_emb": text_emb, "logits": logits}


def set_trainable(model, freeze_whisper_body):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.eeg_encoder.frontend.parameters():
        p.requires_grad = True
    for p in model.eeg_encoder.temporal.parameters():
        p.requires_grad = True
    for p in model.eeg_encoder.to_mel.parameters():
        p.requires_grad = True
    for p in model.eeg_encoder.attn_pool.parameters():
        p.requires_grad = True
    for p in model.eeg_encoder.proj.parameters():
        p.requires_grad = True
    for p in model.text_encoder.parameters():
        p.requires_grad = True
    model.logit_scale.requires_grad = True
    if not freeze_whisper_body:
        for p in model.eeg_encoder.whisper_encoder.layers[-2:].parameters():
            p.requires_grad = True


def make_optimizer(model, steps_per_epoch, epochs, freeze_whisper_body):
    set_trainable(model, freeze_whisper_body)
    groups = []

    def add(params, lr):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": 0.01})

    add(model.eeg_encoder.frontend.parameters(), 3e-4)
    add(model.eeg_encoder.temporal.parameters(), 2e-4)
    add(model.eeg_encoder.to_mel.parameters(), 2e-4)
    add(model.eeg_encoder.attn_pool.parameters(), 1e-4)
    add(model.eeg_encoder.proj.parameters(), 1e-4)
    add(model.text_encoder.parameters(), 1e-4)
    if not freeze_whisper_body:
        add(model.eeg_encoder.whisper_encoder.layers[-2:].parameters(), 1.2e-5)

    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in groups],
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.2,
    )
    return optimizer, scheduler


@torch.no_grad()
def build_candidate_bank(model, all_texts):
    device = next(model.parameters()).device
    unique = []
    seen = set()
    for text in all_texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    wp_list = [text_to_wp_ids(t) for t in unique]
    lengths = torch.tensor([len(x) for x in wp_list], dtype=torch.long, device=device)
    max_len = max(lengths).item() if len(lengths) else 1
    wp_pad = torch.zeros(len(unique), max_len, dtype=torch.long, device=device)
    for i, ids in enumerate(wp_list):
        if ids:
            wp_pad[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    emb = model.text_encoder(wp_pad, lengths)
    return unique, emb


def validate(model, val_loader, epoch, ckpt_prefix, best_acc, all_texts):
    device = next(model.parameters()).device
    model.eval()
    candidates, cand_emb = build_candidate_bank(model, all_texts)
    top1_hits = 0
    top5_hits = 0
    cer_scores, wer_scores = [], []
    total = 0
    print(f"── Val Epoch {epoch} ──")
    with torch.no_grad():
        shown = 0
        for eegs, texts, _, _ in val_loader:
            eeg_emb = model.eeg_encoder(eegs.to(device).to(torch.bfloat16))
            sims = eeg_emb @ cand_emb.T
            topk = sims.topk(k=min(5, cand_emb.shape[0]), dim=-1).indices.cpu().tolist()
            for idxs, ref in zip(topk, texts):
                pred = candidates[idxs[0]]
                if pred == ref:
                    top1_hits += 1
                if ref in [candidates[i] for i in idxs]:
                    top5_hits += 1
                cer_scores.append(compute_cer(pred, ref))
                wer_scores.append(compute_wer(pred, ref))
                total += 1
                if shown < 4:
                    print(f"  REF: '{ref[:120]}'")
                    print(f"  TOP1: '{pred[:120]}'  CER={cer_scores[-1]:.2f} WER={wer_scores[-1]:.2f}")
                    print(f"  TOP5: {[candidates[i][:60] for i in idxs]}")
                    shown += 1
    top1 = top1_hits / max(total, 1)
    top5 = top5_hits / max(total, 1)
    mean_wer = float(np.mean(wer_scores)) if wer_scores else float("inf")
    mean_cer = float(np.mean(cer_scores)) if cer_scores else float("inf")
    print(f"  Top1: {top1:.3f}  Top5: {top5:.3f}  Mean CER: {mean_cer:.3f}  Mean WER: {mean_wer:.3f}")
    torch.save(model.state_dict(), f"{ckpt_prefix}_ep{epoch}.pt")
    ckpt_vol.commit()
    if top1 > best_acc:
        best_acc = top1
        torch.save(model.state_dict(), f"{ckpt_prefix}_best.pt")
        ckpt_vol.commit()
        print(f"  ✓ New best retrieval checkpoint: Top1={best_acc:.3f}")
    return best_acc


def train_stage(model, epochs, train_loader, val_loader, ckpt_prefix, start_epoch, best_acc, freeze_whisper_body, all_texts, stage_name):
    device = next(model.parameters()).device
    optimizer, scheduler = make_optimizer(model, len(train_loader), epochs, freeze_whisper_body)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"── {stage_name} ──")
    print(f"  Trainable params: {trainable:.1f}M")
    for local_epoch in range(1, epochs + 1):
        epoch = start_epoch + local_epoch
        model.train()
        tot_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            eegs, _, wp_ids, lengths = batch
            eegs = eegs.to(device).to(torch.bfloat16)
            wp_ids = wp_ids.to(device)
            lengths = lengths.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(eegs, wp_ids, lengths)
                loss = out["loss"] / ACCUM_STEPS
            loss.backward()
            tot_loss += out["loss"].item()
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
        print(f"Epoch {epoch:3d} | NCE:{tot_loss/n:.3f} | LR0:{lr:.2e}")
        best_acc = validate(model, val_loader, epoch, ckpt_prefix, best_acc, all_texts)
    return best_acc


@app.function(
    image=image,
    gpu="H100",
    timeout=86400,
    volumes={"/data": data_vol, "/persist": ckpt_vol},
    retries=modal.Retries(max_retries=2, backoff_coefficient=1.0, initial_delay=10.0),
)
def run_pipeline(epochs_p1: int = 3):
    download_zuco(Path("/data/EEG_Text"), data_vol)
    ds = EEGDataset("/data/EEG_Text")
    n_train = int(0.9 * len(ds))
    indices = list(range(len(ds)))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    global SP_MODEL, WP_VOCAB_SIZE
    train_texts = [ds.samples[i]["text"] for i in train_idx]
    SP_MODEL = build_sentencepiece_model(train_texts, model_prefix="/persist/v500_wordpiece", vocab_size=256)
    WP_VOCAB_SIZE = SP_MODEL.get_piece_size()

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    mkloader = lambda d, shuf: torch.utils.data.DataLoader(d, batch_size=16, shuffle=shuf, collate_fn=collate_fn)
    train_loader = mkloader(train_ds, True)
    val_loader = mkloader(val_ds, False)

    model = EEGSentenceRetrieval().to(torch.bfloat16).cuda()
    all_texts = [s["text"] for s in ds.samples]

    print("\n============================================================")
    print(" V500: EEG audio encoder -> sentence retrieval over text bank")
    print(" FRESH START")
    print("============================================================\n")

    best_acc = 0.0
    stage_a = min(1, epochs_p1)
    if stage_a > 0:
        best_acc = train_stage(
            model, stage_a, train_loader, val_loader,
            ckpt_prefix="/persist/v500",
            start_epoch=0,
            best_acc=best_acc,
            freeze_whisper_body=True,
            all_texts=all_texts,
            stage_name="Stage A: frozen Whisper encoder",
        )
    if epochs_p1 > stage_a:
        best_acc = train_stage(
            model, epochs_p1 - stage_a, train_loader, val_loader,
            ckpt_prefix="/persist/v500",
            start_epoch=stage_a,
            best_acc=best_acc,
            freeze_whisper_body=False,
            all_texts=all_texts,
            stage_name="Stage B: unfreeze late Whisper layers",
        )

    best_path = Path("/persist/v500_best.pt")
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location="cuda"))
    torch.save(model.state_dict(), "/persist/v500_final.pt")
    ckpt_vol.commit()
    print("\n✓ V500 complete. Saved /persist/v500_final.pt")


@app.local_entrypoint()
def main(epochs_p1: int = 3):
    print(f"Launching V500: Phase1={epochs_p1}ep")
    run_pipeline.remote(epochs_p1=epochs_p1)
