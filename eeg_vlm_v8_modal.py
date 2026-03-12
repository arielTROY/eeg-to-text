
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import subprocess
from pathlib import Path

app = modal.App("eeg-vlm-v8-wordexact")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v8", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)

# Qwen2-VL requires latest transformers
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["curl"])
    .pip_install([
        "torch>=2.4.0", "transformers>=4.45.0", "peft>=0.12",
        "numpy", "scipy", "jiwer", "einops", "h5py", "mne", "pandas",
        "sentence-transformers", "accelerate", "osfclient", "qwen-vl-utils"
    ])
)

# ═══════════════════════════════════════════════════════════════════════════════
# V6: Video-VL Architecture (Qwen2-VL)
# ═══════════════════════════════════════════════════════════════════════════════

def verify_mat_file(path):
    if not path.exists() or path.stat().st_size < 1000: return False
    try:
        with open(path, 'rb') as f:
            header = f.read(128)
            return b'MATLAB 5.0' in header or b'MATLAB' in header
    except: return False

def download_zuco(base_path, vol):
    import subprocess
    zuco_v1 = base_path / "ZuCo_v1"; zuco_v2 = base_path / "ZuCo_v2"; inner_speech = base_path / "InnerSpeech"
    zuco_v1.mkdir(parents=True, exist_ok=True); zuco_v2.mkdir(parents=True, exist_ok=True); inner_speech.mkdir(parents=True, exist_ok=True)

    def fetch_dynamic(project_id, target_dir, names):
        res = subprocess.run(["osf", "-p", project_id, "list"], capture_output=True, text=True)
        paths = res.stdout.splitlines()
        for name in names:
            local = target_dir / name
            if not verify_mat_file(local):
                remote = next((p for p in paths if p.strip().endswith(name)), None)
                if remote:
                    print(f"  Fetching {name}...")
                    subprocess.run(["osf", "-p", project_id, "fetch", remote.strip(), str(local)], timeout=900)
    
    fetch_dynamic("q3zws", zuco_v1, ["resultsZAB_SR.mat", "resultsZDM_SR.mat", "resultsZAB_NR.mat", "resultsZDM_NR.mat"])
    fetch_dynamic("2urht", zuco_v2, ["resultsYAC_NR.mat", "resultsYAG_NR.mat", "resultsYAK_NR.mat"])
    
    if not any(inner_speech.rglob("*-epo.fif")):
        subprocess.run(["pip", "install", "openneuro-py"], check=False)
        import openneuro; openneuro.download(dataset="ds003626", target_dir=str(inner_speech), include=["derivatives/sub-01/"])
    vol.commit()

# --- Character Vocabulary for CTC (V6 Breakthrough) ---
CHAR_VOCAB = "_abcdefghijklmnopqrstuvwxyz0123456789.,!?'\" ()" # _ is blank
CHAR_TO_ID = {c: i for i, c in enumerate(CHAR_VOCAB)}
ID_TO_CHAR = {i: c for i, c in enumerate(CHAR_VOCAB)}

def text_to_char_ids(text):
    text = text.lower()
    return [CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID]

class MultiScaleRasterizer(nn.Module):
    def __init__(self, size=64, n_electrodes=64):
        super().__init__()
        angles = torch.linspace(0, 2 * np.pi, n_electrodes)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing='ij')
        px, py = gx.flatten().unsqueeze(1), gy.flatten().unsqueeze(1)
        ex, ey = pos_2d[:, 0].unsqueeze(0), pos_2d[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex)**2 + (py - ey)**2)
        w = 1.0 / (dist + 1e-4) ** 2.0
        r = torch.sqrt(px**2 + py**2)
        w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / (w.sum(dim=1, keepdim=True).clamp(min=1e-8))
        self.register_buffer("W", w)

    def forward(self, x):
        # x is (B, T, Ch) where Ch = 128 (64 voltage + 64 gamma)
        B, T, Ch = x.shape
        # Reshape to treat channels separately for spatial projection
        # x_split: (B*T, 2, 64)
        x_split = x.reshape(B*T, 2, 64)
        # Apply projection to each channel: (B*T, 2, 4096)
        proj = x_split @ self.W.T.to(device=x.device, dtype=x.dtype)
        # Reshape to (B, T, 2, 64, 64)
        imgs = proj.reshape(B, T, 2, 64, 64)
        return imgs

class SpatioTemporalBridge(nn.Module):
    def __init__(self, in_ch=768, out_ch=1024, n_layers=4, n_heads=16):
        super().__init__()
        # V8: 4-layer Transformer Bridge
        enc_layer = nn.TransformerEncoderLayer(
            d_model=in_ch, nhead=n_heads, dim_feedforward=in_ch*4,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj = nn.Linear(in_ch, out_ch)
        
    def forward(self, x):
        # x is (B, 64, 768)
        x = self.transformer(x)
        return self.proj(x)

class EEG_VLM_V8(nn.Module):
    def __init__(self, q_name):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType
        
        v_cfg = VideoMAEConfig(
            num_channels=2, # V8: Voltage + Gamma
            image_size=64, patch_size=16, 
            num_frames=128, tubelet_size=2, hidden_size=768
        )
        self.video_enc = VideoMAEModel(v_cfg)
        self.rasterizer = MultiScaleRasterizer()
        self.bridge = SpatioTemporalBridge(768, 1024)
        self.ctc_head = nn.Linear(768, len(CHAR_VOCAB))
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        
        self.llm = AutoModelForCausalLM.from_pretrained(q_name, torch_dtype=torch.bfloat16)
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=128, lora_alpha=256,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.llm = get_peft_model(self.llm, lora_cfg)
        self.llm_proj = nn.Linear(1024, 1536)

    def forward(self, traj, char_ids, char_lens, input_ids=None, labels=None):
        B, T, Ch = traj.shape
        imgs = self.rasterizer(traj) # (B, 128, 1, 64, 64)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        
        # V7: Keep 64 temporal steps (128 frames / 2 tubelet)
        v_seq = v_out.reshape(B, 64, 16, 768).mean(dim=2) # (B, 64, 768)
            
        v_seq_128 = F.interpolate(v_seq.transpose(1, 2), size=128, mode='linear', align_corners=False).transpose(1, 2)
        
        # CTC Loss
        ctc_logits = self.ctc_head(v_seq_128).transpose(0, 1)
        input_lens = torch.full((B,), 128, dtype=torch.long, device=traj.device)
        loss_ctc = self.ctc_loss_fn(F.log_softmax(ctc_logits, dim=-1), char_ids, input_lens, char_lens)
        
        # LM Loss with 64-token Prefix
        prefix = self.llm_proj(self.bridge(v_seq)) # (B, 64, 1536)
        
        loss_lock = torch.tensor(0.0, device=traj.device)
        if input_ids is not None:
            embed_fn = self.llm.get_input_embeddings()
            tok_embs = embed_fn(input_ids)
            
            # V8: InfoNCE Word Locking (Tokens matching)
            # We treat the first 64 tokens as candidates for exact alignment
            n_align = min(64, tok_embs.shape[1])
            if n_align > 2:
                # Contrastive similarity: dot product between brain tokens and text tokens
                sims = torch.matmul(prefix[:, :n_align, :], tok_embs[:, :n_align, :].transpose(1, 2))
                # Diagonals are the correct pairs
                labels_nce = torch.arange(n_align, device=traj.device).repeat(B, 1)
                loss_lock = F.cross_entropy(sims.reshape(-1, n_align), labels_nce.flatten())
            
            combined = torch.cat([prefix, tok_embs], dim=1)
            
            # labels prefix padding (64 tokens)
            combined_labels = torch.cat([
                torch.full((B, 64), -100, device=labels.device, dtype=labels.dtype),
                labels
            ], dim=1)
            out = self.llm(inputs_embeds=combined, labels=combined_labels)
            return out.loss, loss_ctc, loss_lock
        return None, loss_ctc, loss_lock

    @torch.no_grad()
    def generate(self, traj, tokenizer, max_tokens=60):
        traj = traj.to(next(self.parameters()).dtype)
        imgs = self.rasterizer(traj)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        v_seq = v_out.reshape(traj.shape[0], 64, 16, 768).mean(dim=2)
        v_seq_128 = F.interpolate(v_seq.transpose(1, 2), size=128, mode='linear', align_corners=False).transpose(1, 2)
        
        ctc_ids = self.ctc_head(v_seq_128).argmax(dim=-1)
        ctc_texts = []
        for b in range(traj.shape[0]):
            ids = ctc_ids[b].tolist()
            res = ""; prev = -1
            for i in ids:
                if i != prev and i != 0: res += ID_TO_CHAR.get(i, "")
                prev = i
            ctc_texts.append(res)
            
        # LLM Generate
        prefix = self.llm_proj(self.bridge(v_seq))
        out_ids = self.llm.generate(
            inputs_embeds=prefix, max_new_tokens=max_tokens,
            do_sample=True, temperature=0.6, top_p=0.9, pad_token_id=tokenizer.eos_token_id
        )
        return [tokenizer.decode(g, skip_special_tokens=True) for g in out_ids], ctc_texts

# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading & Training (V6)
# ═══════════════════════════════════════════════════════════════════════════════

def load_mat_any(path):
    import scipy.io as sio
    import h5py
    try:
        data = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        sentences = data['sentenceData']
        if not isinstance(sentences, (np.ndarray, list)): sentences = [sentences]
        for s in sentences:
            if hasattr(s, 'rawData') and hasattr(s, 'content') and isinstance(s.rawData, np.ndarray):
                yield s.rawData, str(s.content)
    except:
        try:
            with h5py.File(path, 'r') as f:
                if 'sentenceData' in f:
                    s_group = f['sentenceData']
                    # H5py groups can be tricky to iterate but let's try 
                    # ZuCo H5 structure: sentenceData/results is a list of refs
                    for key in s_group.keys():
                        s = s_group[key]
                        if 'rawData' in s and 'content' in s:
                            # Content is often stored as a reference to a char array
                            raw = np.array(s['rawData']).T
                            # Content extraction is complex in H5, let's use a simpler proxy if needed 
                            # but for now let's hope scipy handles most or we skip the complex H5 ones.
                            yield raw, "hdf5_content_placeholder"
        except: pass

def normalize_eeg(eeg, target_ch=64, target_t=128):
    if not isinstance(eeg, np.ndarray) or eeg.ndim != 2: return None
    ch, t = eeg.shape
    if ch < t: pass
    else: eeg = eeg.T; ch, t = eeg.shape
    if ch > target_ch: eeg = eeg[:target_ch, :]
    elif ch < target_ch: eeg = np.pad(eeg, ((0, target_ch - ch), (0, 0)))
    if t > target_t: eeg = eeg[:, :target_t]
    else: eeg = np.pad(eeg, ((0, 0), (0, target_t - t)))
    
    # V8: Spectral Gamma (30-100Hz) Injection
    from scipy.signal import welch
    # Compute power spectral density for each channel
    # This is a simplification: we'll just use the raw voltage as Ch1
    # and a smoothed envelope of the 30-100Hz band as Ch2.
    from scipy.signal import butter, filtfilt
    try:
        b, a = butter(4, [30/64, 60/64], btype='bandpass') 
        gamma = filtfilt(b, a, eeg, axis=1)
        gamma_env = np.abs(gamma)
        # V8.1: Temporal smoothing for Gamma envelope (5-frame window)
        gamma_env = np.convolve(gamma_env.flatten(), np.ones(5)/5, mode='same').reshape(gamma_env.shape)
    except:
        gamma_env = np.zeros_like(eeg)

    # Output shape (target_t, target_ch * 2)
    # Voltage first 64, Gamma next 64 for each time step
    combined = np.concatenate([eeg.T, gamma_env.T], axis=1) # (128, 128)
    return combined.astype(np.float32)

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, is_train=True):
        import mne
        self.samples = []
        p = Path(base_path)
        
        # ZuCo
        for zuco_path in [p/"ZuCo_v1", p/"ZuCo_v2"]:
            for mat in zuco_path.rglob("*.mat"):
                for raw, text in load_mat_any(mat):
                    normed = normalize_eeg(raw)
                    if normed is not None:
                        self.samples.append({'eeg': normed, 'text': text})
        
        # InnerSpeech
        for fif in (p/"InnerSpeech").rglob("*-epo.fif"):
            try:
                epochs = mne.read_epochs(str(fif), preload=True, verbose=False)
                epochs.resample(128)
                data = epochs.get_data()
                labels = epochs.metadata['condition'].tolist() if (epochs.metadata is not None and 'condition' in epochs.metadata.columns) else [f"word_{i}" for i in range(len(data))]
                for d, l in zip(data, labels):
                    normed = normalize_eeg(d)
                    if normed is not None:
                        self.samples.append({'eeg': normed, 'text': str(l)})
            except: continue
            
        print(f"Loaded {len(self.samples)} samples for {'Train' if is_train else 'Val'}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        char_ids = text_to_char_ids(s['text'])
        return s['eeg'], s['text'], char_ids

def collate_fn(batch, tokenizer):
    eegs, texts, chars = zip(*batch)
    eegs = torch.from_numpy(np.stack(eegs)) # (B, 128, 64)
    
    # Char padding for CTC
    char_lens = torch.tensor([len(c) for c in chars], dtype=torch.long)
    max_char = max(char_lens)
    char_padded = torch.zeros((len(chars), max_char), dtype=torch.long)
    for i, c in enumerate(chars): char_padded[i, :len(c)] = torch.tensor(c)
    
    # LLM padding
    enc = tokenizer(list(texts), padding=True, truncation=True, max_length=128, return_tensors="pt")
    return eegs, enc.input_ids, enc.input_ids.clone(), char_padded, char_lens

@app.function(
    image=image, gpu="H100", timeout=72000,
    volumes={"/data": data_vol, "/persist": ckpt_vol}
)
def train_v8(epochs=30, phase=1):
    from transformers import AutoTokenizer
    q_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(q_name)
    
    download_zuco(Path("/data/EEG_Text"), data_vol)
    ds = EEGDataset("/data/EEG_Text")
    train_size = int(0.9 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, len(ds) - train_size])
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=4, shuffle=True, 
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )
    
    model = EEG_VLM_V8(q_name).to(torch.bfloat16).cuda()
    
    if phase == 2:
        print("--- PHASE 2: Word-Exact Locking (High Pressure) ---")
        # Load Phase 1 weights (Epoch 18)
        ckpt_path = Path("/persist/v8_phase1_epoch18.pt")
        if ckpt_path.exists():
            print(f"Loading Phase 1 weights: {ckpt_path}")
            model.load_state_dict(torch.load(str(ckpt_path), map_location="cuda"))
        else:
            print("WARNING: Phase 1 checkpoint not found. Starting from scratch.")
            
        # Freeze Encoder but allow Bridge & LLM to learn
        for param in model.video_enc.parameters(): param.requires_grad = False
        for param in model.rasterizer.parameters(): param.requires_grad = False
        
        for param in model.bridge.parameters(): param.requires_grad = True
        for name, param in model.llm.named_parameters():
            if "lora_" in name: param.requires_grad = True
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) # Lower LR for precision
    else:
        print("--- PHASE 1: Temporal Alignment ---")
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print(f"Starting V8 Phase {phase} Training: {len(ds)} samples")
    for epoch in range(1, epochs + 1):
        model.train()
        total_lm, total_ctc, total_lock = 0, 0, 0
        for eeg, input_ids, labels, c_ids, c_lens in train_loader:
            eeg, input_ids, labels, c_ids, c_lens = [x.cuda() for x in [eeg, input_ids, labels, c_ids, c_lens]]
            eeg = eeg.to(torch.bfloat16)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss_lm, loss_ctc, loss_lock = model(eeg, c_ids, c_lens, input_ids, labels)
                
                # V8.1: ANNEALED LOCKING
                # Ramp from 1.0 to 15.0 over 10 epochs
                if phase == 2:
                    current_penalty = 1.0 + (14.0 * min(epoch/10, 1.0))
                else:
                    current_penalty = 0.5
                
                # V8.1: CTC BOOST (1.0 weight instead of 0.5)
                loss = loss_lm + 1.0 * loss_ctc + current_penalty * loss_lock
            
            loss.backward()
            optimizer.step()
            total_lm += loss_lm.item(); total_ctc += loss_ctc.item(); total_lock += loss_lock.item()
            
        print(f"Epoch {epoch} | LM: {total_lm/len(train_loader):.3f} | CTC: {total_ctc/len(train_loader):.3f} | Lock: {total_lock/len(train_loader):.4f}")
        
        # Val
        if epoch % 3 == 0:
            model.eval()
            print(f"--- Decoded Examples Epoch {epoch} ---")
            with torch.no_grad():
                # Try to find a batch with at least one ZuCo sentence
                sentence_found = False
                for eeg, input_ids, labels, c_ids, c_lens in val_loader:
                    llm_preds, ctc_preds = model.generate(eeg.cuda(), tokenizer)
                    
                    for b in range(eeg.shape[0]):
                        # We don't easily have the 'is_sentence' flag here without more logic, 
                        # so let's just show more samples to ensure we catch sentences.
                        ref_text = tokenizer.decode(input_ids[b], skip_special_tokens=True)
                        if len(ref_text.split()) > 3: # Likely a ZuCo sentence
                            print(f"  [SENT] REF='{ref_text[:120]}'")
                            print(f"         GEN='{llm_preds[b][:120]}'")
                            print(f"         CTC='{ctc_preds[b][:120]}'")
                        else:
                            print(f"  [WORD] REF='{ref_text}'")
                            print(f"         CTC='{ctc_preds[b]}'")
                    
                    break # Just one batch for speed
            torch.save(model.state_dict(), f"/persist/v8_phase{phase}_epoch{epoch}.pt")

@app.local_entrypoint()
def main(phase: int = 1, epochs: int = 40):
    train_v8.remote(epochs=epochs, phase=phase)
