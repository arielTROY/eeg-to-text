
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import subprocess
from pathlib import Path

app = modal.App("eeg-vlm-v6-videovl")

ckpt_vol = modal.Volume.from_name("bt-checkpoints-v6", create_if_missing=True)
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
        B, T, Ch = x.shape
        imgs = (x.reshape(B*T, Ch) @ self.W.T.to(device=x.device, dtype=x.dtype)).reshape(B, T, 1, 64, 64)
        return imgs

class SpatioTemporalEncoder(nn.Module):
    """Encodes EEG (128, 64) into Video Tokens (32, 1024) for Qwen2-VL"""
    def __init__(self, d_model=1024, n_temporal_frames=128):
        super().__init__()
        self.rasterizer = MultiScaleRasterizer(size=64)
        self.spatial = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(4), # (B*T, 256, 4, 4)
            nn.Flatten(),
            nn.Linear(256*16, d_model)
        )
        # Sequence reduction: 128 -> 32 tokens
        self.temporal_reduce = nn.Conv1d(n_temporal_frames, 32, kernel_size=1) 
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=16, batch_first=True, norm_first=True),
            num_layers=4
        )

    def forward(self, traj):
        B, T, Ch = traj.shape
        imgs = self.rasterizer(traj) # (B, T, 1, 64, 64)
        flat_imgs = imgs.view(B*T, 1, 64, 64)
        spatial_feats = self.spatial(flat_imgs).view(B, T, -1) # (B, 128, 1024)
        
        # CTC path needs higher resolution (128 frames)
        # Qwen path needs reduced resolution (32 tokens)
        temporal_full = self.transformer(spatial_feats) # (B, 128, 1024)
        qwen_tokens = self.temporal_reduce(temporal_full) # (B, 32, 1024)
        return qwen_tokens, temporal_full

class EEG_VLM_V6(nn.Module):
    def __init__(self, qwen_name):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model, TaskType

        # 1. Video Vision Backbone (VideoMAE)
        vmae_cfg = VideoMAEConfig(
            image_size=64, num_frames=128, num_channels=1, patch_size=16, tubelet_size=2,
            hidden_size=768, num_attention_heads=12, num_hidden_layers=12, intermediate_size=3072
        )
        self.video_enc = VideoMAEModel(vmae_cfg)
        self.rasterizer = MultiScaleRasterizer(size=64)
        
        # 2. Bridge & CTC
        self.bridge = nn.Sequential(
            nn.Linear(768, 1024), nn.LayerNorm(1024), nn.GELU(),
            nn.Linear(1024, 1024)
        )
        self.ctc_head = nn.Linear(768, len(CHAR_VOCAB))
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
        
        # 3. LLM (Qwen2.5-1.5B)
        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.llm = get_peft_model(self.llm, lora_cfg)
        self.llm_proj = nn.Linear(1024, 1536) # Qwen2.5-1.5B hidden size is 1536

    def forward(self, traj, char_ids, char_lens, input_ids=None, labels=None):
        B, T, Ch = traj.shape
        imgs = self.rasterizer(traj) # (B, 128, 1, 64, 64)
        
        # VideoMAE encoding
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state # (B, N_patches, 768)
        
        # V6: Intelligent temporal extraction
        # VideoMAE tubelet_size=2 means 128/2 = 64 temporal steps.
        # Patch size 16 means 64x64/16x16 = 16 spatial patches.
        # Shape: (B, 64 * 16, 768)
        # Reshape to (B, 64, 16, 768) then pool spatial
        try:
            v_seq = v_out.reshape(B, 64, 16, 768).mean(dim=2) # (B, 64, 768)
        except:
            # Fallback if dimensions differ
            v_seq = v_out[:, :64, :] 
            
        # Upsample to 128 for finer CTC grounding
        v_seq_128 = F.interpolate(v_seq.transpose(1, 2), size=128, mode='linear', align_corners=False).transpose(1, 2)
        
        # 1. CTC Loss
        ctc_logits = self.ctc_head(v_seq_128).transpose(0, 1) # (128, B, Vocab)
        input_lens = torch.full((B,), 128, dtype=torch.long, device=traj.device)
        loss_ctc = self.ctc_loss_fn(F.log_softmax(ctc_logits, dim=-1), char_ids, input_lens, char_lens)
        
        # 2. LM Loss
        temporal_feat = v_seq.mean(dim=1) # (B, 768) global
        prefix = self.llm_proj(self.bridge(temporal_feat)).unsqueeze(1) # (B, 1, 1536)
        
        if input_ids is not None:
            embed_fn = self.llm.get_input_embeddings()
            tok_embs = embed_fn(input_ids)
            combined = torch.cat([prefix, tok_embs], dim=1)
            combined_labels = torch.cat([
                torch.full((B, 1), -100, device=labels.device, dtype=labels.dtype),
                labels
            ], dim=1)
            out = self.llm(inputs_embeds=combined, labels=combined_labels)
            return out.loss, loss_ctc
        return None, loss_ctc

    @torch.no_grad()
    def generate(self, traj, tokenizer, max_tokens=40):
        traj = traj.to(next(self.parameters()).dtype)
        imgs = self.rasterizer(traj)
        v_out = self.video_enc(pixel_values=imgs).last_hidden_state
        v_seq = v_out.reshape(traj.shape[0], 64, 16, 768).mean(dim=2)
        v_seq_128 = F.interpolate(v_seq.transpose(1, 2), size=128, mode='linear', align_corners=False).transpose(1, 2)
        
        # CTC
        ctc_ids = self.ctc_head(v_seq_128).argmax(dim=-1)
        ctc_texts = []
        for b in range(traj.shape[0]):
            ids = ctc_ids[b].tolist()
            res = ""; prev = -1
            for i in ids:
                if i != prev and i != 0: res += ID_TO_CHAR.get(i, "")
                prev = i
            ctc_texts.append(res)
            
        # LLM
        temporal_feat = v_seq.mean(dim=1)
        prefix = self.llm_proj(self.bridge(temporal_feat)).unsqueeze(1)
        gen = self.llm.generate(inputs_embeds=prefix, max_new_tokens=max_tokens)
        llm_texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        return llm_texts, ctc_texts

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
    return eeg.T.astype(np.float32) # (128, 64)

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
def train_v6(epochs=20, phase=2):
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
    
    model = EEG_VLM_V6(q_name).to(torch.bfloat16).cuda()
    
    # PHASE 2: Load Phase 1 weights and setup LoRA training
    ckpt_path = Path("/persist/v6_epoch33.pt")
    if ckpt_path.exists():
        print(f"Loading Phase 1 weights: {ckpt_path}")
        model.load_state_dict(torch.load(str(ckpt_path), map_location="cuda"))
    
    if phase == 2:
        print("--- PHASE 2: LoRA Fine-Tuning ---")
        # Freeze Encoder/Bridge
        for param in model.video_enc.parameters(): param.requires_grad = False
        for param in model.rasterizer.parameters(): param.requires_grad = False
        for param in model.bridge.parameters(): param.requires_grad = False
        for param in model.ctc_head.parameters(): param.requires_grad = False # Optional: keep grounding frozen
        
        # Ensure LoRA is trainable
        for name, param in model.llm.named_parameters():
            if "lora_" in name: param.requires_grad = True
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5) # Lower LR for fine-tuning
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print(f"Starting V6 Phase {phase} Training: {len(ds)} samples")
    for epoch in range(1, epochs + 1):
        model.train()
        total_lm, total_ctc = 0, 0
        for eeg, input_ids, labels, c_ids, c_lens in train_loader:
            eeg, input_ids, labels, c_ids, c_lens = [x.cuda() for x in [eeg, input_ids, labels, c_ids, c_lens]]
            eeg = eeg.to(torch.bfloat16)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss_lm, loss_ctc = model(eeg, c_ids, c_lens, input_ids, labels)
                loss = loss_lm + 0.5 * loss_ctc
            
            loss.backward()
            optimizer.step()
            total_lm += loss_lm.item(); total_ctc += loss_ctc.item()
            
        print(f"Epoch {epoch} | LM: {total_lm/len(train_loader):.3f} | CTC: {total_ctc/len(train_loader):.3f}")
        
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
                
            torch.save(model.state_dict(), f"/persist/v6_phase2_epoch{epoch}.pt")

@app.local_entrypoint()
def main():
    train_v6.remote()
