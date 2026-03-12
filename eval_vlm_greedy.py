
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoTokenizer, VideoMAEConfig, VideoMAEModel, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# Image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.3.0", "transformers>=4.40", "peft>=0.11", 
        "numpy", "scipy", "jiwer", "rouge-score", "einops",
        "sentence-transformers"
    ])
)

app = modal.App("eeg-vlm-v2-greedy-eval")
ckpt_vol = modal.Volume.from_name("bt-checkpoints")
data_vol = modal.Volume.from_name("mindvoice-data")

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

class SubjectClassifier(nn.Module):
    def __init__(self, input_dim, num_subjects):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_subjects) # This was the mismatch: Dropout(0.1) adds a layer index
        )
    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha)
        return self.net(x)

@app.function(image=image, gpu="H100", volumes={"/data": data_vol, "/persist": ckpt_vol}, timeout=3600)
def run_greedy_eval():
    device = torch.device("cuda")
    
    # 1. Load Data to determine num_subjects
    mv = np.load("/data/traj_cache_mindvoice.npz", allow_pickle=True)
    bl = np.load("/data/traj_cache.npz", allow_pickle=True) # Assuming baseline is here
    
    subjects = [f"mv_{s}" for s in mv["subjects"]] + [f"bl_{s}" for s in bl["subjects"]]
    num_subjects = len(set(subjects))
    
    trajectories = mv["trajectories"].astype(np.float32)
    texts = [str(t) for t in mv["texts"]]
    mv_subjects = [f"mv_{s}" for s in mv["subjects"]]
    
    # 2. Checkpoint path
    ckpt_path = "/persist/eeg_vlm_v2_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/persist/eeg_vlm_v2_last.pt"
        if not os.path.exists(ckpt_path):
            return f"No checkpoint found at /persist/eeg_vlm_v2_best.pt or last.pt"
    
    print(f"Loading checkpoint from {ckpt_path}... Subjects: {num_subjects}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # 3. Model Definition
    qwen_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(qwen_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    class TopographicRasterizer(nn.Module):
        def __init__(self, img_size=32):
            super().__init__()
            self.img_size = img_size
            angles = torch.linspace(0, 2*np.pi, 64)
            pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1).to(device)
            gx, gy = torch.meshgrid(
                torch.linspace(-1, 1, img_size, device=device), 
                torch.linspace(-1, 1, img_size, device=device), 
                indexing='ij'
            )
            px, py = gx.flatten().unsqueeze(1), gy.flatten().unsqueeze(1)
            ex, ey = pos_2d[:, 0].unsqueeze(0), pos_2d[:, 1].unsqueeze(0)
            dist = torch.sqrt((px - ex)**2 + (py - ey)**2)
            w = 1.0 / (dist + 1e-4) ** 2.0
            r = torch.sqrt(px**2 + py**2)
            w[(r > 1.05).squeeze(1), :] = 0.0
            w = w / (w.sum(dim=1, keepdim=True).clamp(min=1e-8))
            self.register_buffer("W", w)
        def forward(self, x):
            B, T, Ch = x.shape
            imgs = (x.view(B*T, Ch) @ self.W.T.to(x.dtype)).view(B, T, 1, self.img_size, self.img_size)
            return imgs

    vmae_cfg = VideoMAEConfig(
        image_size=32, num_frames=128, num_channels=3, 
        patch_size=8, tubelet_size=2, hidden_size=768, num_hidden_layers=12
    )

    class EEGVideoVLM_V2(nn.Module):
        def __init__(self, num_subjects):
            super().__init__()
            self.rasterizer = TopographicRasterizer()
            self.ch_expand = nn.Conv2d(1, 3, 1)
            self.video_enc = VideoMAEModel(vmae_cfg)
            self.connector = nn.Sequential(
                nn.LayerNorm(768), nn.Linear(768, 1536), nn.GELU(), nn.Dropout(0.15), nn.Linear(1536, 1536), nn.LayerNorm(1536)
            )
            self.token_pool = nn.AdaptiveAvgPool1d(64)
            self.subject_classifier = SubjectClassifier(768, num_subjects)
            self.grounding_proj = nn.Linear(1536, 768)
            
            llm_base = AutoModelForCausalLM.from_pretrained(qwen_name, torch_dtype=torch.bfloat16)
            lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            self.llm = get_peft_model(llm_base, lora_cfg)

        def encode_eeg(self, traj):
            imgs = self.rasterizer(traj)
            B, T, _, H, W = imgs.shape
            imgs_3ch = F.gelu(self.ch_expand(imgs.view(B*T, 1, H, W)))
            enc = self.video_enc(pixel_values=imgs_3ch.view(B, T, 3, H, W)).last_hidden_state
            feats = self.connector(enc)
            prefix = self.token_pool(feats.transpose(1, 2)).transpose(1, 2)
            return prefix

        def generate_greedy(self, traj):
            prefix = self.encode_eeg(traj)
            gen = self.llm.generate(
                inputs_embeds=prefix,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            return tokenizer.batch_decode(gen, skip_special_tokens=True)

    model = EEGVideoVLM_V2(num_subjects).to(device).to(torch.bfloat16)
    
    # Load weights
    model.video_enc.load_state_dict(ckpt["video_enc"])
    model.connector.load_state_dict(ckpt["connector"])
    model.ch_expand.load_state_dict(ckpt["ch_expand"])
    model.subject_classifier.load_state_dict(ckpt["sub_clf"])
    model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    model.eval()
    
    # 4. Evaluate Held-out Subject
    held_out_sub = "mv_mindvoice_sub15"
    val_idx = [i for i, s in enumerate(mv_subjects) if s == held_out_sub]
    
    print(f"Evaluating {len(val_idx)} samples from {held_out_sub} in Greedy Mode (Temp 0)...")
    
    results = []
    for i in range(0, min(128, len(val_idx)), 8):
        batch = val_idx[i:i+8]
        t_traj = torch.from_numpy(trajectories[batch]).to(device, dtype=torch.bfloat16)
        batch_texts = [texts[idx] for idx in batch]
        
        with torch.no_grad():
            preds = model.generate_greedy(t_traj)
            
        for ref, pred in zip(batch_texts, preds):
            line = f"REF: {ref.strip()} | PRED: {pred.strip()}"
            results.append(line)
            print(line)
            
    return results

@app.local_entrypoint()
def main():
    res = run_greedy_eval.remote()
    print("\nFinal Greedy Results:")
    for line in res:
        print(line)
