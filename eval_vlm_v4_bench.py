
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

app = modal.App("eeg-vlm-v4-1-final-bench")

ckpt_vol = modal.Volume.from_name("bt-checkpoints", create_if_missing=True)
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.3.0", "transformers>=4.40", "peft>=0.11", 
        "numpy", "scipy", "jiwer", "rouge-score", "einops",
        "sentence-transformers", "bitsandbytes", "accelerate"
    ])
)

class CrossAttentionBridge(nn.Module):
    def __init__(self, d_model, n_heads=16, n_queries=64):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model))
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x):
        B = x.shape[0]; q = self.queries.expand(B, -1, -1)
        out, _ = self.mha(q, x, x); return self.ln(out + q)

class TopographicRasterizerV4(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()
        self.img_size = img_size
        angles = torch.linspace(0, 2*np.pi, 64)
        pos_2d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        gx, gy = torch.meshgrid(torch.linspace(-1, 1, img_size), torch.linspace(-1, 1, img_size), indexing='ij')
        px, py = gx.flatten().unsqueeze(1), gy.flatten().unsqueeze(1)
        ex, ey = pos_2d[:, 0].unsqueeze(0), pos_2d[:, 1].unsqueeze(0)
        dist = torch.sqrt((px - ex)**2 + (py - ey)**2)
        w = 1.0 / (dist + 1e-4) ** 2.0
        r = torch.sqrt(px**2 + py**2); w[(r > 1.1).squeeze(1), :] = 0.0
        w = w / (w.sum(dim=1, keepdim=True).clamp(min=1e-8))
        self.register_buffer("W", w)
    def forward(self, x):
        B, T, Ch = x.shape
        imgs = (x.view(B*T, Ch) @ self.W.T.to(x.dtype)).view(B, T, 1, self.img_size, self.img_size)
        return imgs

@app.function(image=image, gpu="H100", volumes={"/data": data_vol, "/persist": ckpt_vol}, timeout=3600)
def run_v4_1_final_bench(ckpt_epoch=3):
    import jiwer
    from transformers import AutoTokenizer, AutoModelForCausalLM, VideoMAEConfig, VideoMAEModel
    from peft import get_peft_model, LoraConfig, TaskType
    
    device = torch.device("cuda")
    local_path = f"/persist/eeg_vlm_v4_1_epoch{ckpt_epoch}.pt"
    local_data = "/data/traj_cache_mindvoice.npz"
    
    # 1. Chunked Streaming for 15GB stability
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1024**3:
        print(f"Streaming Epoch {ckpt_epoch} (15GB)...")
        old_client = modal.Client.from_credentials("ak-zbkRnV4DZtd1H3fSkjQKtQ", "as-JpYfpc1gA08h3lP55N4aZ3")
        
        if not os.path.exists(local_data):
            old_vol = modal.Volume.from_name("mindvoice-data", client=old_client)
            with open(local_data, "wb") as f:
                for chunk in old_vol.read_file("/traj_cache_mindvoice.npz"): f.write(chunk)
            print("Data Streamed.")

        old_vol = modal.Volume.from_name("bt-checkpoints", client=old_client)
        with open(local_path + ".tmp", "wb") as f:
            for chunk in old_vol.read_file(f"/eeg_vlm_v4_1_epoch{ckpt_epoch}.pt"): 
                f.write(chunk)
        os.rename(local_path + ".tmp", local_path)
        print(f"Epoch {ckpt_epoch} Sync Complete ({os.path.getsize(local_path)/(1024**3):.1f} GB).")
        ckpt_vol.commit()

    mv = np.load(local_data, allow_pickle=True)
    mv_texts = [str(t).lower().strip() for t in mv["texts"]]
    mv_subjects = [f"mv_{s}" for s in mv["subjects"]]
    val_idx = [i for i, s in enumerate(mv_subjects) if "sub15" in s]
    
    qwen_name = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(qwen_name)
    vmae_cfg = VideoMAEConfig(
        image_size=64, num_frames=128, num_channels=3, patch_size=16, tubelet_size=2, 
        hidden_size=1024, num_attention_heads=16, num_hidden_layers=24, intermediate_size=4096
    )

    class EEGVideoVLM_V4_1_Eval(nn.Module):
        def __init__(self):
            super().__init__()
            self.rasterizer = TopographicRasterizerV4(); self.ch_expand = nn.Conv2d(1, 3, 1)
            self.video_enc = VideoMAEModel(vmae_cfg); self.bridge = CrossAttentionBridge(1024, n_heads=16)
            self.llm_proj = nn.Linear(1024, 3584)
            base = AutoModelForCausalLM.from_pretrained(qwen_name, torch_dtype=torch.bfloat16, device_map="auto")
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, 
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            self.llm = get_peft_model(base, lora_cfg)

        def generate(self, traj, tokenizer):
            B = traj.shape[0]
            imgs = self.rasterizer(traj); _,T,_,H,W = imgs.shape
            imgs_3ch = F.gelu(self.ch_expand(imgs.view(B*T,1,H,W)))
            v_in = imgs_3ch.view(B,T,3,H,W).contiguous()
            enc = self.video_enc(pixel_values=v_in).last_hidden_state
            pooled = self.bridge(enc); prefix = self.llm_proj(pooled).to(self.llm.device)
            # NO BOS (Matches Training)
            anchor = tokenizer("[EEG: ]", return_tensors="pt", add_special_tokens=False).input_ids.to(self.llm.device)
            try: embed_fn = self.llm.model.model.embed_tokens
            except: embed_fn = self.llm.base_model.model.model.embed_tokens
            anchor_emb = embed_fn(anchor).expand(B, -1, -1)
            combined = torch.cat([prefix, anchor_emb], dim=1)
            gen = self.llm.generate(inputs_embeds=combined, max_new_tokens=10, num_beams=3, early_stopping=True)
            return tokenizer.batch_decode(gen, skip_special_tokens=True)

    model = EEGVideoVLM_V4_1_Eval().to(device).to(torch.bfloat16)
    print("Loading Weights...")
    ckpt = torch.load(local_path, map_location='cpu')
    model.video_enc.load_state_dict(ckpt["video_enc"])
    model.bridge.load_state_dict(ckpt["bridge"])
    model.llm_proj.load_state_dict(ckpt["llm_proj"])
    model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    model.eval()

    # Evaluation Pass
    results = []
    for idx in val_idx[:15]:
        traj = torch.from_numpy(mv["trajectories"][idx]).unsqueeze(0).to(device, dtype=torch.bfloat16)
        ground_truth = mv_texts[idx]
        raw_pred = model.generate(traj, tokenizer)[0]
        pred = raw_pred.lower().strip()
        if "eeg: ]" in pred: pred = pred.split("eeg: ]")[-1].strip()
        results.append({"wer": jiwer.wer(ground_truth, pred), "exact": 1.0 if pred == ground_truth else 0.0})
        print(f"RAW: '{raw_pred}' | Pred: '{pred}' | GT: '{ground_truth}'")

    accuracy = np.mean([r["exact"] for r in results])
    print(f"\n======================\nFINAL V4.1 EPOCH {ckpt_epoch} | ACCURACY: {accuracy:.4f}\n======================")
    return {"accuracy": float(accuracy)}

@app.local_entrypoint()
def main(epoch: int = 3):
    run_v4_1_final_bench.remote(ckpt_epoch=epoch)
