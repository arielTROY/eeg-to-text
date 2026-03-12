
import modal
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

app = modal.App("eeg-vlm-v3-baseline")

ckpt_vol = modal.Volume.from_name("bt-checkpoints")
data_vol = modal.Volume.from_name("mindvoice-data")

image = modal.Image.debian_slim().pip_install(["torch", "transformers", "peft", "numpy", "jiwer"])

# V3 Architecture (Smaller LoRA, likely no Proj if older)
class CrossAttentionBridge(nn.Module):
    def __init__(self, d_model, n_heads=8, n_queries=64):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model))
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x):
        B = x.shape[0]; q = self.queries.expand(B, -1, -1)
        out, _ = self.mha(q, x, x); return self.ln(out + q)

@app.function(image=image, gpu="H100", volumes={"/data": data_vol, "/persist": ckpt_vol})
def run_v3_bench():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType
    import jiwer
    
    device = torch.device("cuda")
    local_path = "/persist/eeg_vlm_v3_final.pt"
    local_data = "/data/traj_cache_mindvoice.npz"
    
    if not os.path.exists(local_path):
        print("Streaming V3...")
        old_client = modal.Client.from_credentials("ak-zbkRnV4DZtd1H3fSkjQKtQ", "as-JpYfpc1gA08h3lP55N4aZ3")
        old_vol = modal.Volume.from_name("bt-checkpoints", client=old_client)
        with open(local_path, "wb") as f:
            for chunk in old_vol.read_file("/eeg_vlm_v3_final.pt"): f.write(chunk)
        print("V3 Streamed.")
        ckpt_vol.commit()

    mv = np.load(local_data, allow_pickle=True)
    val_idx = [i for i, s in enumerate(mv["subjects"]) if "sub15" in s]
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    
    # Estimate V3 Structure (Based on 3.1GB size)
    # V3 usually used 7B models too.
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", torch_dtype=torch.bfloat16, device_map="auto")
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
    llm = get_peft_model(base, lora_cfg)
    
    ckpt = torch.load(local_path, map_location='cpu')
    print("V3 Checkpoint Keys:", ckpt.keys())
    
    # V3 Logic: Just prefix + prompt
    def generate(traj):
        # Dummy rasterizer/prefix for V3 (Need to match V3 exact logic)
        # Assuming V3 used the same TopographicRasterizer
        return "v3_stub"

    print("V3 Verification Pass... (This is a structural check first)")
    return {"keys": list(ckpt.keys())}
