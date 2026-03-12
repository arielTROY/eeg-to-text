
import modal
import torch
from pathlib import Path

app = modal.App("eeg-vlm-eval")
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "torch>=2.3.0", "transformers>=4.40", "peft>=0.11", 
    "numpy", "scipy", "jiwer", "rouge-score", "einops"
])

ckpt_vol = modal.Volume.from_name("bt-checkpoints")
data_vol = modal.Volume.from_name("mindvoice-data")

@app.function(image=image, gpu="H100", volumes={"/data": data_vol, "/persist": ckpt_vol}, timeout=3600)
def run_eval():
    import json, sys, math
    import numpy as np
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, VideoMAEConfig, VideoMAEModel, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    
    # 1. Setup Model (Mirroring training script)
    device = torch.device("cuda")
    ckpt_path = Path("/persist/eeg_video_vlm_mindvoice.pt")
    if not ckpt_path.exists():
        return "Checkpoint not found"
        
    ckpt = torch.load(ckpt_path, map_location="cpu")
    history = ckpt.get("history", [])
    best_epoch = ckpt.get("best_epoch", "unknown")
    print(f"Loading best checkpoint from epoch {best_epoch}")

    # Data info
    raw = np.load("/data/traj_cache_mindvoice.npz", allow_pickle=True)
    trajectories = raw["trajectories"].astype(np.float32)
    texts = [str(t) for t in raw["texts"]]
    subjects = [str(s) for s in raw["subjects"]]
    POSITIONS_2D = ckpt["positions_2d"].to(device)
    
    # Architecture
    qwen_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(qwen_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    llm_base = AutoModelForCausalLM.from_pretrained(qwen_name, torch_dtype=torch.bfloat16)
    lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=64, target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    llm = get_peft_model(llm_base, lora_cfg)
    
    # Rasterizer (copied from main script for stand-alone run)
    class TopographicRasterizer(torch.nn.Module):
        def __init__(self, pos_2d, img_size=32):
            super().__init__()
            self.img_size = img_size
            gx, gy = torch.meshgrid(
                torch.linspace(-1, 1, img_size, device=pos_2d.device), 
                torch.linspace(-1, 1, img_size, device=pos_2d.device), 
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
            imgs = (x.view(B*T, Ch) @ self.W.T).view(B, T, 1, self.img_size, self.img_size)
            return imgs

    # The VLM
    vmae_cfg = VideoMAEConfig(image_size=32, num_frames=128, num_channels=3, patch_size=8, tubelet_size=2, hidden_size=768, num_hidden_layers=8)
    
    class EEGVideoVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rasterizer = TopographicRasterizer(POSITIONS_2D)
            self.ch_expand = torch.nn.Conv2d(1, 3, 1)
            self.video_enc = VideoMAEModel(vmae_cfg)
            self.connector = torch.nn.Sequential(torch.nn.LayerNorm(768), torch.nn.Linear(768, 1536), torch.nn.GELU(), torch.nn.Linear(1536, 1536))
            self.token_pool = torch.nn.AdaptiveAvgPool1d(64)
            self.llm = llm
        def encode_eeg(self, traj):
            imgs = self.rasterizer(traj)
            B, T, _, H, W = imgs.shape
            imgs_3ch = torch.nn.functional.gelu(self.ch_expand(imgs.view(B*T, 1, H, W)))
            v_in = imgs_3ch.view(B, T, 3, H, W).contiguous()
            enc = self.video_enc(pixel_values=v_in).last_hidden_state
            feats = self.connector(enc)
            return self.token_pool(feats.transpose(1, 2)).transpose(1, 2)
        def generate(self, traj, max_len=32):
            prefix = self.encode_eeg(traj)
            gen = self.llm.generate(inputs_embeds=prefix, attention_mask=torch.ones(prefix.shape[0], 64, device=prefix.device), max_new_tokens=max_len, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            return tokenizer.batch_decode(gen, skip_special_tokens=True)

    model = EEGVideoVLM().to(device)
    model.video_enc.load_state_dict(ckpt["video_enc"])
    model.connector.load_state_dict(ckpt["connector"])
    model.ch_expand.load_state_dict(ckpt["ch_expand"])
    model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    model.eval()

    # 2. Run Benchmark
    held_out = "mindvoice_sub15"
    test_idx = [i for i, s in enumerate(subjects) if s == held_out]
    print(f"Benchmarking on {held_out} ({len(test_idx)} trials)")
    
    hypotheses, references = [], []
    for i in range(0, len(test_idx), 8):
        batch_idx = test_idx[i:i+8]
        traj = torch.from_numpy(trajectories[batch_idx]).to(device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model.generate(traj)
        hypotheses.extend([p.strip() for p in preds])
        references.extend([texts[idx].strip() for idx in batch_idx])

    # 3. Compute Metrics
    from jiwer import wer
    from rouge_score import rouge_scorer
    
    wer_val = wer(references, hypotheses)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = np.mean([scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)])
    
    print(f"WER: {wer_val:.4f} | ROUGE-L: {rouge_l:.4f}")
    for r, h in zip(references[:10], hypotheses[:10]):
        print(f"REF: {r!r} | PRED: {h!r}")
        
    return {"wer": wer_val, "rouge_l": rouge_l, "samples": hypotheses[:20]}

@app.local_entrypoint()
def main():
    res = run_eval.remote()
    print(f"\nFinal Eval Metrics: {res}")
