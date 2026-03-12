
import torch
import modal

def check_ckpt(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print("Keys in Checkpoint:", ckpt.keys())
    print("LLM LoRA Keys (First 10):", list(ckpt["llm_lora"].keys())[:10])
    
    # Check for NaNs
    for k, v in ckpt.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                if torch.isnan(subv).any():
                    print(f"NAN FOUND IN {k}.{subk}")
        elif torch.is_tensor(v) and torch.isnan(v).any():
            print(f"NAN FOUND IN {k}")

if __name__ == "__main__":
    check_ckpt("/persist/eeg_vlm_v4_1_epoch3.pt")
