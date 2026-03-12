
import modal
import torch

app = modal.App("diag-weights")
volume = modal.Volume.from_name("bt-checkpoints")

@app.function(image=modal.Image.debian_slim().pip_install(["torch"]), volumes={"/persist": volume})
def check_ckpt(ckpt_epoch=3):
    ckpt_path = f"/persist/eeg_vlm_v4_1_epoch{ckpt_epoch}.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"--- Diagnostic for Epoch {ckpt_epoch} ---")
    
    for k in ["bridge", "llm_proj"]:
        if k in ckpt:
            first_key = list(ckpt[k].keys())[0]
            val = ckpt[k][first_key]
            print(f"{k} [{first_key}] - Mean: {val.mean().item():.6f}, Std: {val.std().item():.6f}")
        else:
            print(f"KEY {k} MISSING!")

@app.local_entrypoint()
def main(epoch: int = 3):
    check_ckpt.remote(ckpt_epoch=epoch)
