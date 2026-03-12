
import modal
import torch

image = modal.Image.debian_slim().pip_install(["torch"])
app = modal.App("check-v2-ckpt-metadata")
ckpt_vol = modal.Volume.from_name("bt-checkpoints")

@app.function(image=image, volumes={"/persist": ckpt_vol})
def check_ckpt():
    import os
    results = []
    for f in ["eeg_vlm_v2_best.pt", "eeg_vlm_v2_last.pt"]:
        p = f"/persist/{f}"
        if os.path.exists(p):
            ckpt = torch.load(p, map_location="cpu")
            results.append(f"{f}: Epoch {ckpt.get('epoch', 'N/A')} | Val: {ckpt.get('val', 'N/A')}")
        else:
            results.append(f"{f}: Not found")
    return "\n".join(results)

@app.local_entrypoint()
def main():
    print(check_ckpt.remote())
