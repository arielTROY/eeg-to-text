
import modal
import torch
import os

image = modal.Image.debian_slim().pip_install(["torch"])
app = modal.App("scan-all-ckpts")
ckpt_vol = modal.Volume.from_name("bt-checkpoints")

@app.function(image=image, volumes={"/persist": ckpt_vol})
def scan_ckpts():
    all_files = []
    for root, dirs, files in os.walk("/persist"):
        for f in files:
            if f.endswith(".pt"):
                all_files.append(os.path.join(root, f))
    
    results = []
    for p in all_files:
        size = os.path.getsize(p)
        try:
            # Try to load just the metadata first
            ckpt = torch.load(p, map_location="cpu")
            epoch = ckpt.get("epoch", "N/A")
            val = ckpt.get("val", "N/A")
            results.append(f"{p}: {size/1024/1024:.2f} MB | EPOCH: {epoch} | VAL: {val} | OK")
        except Exception as e:
            results.append(f"{p}: {size/1024/1024:.2f} MB | FAIL: {str(e)[:100]}")
            
    return "\n".join(results)

@app.local_entrypoint()
def main():
    print(scan_ckpts.remote())
