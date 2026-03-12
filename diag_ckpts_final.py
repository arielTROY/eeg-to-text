
import modal
import torch
import os

image = modal.Image.debian_slim().pip_install(["torch"])
app = modal.App("diag-v2-ckpts-new")
ckpt_vol = modal.Volume.from_name("bt-checkpoints")

@app.function(image=image, volumes={"/persist": ckpt_vol})
def diag_ckpts():
    files = os.listdir("/persist")
    results = []
    for f in sorted(files):
        if "eeg_vlm_v2" in f and f.endswith(".pt"):
            p = f"/persist/{f}"
            size = os.path.getsize(p)
            try:
                ckpt = torch.load(p, map_location="cpu")
                epoch = ckpt.get("epoch", "N/A")
                val = ckpt.get("val", "N/A")
                results.append(f"{f}: {size/1024/1024:.2f} MB | EPOCH: {epoch} | VAL: {val} | STATUS: OK")
            except Exception as e:
                # Try simple open for head/tail check
                with open(p, 'rb') as fo:
                    header = fo.read(4)
                    fo.seek(-4, 2)
                    footer = fo.read(4)
                results.append(f"{f}: {size/1024/1024:.2f} MB | STATUS: ERROR ({e}) | H: {header} | F: {footer}")
    return "\n".join(results)

@app.local_entrypoint()
def main():
    print(diag_ckpts.remote())
