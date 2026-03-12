
import modal
import torch
import os

image = modal.Image.debian_slim().pip_install(["torch"])
app = modal.App("diag-v2-ckpts")
ckpt_vol = modal.Volume.from_name("bt-checkpoints")

@app.function(image=image, volumes={"/persist": ckpt_vol})
def diag_ckpts():
    files = os.listdir("/persist")
    results = []
    for f in files:
        if "eeg_vlm_v2" in f:
            p = f"/persist/{f}"
            size = os.path.getsize(p)
            status = "Unknown"
            try:
                # Try to check if it's a valid torch file (metadata only)
                with open(p, 'rb') as f_obj:
                    # Just check first few bytes for 'PK' (zip header)
                    header = f_obj.read(2)
                    if header == b'PK':
                        status = "Valid ZIP Header"
                    else:
                        status = f"Invalid Header: {header}"
            except Exception as e:
                status = f"Error reading: {e}"
            results.append(f"{f}: {size/1024/1024:.2f} MB | {status}")
    return "\n".join(results)

@app.local_entrypoint()
def main():
    print(diag_ckpts.remote())
