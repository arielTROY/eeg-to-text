
import modal
import torch

app = modal.App("diag-keys")
volume = modal.Volume.from_name("bt-checkpoints")

@app.function(image=modal.Image.debian_slim().pip_install(["torch"]), volumes={"/persist": volume})
def list_keys(ckpt_epoch=3):
    ckpt_path = f"/persist/eeg_vlm_v4_1_epoch{ckpt_epoch}.pt"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"--- Keys in Epoch {ckpt_epoch} ---")
    print("Main Keys:", ckpt.keys())
    for k in ckpt.keys():
        if isinstance(ckpt[k], dict):
            print(f"Sub-keys for {k}: {len(ckpt[k])} keys found.")
            if len(ckpt[k]) < 50:
                print(f"List {k}:", list(ckpt[k].keys()))
            else:
                print(f"List {k} (First 10):", list(ckpt[k].keys())[:10])
        elif torch.is_tensor(ckpt[k]):
            print(f"Tensor {k}: shape {ckpt[k].shape}")

@app.local_entrypoint()
def main(epoch: int = 3):
    list_keys.remote(ckpt_epoch=epoch)
