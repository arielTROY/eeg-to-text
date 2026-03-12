
import modal
import os
import subprocess

app = modal.App("diag-volume")
data_vol = modal.Volume.from_name("mindvoice-data")

@app.function(volumes={"/data": data_vol})
def diag():
    print("Listing /data/EEG_Text:")
    subprocess.run(["ls", "-R", "/data/EEG_Text"])
    
    zuco_files = []
    for root, dirs, files in os.walk("/data/EEG_Text"):
        for f in files:
            if f.endswith(".mat"):
                p = os.path.join(root, f)
                zuco_files.append(p)
    
    for p in zuco_files:
        size = os.path.getsize(p)
        print(f"File: {p} | Size: {size} bytes")
        if size < 1000:
            print(f"  [Error] File too small. Content:")
            subprocess.run(["cat", p])
        else:
            print(f"  Head of file:")
            subprocess.run(["head", "-c", "128", p])
            print("\n")

@app.local_entrypoint()
def main():
    diag.remote()
