import modal
import h5py
import numpy as np

app = modal.App("diag-zuco-v2")
data_vol = modal.Volume.from_name("mindvoice-data")

@app.function(volumes={"/data": data_vol})
def diag():
    path = "/data/EEG_Text/ZuCo_v2/resultsYAC_NR.mat"
    print(f"Inspecting {path}...")
    try:
        with h5py.File(path, 'r') as f:
            print(f"Root keys: {list(f.keys())}")
            if 'results' in f:
                res = f['results']
                print(f"Results keys: {list(res.keys())}")
                if 'sentenceData' in res:
                    sd = res['sentenceData']
                    print(f"sentenceData type: {type(sd)}, shape: {getattr(sd, 'shape', 'N/A')}")
                    # Look at first item
                    first_ref = sd[0][0]
                    obj = f[first_ref]
                    print(f"First sentence keys: {list(obj.keys())}")
                    
                    if 'content' in obj:
                        c_ref = obj['content'][0][0]
                        c_data = f[c_ref]
                        print(f"Content data shape: {c_data.shape}")
                        text = "".join([chr(c[0]) for c in c_data])
                        print(f"Text: {text[:50]}...")
                        
                    for k in ['rawData', 'mean_EEG', 'processedEEG']:
                        if k in obj:
                            e_ref = obj[k][0][0]
                            e_data = f[e_ref]
                            print(f"{k} shape: {e_data.shape}")
                            break
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    with app.run():
        diag.remote()
