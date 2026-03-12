"""Diagnostic: inspect ZuCo volume data and debug .mat parsing."""
import modal

app = modal.App("diag-zuco-data")
data_vol = modal.Volume.from_name("mindvoice-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["scipy", "h5py", "numpy"])
)

@app.function(image=image, volumes={"/data": data_vol}, timeout=600)
def diagnose():
    import os, glob
    from pathlib import Path
    import numpy as np

    base = Path("/data/EEG_Text")
    print("=" * 60)
    print("DIRECTORY LISTING")
    print("=" * 60)

    # List what exists
    for root, dirs, files in os.walk("/data"):
        level = root.replace("/data", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 4:
            for f in files[:20]:
                fpath = os.path.join(root, f)
                size = os.path.getsize(fpath)
                print(f"{indent}  {f}  ({size / 1024 / 1024:.1f} MB)")
            if len(files) > 20:
                print(f"{indent}  ... and {len(files) - 20} more files")

    print("\n" + "=" * 60)
    print("MAT FILE ANALYSIS")
    print("=" * 60)

    mat_files = list(Path("/data").rglob("*.mat"))
    print(f"\nFound {len(mat_files)} .mat files total")
    for mf in mat_files:
        print(f"\n--- {mf} ({mf.stat().st_size / 1024 / 1024:.1f} MB) ---")

        # Try scipy first
        try:
            import scipy.io as sio
            mat = sio.loadmat(str(mf), squeeze_me=True, struct_as_record=False)
            print(f"  [scipy] Top-level keys: {[k for k in mat.keys() if not k.startswith('__')]}")
            for key in [k for k in mat.keys() if not k.startswith('__')]:
                obj = mat[key]
                print(f"    '{key}': type={type(obj).__name__}", end="")
                if hasattr(obj, '__len__'):
                    print(f", len={len(obj)}", end="")
                if hasattr(obj, 'dtype'):
                    print(f", dtype={obj.dtype}", end="")
                print()

                # Inspect sub-attributes
                if hasattr(obj, '_fieldnames'):
                    print(f"      fieldnames: {obj._fieldnames}")
                    for fn in obj._fieldnames:
                        sub = getattr(obj, fn, None)
                        if sub is not None:
                            print(f"      .{fn}: type={type(sub).__name__}", end="")
                            if isinstance(sub, np.ndarray):
                                print(f", shape={sub.shape}, dtype={sub.dtype}", end="")
                                # If it's an array of objects, inspect first
                                if sub.dtype == object and len(sub) > 0:
                                    first = sub[0] if sub.ndim == 1 else sub.flat[0]
                                    print(f"\n        first element type={type(first).__name__}", end="")
                                    if hasattr(first, '_fieldnames'):
                                        print(f", fieldnames={first._fieldnames}", end="")
                                        for ffn in first._fieldnames[:5]:
                                            fval = getattr(first, ffn, None)
                                            if fval is not None:
                                                print(f"\n          .{ffn}: type={type(fval).__name__}", end="")
                                                if isinstance(fval, np.ndarray):
                                                    print(f", shape={fval.shape}", end="")
                                                elif isinstance(fval, str):
                                                    print(f", val='{fval[:80]}'", end="")
                            print()
                elif isinstance(obj, np.ndarray) and obj.dtype == object:
                    # Array of objects
                    if len(obj) > 0:
                        first = obj.flat[0]
                        print(f"      first element type={type(first).__name__}")
                        if hasattr(first, '_fieldnames'):
                            print(f"      first._fieldnames: {first._fieldnames}")
            continue  # Don't try h5py if scipy worked
        except NotImplementedError:
            print(f"  [scipy] v7.3+ HDF5 format, trying h5py...")
        except Exception as e:
            print(f"  [scipy] Error: {e}")

        # Try h5py
        try:
            import h5py
            with h5py.File(str(mf), 'r') as f:
                def show_group(g, prefix=""):
                    for k in list(g.keys())[:30]:
                        item = g[k]
                        if isinstance(item, h5py.Group):
                            print(f"  [h5py] {prefix}{k}/ (Group, {len(item)} items)")
                            if prefix.count('/') < 2:
                                show_group(item, prefix + k + "/")
                        elif isinstance(item, h5py.Dataset):
                            print(f"  [h5py] {prefix}{k}: shape={item.shape}, dtype={item.dtype}")
                            # If it's references, try resolving first one
                            if item.dtype == h5py.ref_dtype and len(item) > 0:
                                try:
                                    ref = item[0] if item.ndim == 1 else item[0, 0]
                                    resolved = f[ref]
                                    print(f"    -> resolves to: type={type(resolved).__name__}", end="")
                                    if isinstance(resolved, h5py.Group):
                                        print(f", keys={list(resolved.keys())[:10]}")
                                    elif isinstance(resolved, h5py.Dataset):
                                        print(f", shape={resolved.shape}, dtype={resolved.dtype}")
                                    else:
                                        print()
                                except Exception as e2:
                                    print(f"    -> resolve error: {e2}")
                show_group(f)
        except Exception as e:
            print(f"  [h5py] Error: {e}")

    # Now test actual parsing
    print("\n" + "=" * 60)
    print("PARSE TEST")
    print("=" * 60)
    for mf in mat_files[:3]:
        print(f"\nParsing: {mf}")
        count = 0
        try:
            import scipy.io as sio
            mat = sio.loadmat(str(mf), squeeze_me=True, struct_as_record=False)
            keys = [k for k in mat.keys() if not k.startswith('__')]
            print(f"  scipy keys: {keys}")

            # Try all paths to find sentence data
            for root_key in keys:
                root = mat[root_key]
                print(f"  Checking '{root_key}'...")

                # Direct array of structs
                if isinstance(root, np.ndarray) and root.dtype == object:
                    for item in root.flat[:3]:
                        if hasattr(item, '_fieldnames'):
                            print(f"    item fieldnames: {item._fieldnames}")
                            content = getattr(item, 'content', None)
                            print(f"    content={repr(content)[:100] if content is not None else 'None'}")
                            for ek in ['mean_EEG', 'processedEEG', 'RawEEG', 'rawEEG']:
                                eeg = getattr(item, ek, None)
                                if eeg is not None:
                                    print(f"    {ek}={type(eeg).__name__}, shape={np.array(eeg).shape if hasattr(eeg, 'shape') or isinstance(eeg, np.ndarray) else 'N/A'}")
                            break

                # Struct with sub-fields
                if hasattr(root, '_fieldnames'):
                    print(f"    struct fields: {root._fieldnames}")
                    for sub_key in root._fieldnames:
                        sub = getattr(root, sub_key, None)
                        if sub is not None:
                            print(f"    .{sub_key}: type={type(sub).__name__}", end="")
                            if isinstance(sub, np.ndarray):
                                print(f", shape={sub.shape}, dtype={sub.dtype}", end="")
                                if sub.dtype == object and len(sub) > 0:
                                    first = sub.flat[0]
                                    if hasattr(first, '_fieldnames'):
                                        print(f"\n      first item fieldnames: {first._fieldnames}", end="")
                            print()
        except Exception as e:
            print(f"  scipy error: {e}")

@app.local_entrypoint()
def main():
    diagnose.remote()
