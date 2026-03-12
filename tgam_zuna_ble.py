"""
TGAM (HR-S0C2282B) → BLE → ZUNA → 64ch EEG Pipeline
=====================================================
BLE device: HR-S0C2282B
UUID:       42025A46-1742-3B16-14F7-AD73820062C6
Data char:  0000ffe1-0000-1000-8000-00805f9b34fb  (ThinkGear stream)

Usage:
    python3 tgam_zuna_ble.py              # live run, saves ZUNA output FIFs
    python3 tgam_zuna_ble.py --headless   # no plot (for testing)
"""

import asyncio, argparse, struct, os, time, tempfile, shutil, threading, queue
import numpy as np
import mne
mne.set_log_level("WARNING")

# ── BLE identifiers ───────────────────────────────────────────────────────────
DEVICE_UUID = "42025A46-1742-3B16-14F7-AD73820062C6"
DATA_CHAR   = "0000ffe1-0000-1000-8000-00805f9b34fb"

# ── Pipeline constants ────────────────────────────────────────────────────────
TGAM_SRATE      = 512      # TGAM raw EEG output rate (Hz)
ZUNA_SRATE      = 256      # ZUNA required sample rate
EPOCH_SEC       = 5        # seconds per ZUNA epoch
ZUNA_N_CH       = 64       # target channel count
DOWNSAMPLE      = TGAM_SRATE // ZUNA_SRATE   # 2x

# ── Workspace ─────────────────────────────────────────────────────────────────
WORKDIR     = tempfile.mkdtemp(prefix="tgam_zuna_")
DIR_RAW     = os.path.join(WORKDIR, "0_raw_fif")
DIR_PT_IN   = os.path.join(WORKDIR, "2_pt_in")
DIR_PT_OUT  = os.path.join(WORKDIR, "3_pt_out")
DIR_FIF_OUT = os.path.join(WORKDIR, "4_fif_out")
DIR_PREFIF  = os.path.join(WORKDIR, "1_prefif")
DIR_RESULTS = os.path.join(os.path.dirname(__file__), "zuna_results")

for d in [DIR_RAW, DIR_PT_IN, DIR_PT_OUT, DIR_FIF_OUT, DIR_PREFIF, DIR_RESULTS]:
    os.makedirs(d, exist_ok=True)


# ── ThinkGear parser ──────────────────────────────────────────────────────────
class ThinkGearParser:
    def __init__(self):
        self._buf = bytearray()

    def feed(self, data: bytes):
        self._buf.extend(data)
        results = []
        while len(self._buf) >= 4:
            if self._buf[0] != 0xAA or self._buf[1] != 0xAA:
                self._buf.pop(0); continue
            plen = self._buf[2]
            if plen == 0xAA or plen > 169:
                self._buf.pop(0); continue
            total = 3 + plen + 1
            if len(self._buf) < total:
                break
            payload  = self._buf[3:3 + plen]
            checksum = self._buf[3 + plen]
            self._buf = self._buf[total:]
            if ((~sum(payload)) & 0xFF) != checksum:
                continue
            # parse rows
            i = 0
            while i < len(payload):
                while i < len(payload) and payload[i] == 0x55:
                    i += 1
                if i >= len(payload): break
                code = payload[i]; i += 1
                if code >= 0x80:
                    if i >= len(payload): break
                    vl = payload[i]; i += 1
                    vb = payload[i:i+vl]; i += vl
                    if code == 0x80 and vl == 2:
                        results.append(("raw", struct.unpack(">h", bytes(vb))[0] * 0.1))
                else:
                    if i < len(payload):
                        v = payload[i]; i += 1
                        if code == 0x04: results.append(("att", v))
                        elif code == 0x05: results.append(("med", v))
                        elif code == 0x02: results.append(("sq", v))
        return results


# ── FIF builder ───────────────────────────────────────────────────────────────
def build_fif(samples_uv: np.ndarray, path: str):
    data = samples_uv.reshape(1, -1) * 1e-6
    info = mne.create_info(["Fp1"], sfreq=ZUNA_SRATE, ch_types=["eeg"])
    raw  = mne.io.RawArray(data, info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    raw.save(path, overwrite=True, verbose=False)
    return raw


# ── ZUNA pipeline ─────────────────────────────────────────────────────────────
def run_zuna(epoch_id: int, raw_uv: np.ndarray):
    from zuna import preprocessing, inference, pt_to_fif as zuna_pt2fif

    # clean dirs
    for d in [DIR_RAW, DIR_PT_IN, DIR_PT_OUT, DIR_FIF_OUT, DIR_PREFIF]:
        for f in os.listdir(d): os.remove(os.path.join(d, f))

    fif_in = os.path.join(DIR_RAW, f"ep{epoch_id:04d}.fif")
    build_fif(raw_uv, fif_in)

    try:
        preprocessing(
            input_dir              = DIR_RAW,
            output_dir             = DIR_PT_IN,
            apply_notch_filter     = True,
            apply_highpass_filter  = True,
            apply_average_reference= False,
            target_channel_count   = ZUNA_N_CH,
            save_preprocessed_fif  = True,
            preprocessed_fif_dir   = DIR_PREFIF,
            n_jobs                 = 1,
        )
        inference(
            input_dir            = DIR_PT_IN,
            output_dir           = DIR_PT_OUT,
            gpu_device           = "",          # CPU; change to 0 for GPU
            tokens_per_batch     = 50000,
            diffusion_sample_steps = 20,
        )
        zuna_pt2fif(input_dir=DIR_PT_OUT, output_dir=DIR_FIF_OUT)
    except Exception as e:
        print(f"[ZUNA] Error: {e}")
        return None

    fifiles = [f for f in os.listdir(DIR_FIF_OUT) if f.endswith(".fif")]
    if not fifiles: return None

    out = mne.io.read_raw_fif(os.path.join(DIR_FIF_OUT, fifiles[0]),
                               preload=True, verbose=False)
    # save a permanent copy
    save_path = os.path.join(DIR_RESULTS, f"zuna_ep{epoch_id:04d}.fif")
    out.save(save_path, overwrite=True, verbose=False)
    return out


# ── Report / visualize ────────────────────────────────────────────────────────
def report_epoch(epoch_id, raw_uv_fp1, raw_out, att, med, sq, t_zuna):
    ch_names = raw_out.ch_names
    data_uv  = raw_out.get_data(units="uV")   # (64, n_times)
    sfreq    = raw_out.info["sfreq"]

    # band power per channel
    from scipy.signal import welch
    band_powers = {}
    for band, (lo, hi) in [("delta",(1,4)),("theta",(4,8)),
                            ("alpha",(8,13)),("beta",(13,30)),("gamma",(30,45))]:
        bp = []
        for ch_idx in range(data_uv.shape[0]):
            f, pxx = welch(data_uv[ch_idx], fs=sfreq, nperseg=min(256, data_uv.shape[1]))
            mask = (f >= lo) & (f < hi)
            bp.append(float(np.mean(pxx[mask])))
        band_powers[band] = bp

    # Fp1 raw stats
    fp1_std = float(np.std(raw_uv_fp1))
    fp1_pk  = float(np.max(np.abs(raw_uv_fp1)))

    # top alpha channels
    alpha_per_ch = band_powers["alpha"]
    top_alpha = sorted(zip(alpha_per_ch, ch_names), reverse=True)[:5]

    print(f"\n{'='*60}")
    print(f"  EPOCH {epoch_id}  —  ZUNA took {t_zuna:.1f}s")
    print(f"{'='*60}")
    print(f"  Fp1 raw EEG  : std={fp1_std:.2f} µV  peak={fp1_pk:.2f} µV")
    print(f"  Signal quality (0=best): {sq}")
    print(f"  Attention / Meditation : {att} / {med}")
    print(f"\n  ZUNA 64-channel reconstruction:")
    print(f"    Channels    : {len(ch_names)}")
    print(f"    Duration    : {data_uv.shape[1]/sfreq:.1f}s @ {sfreq:.0f} Hz")

    # mean band power across all channels
    for band, vals in band_powers.items():
        mean_p = np.mean(vals)
        bar = "█" * int(np.clip(mean_p / max(np.mean(list(band_powers.values())
                                              for band_powers in [band_powers])[0]), 0, 1) * 20)
        print(f"    {band:6s} ({['1-4','4-8','8-13','13-30','30-45'][list(band_powers).index(band)]} Hz) : {mean_p:.4f} µV²/Hz")

    print(f"\n  Top alpha channels (8-13 Hz):")
    for p, name in top_alpha:
        print(f"    {name:5s}: {p:.4f} µV²/Hz")

    # save numpy arrays for offline analysis
    np_path = os.path.join(DIR_RESULTS, f"ep{epoch_id:04d}_data.npz")
    np.savez(np_path,
             fp1_raw=raw_uv_fp1,
             zuna_64ch=data_uv,
             ch_names=np.array(ch_names),
             sfreq=sfreq,
             band_powers_alpha=np.array(alpha_per_ch),
             attention=att, meditation=med, signal_quality=sq)
    print(f"\n  Saved: {np_path}")
    print(f"  FIF:   {DIR_RESULTS}/zuna_ep{epoch_id:04d}.fif")
    print(f"{'='*60}\n")

    return band_powers, alpha_per_ch, ch_names


def plot_results(epoch_id, raw_uv_fp1, raw_out):
    """Quick matplotlib summary plot saved to disk."""
    import matplotlib
    matplotlib.use("Agg")   # no display needed
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    data_uv  = raw_out.get_data(units="uV")
    ch_names = raw_out.ch_names
    sfreq    = raw_out.info["sfreq"]
    info     = raw_out.info

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0d0d0d")
    fig.suptitle(f"TGAM → ZUNA  |  Epoch {epoch_id}  |  HR-S0C2282B",
                 color="white", fontsize=14)

    STYLE = dict(facecolor="#111111")
    for ax in axes.flat:
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#aaaaaa")
        for sp in ax.spines.values(): sp.set_color("#333333")

    # 1. Raw Fp1 time-series
    t = np.arange(len(raw_uv_fp1)) / TGAM_SRATE
    axes[0,0].plot(t, raw_uv_fp1, color="#00e5ff", lw=0.6)
    axes[0,0].set_title("Raw EEG — Fp1 (512 Hz)", color="white")
    axes[0,0].set_xlabel("Time (s)", color="#aaa"); axes[0,0].set_ylabel("µV", color="#aaa")

    # 2. Fp1 PSD
    f, pxx = welch(raw_uv_fp1, fs=TGAM_SRATE, nperseg=512)
    mask = f < 60
    axes[0,1].semilogy(f[mask], pxx[mask], color="#ff6f00")
    for fq, lb, c in [(4,"δ","#4fc3f7"),(8,"θ","#81c784"),(13,"α","#ffb74d"),
                       (30,"β","#f06292"),(50,"γ","#ce93d8")]:
        axes[0,1].axvline(fq, color=c, alpha=0.5, lw=0.8, ls="--")
        axes[0,1].text(fq+0.3, axes[0,1].get_ylim()[1], lb, color=c, fontsize=8)
    axes[0,1].set_title("PSD — Fp1", color="white")
    axes[0,1].set_xlabel("Hz", color="#aaa"); axes[0,1].set_ylabel("µV²/Hz", color="#aaa")

    # 3. ZUNA 64ch topomap (alpha power)
    from scipy.signal import welch as scipy_welch
    alpha_power = []
    for ch_idx in range(data_uv.shape[0]):
        f2, p2 = scipy_welch(data_uv[ch_idx], fs=sfreq, nperseg=min(256, data_uv.shape[1]))
        mask2 = (f2 >= 8) & (f2 < 13)
        alpha_power.append(float(np.mean(p2[mask2])))
    alpha_power = np.array(alpha_power)

    im, cn = mne.viz.plot_topomap(
        alpha_power, info,
        axes=axes[0,2], show=False,
        cmap="RdBu_r",
        vlim=(np.percentile(alpha_power,5), np.percentile(alpha_power,95)),
        sensors=True, contours=4,
    )
    axes[0,2].set_title("ZUNA 64ch Alpha Power (8-13 Hz)", color="white")
    plt.colorbar(im, ax=axes[0,2])

    # 4. ZUNA reconstructed channels: Fz, Cz, Pz
    t2 = np.arange(data_uv.shape[1]) / sfreq
    colors = ["#76ff03","#ff6f00","#e040fb"]
    for i, (chname, col) in enumerate(zip(["Fz","Cz","Pz"], colors)):
        if chname in ch_names:
            idx = ch_names.index(chname)
            axes[1,i].plot(t2, data_uv[idx], color=col, lw=0.7)
            axes[1,i].set_title(f"ZUNA: {chname}", color="white")
            axes[1,i].set_xlabel("Time (s)", color="#aaa")
            axes[1,i].set_ylabel("µV", color="#aaa")

    plt.tight_layout()
    plot_path = os.path.join(DIR_RESULTS, f"ep{epoch_id:04d}_plot.png")
    plt.savefig(plot_path, dpi=150, facecolor="#0d0d0d")
    plt.close()
    print(f"  Plot: {plot_path}")
    return plot_path


# ── Main BLE capture + pipeline loop ─────────────────────────────────────────
async def main_loop(n_epochs: int = 3, headless: bool = False):
    from bleak import BleakClient

    parser   = ThinkGearParser()
    sample_q = asyncio.Queue()
    esense   = {"att": 0, "med": 0, "sq": 200}

    def ble_handler(sender, data: bytearray):
        results = parser.feed(bytes(data))
        for code, val in results:
            if code == "raw":
                sample_q.put_nowait(val)
            elif code == "att":
                esense["att"] = val
            elif code == "med":
                esense["med"] = val
            elif code == "sq":
                esense["sq"]  = val

    n_needed = EPOCH_SEC * TGAM_SRATE

    print(f"\n[BLE] Connecting to HR-S0C2282B…")
    async with BleakClient(DEVICE_UUID, timeout=20.0) as client:
        print(f"[BLE] ✓ Connected  MTU={client.mtu_size}")
        await client.start_notify(DATA_CHAR, ble_handler)
        print(f"[BLE] Subscribed to {DATA_CHAR[-8:]}")
        print(f"[BLE] Collecting data — wear electrode on left forehead (Fp1)\n")

        epoch_buf = []
        epoch_id  = 0
        t_start   = time.time()

        while epoch_id < n_epochs:
            await asyncio.sleep(0.02)

            # drain async queue
            while not sample_q.empty():
                epoch_buf.append(await sample_q.get())

            rate = len(epoch_buf) / max(time.time() - t_start, 0.1)
            if len(epoch_buf) % (TGAM_SRATE * 2) < 50:   # print every ~2s
                sq_str = "OK" if esense["sq"] == 0 else f"poor({esense['sq']})"
                print(f"  Buffered {len(epoch_buf):5d}/{n_needed} samples  "
                      f"rate≈{rate:.0f}Hz  SQ={sq_str}  "
                      f"att={esense['att']}  med={esense['med']}", end="\r")

            if len(epoch_buf) >= n_needed:
                epoch_id += 1
                raw_512hz = np.array(epoch_buf[:n_needed], dtype=np.float32)
                epoch_buf  = epoch_buf[n_needed:]
                t_start    = time.time()

                # downsample 512→256 Hz
                raw_256hz = raw_512hz[::DOWNSAMPLE]

                print(f"\n[EP{epoch_id}] Captured {len(raw_512hz)} samples  "
                      f"std={np.std(raw_512hz):.2f}µV  peak={np.max(np.abs(raw_512hz)):.2f}µV")
                print(f"[EP{epoch_id}] Running ZUNA ({ZUNA_N_CH}ch reconstruction)…")

                t0 = time.time()
                raw_out = await asyncio.get_event_loop().run_in_executor(
                    None, run_zuna, epoch_id, raw_256hz)
                t_zuna = time.time() - t0

                if raw_out is None:
                    print(f"[EP{epoch_id}] ZUNA failed — skipping")
                    continue

                report_epoch(epoch_id, raw_512hz, raw_out,
                             esense["att"], esense["med"], esense["sq"], t_zuna)

                if not headless:
                    plot_path = plot_results(epoch_id, raw_512hz, raw_out)
                    try:
                        os.system(f"open {plot_path}")   # open in Preview on Mac
                    except:
                        pass

        await client.stop_notify(DATA_CHAR)
        print(f"\n[DONE] {epoch_id} epochs processed. Results in: {DIR_RESULTS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",   type=int,  default=3,     help="Number of 5-s epochs to process")
    ap.add_argument("--headless", action="store_true",       help="Skip opening plots")
    args = ap.parse_args()

    try:
        asyncio.run(main_loop(n_epochs=args.epochs, headless=args.headless))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        shutil.rmtree(WORKDIR, ignore_errors=True)
