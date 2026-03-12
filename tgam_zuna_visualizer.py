"""
TGAM → ZUNA → 64ch Live EEG Visualizer
========================================
Pipeline:
  1. Reads raw EEG from TGAM module via USB-UART serial (ThinkGear protocol)
  2. Buffers 5-second windows at 512 Hz → downsamples to 256 Hz
  3. Saves single-channel Fp1 .fif with 3D montage
  4. Runs ZUNA to upsample 1 → 64 channels
  5. Visualizes: raw Fp1 signal + reconstructed 64ch topomap + channel time-series

Hardware setup:
  TGAM TX  → USB-UART RX
  TGAM RX  → USB-UART TX
  TGAM VCC → 3.3V
  TGAM GND → GND
  Baud: 57600 (raw EEG mode — MODE pin HIGH or floating)

Electrode placement:
  Main electrode  → Left forehead (Fp1, above left eyebrow)
  Ear clip ref    → Left earlobe
  Run: python tgam_zuna_visualizer.py --port /dev/tty.usbserial-XXXX
"""

import os
import sys
import time
import shutil
import struct
import argparse
import threading
import queue
import tempfile
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # works headless-safe on Mac; try 'Qt5Agg' if TkAgg fails
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
import mne
mne.set_log_level("WARNING")

# ─── CONFIG ──────────────────────────────────────────────────────────────────
TGAM_BAUD       = 57600          # 57600 = raw EEG mode; 9600 = processed only
TGAM_SRATE      = 512            # TGAM raw EEG output rate (Hz)
ZUNA_SRATE      = 256            # ZUNA fixed input rate
EPOCH_SEC       = 5              # ZUNA epoch window (fixed by pretrained model)
ZUNA_N_CHANNELS = 64             # target channel count after upsampling

# 64-channel target: first 64 channels from standard 10-20 extended montage
# (ZUNA uses greedy selection from 10-05 to reach this count)
ZUNA_TARGET_CH  = 64

# Display: which reconstructed channels to show as time-series (besides Fp1)
DISPLAY_CHANNELS = ["Fz", "Cz", "Pz", "O1", "T7", "T8"]

# Workspace dirs (auto-cleaned each cycle)
WORKDIR = tempfile.mkdtemp(prefix="tgam_zuna_")
RAW_FIF_DIR   = os.path.join(WORKDIR, "0_raw_fif")
PT_INPUT_DIR  = os.path.join(WORKDIR, "2_pt_input")
PT_OUTPUT_DIR = os.path.join(WORKDIR, "3_pt_output")
FIF_OUTPUT_DIR = os.path.join(WORKDIR, "4_fif_output")
FIF_FILTER_DIR = os.path.join(WORKDIR, "1_fif_filter")

for d in [RAW_FIF_DIR, PT_INPUT_DIR, PT_OUTPUT_DIR, FIF_OUTPUT_DIR, FIF_FILTER_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── THINKGEAR PARSER ────────────────────────────────────────────────────────
class ThinkGearParser:
    """
    Minimal ThinkGear ASIC Module (TGAM) binary protocol parser.
    Extracts raw EEG samples (CODE 0x80) at ~512 Hz.

    Packet structure:
      [0xAA] [0xAA] [PLENGTH] [PAYLOAD...] [CHECKSUM]
      Inside payload, each data row: [EXCODE...] [CODE] [VLENGTH] [VALUE...]
      Raw EEG: CODE=0x80, VLENGTH=2, VALUE=big-endian signed int16
    """
    SYNC = 0xAA

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data: bytes):
        """Feed raw bytes; returns list of (code, value) tuples parsed."""
        self._buf.extend(data)
        results = []
        while len(self._buf) >= 4:
            # Find sync pair
            if self._buf[0] != self.SYNC or self._buf[1] != self.SYNC:
                self._buf.pop(0)
                continue
            if len(self._buf) < 3:
                break
            plength = self._buf[2]
            if plength == self.SYNC:          # invalid length
                self._buf.pop(0)
                continue
            if plength > 169:                 # max payload
                self._buf.pop(0)
                continue
            total = 3 + plength + 1           # sync(2)+plen(1)+payload+checksum
            if len(self._buf) < total:
                break                         # wait for more bytes
            payload  = self._buf[3:3 + plength]
            checksum = self._buf[3 + plength]
            # Verify checksum
            calc = (~sum(payload)) & 0xFF
            self._buf = self._buf[total:]     # consume regardless
            if calc != checksum:
                continue
            # Parse payload rows
            i = 0
            while i < len(payload):
                # Skip EXCODE bytes (0x55)
                while i < len(payload) and payload[i] == 0x55:
                    i += 1
                if i >= len(payload):
                    break
                code = payload[i]; i += 1
                if code >= 0x80:              # multi-byte value
                    if i >= len(payload):
                        break
                    vlength = payload[i]; i += 1
                    val_bytes = payload[i:i + vlength]; i += vlength
                    if code == 0x80 and vlength == 2:
                        raw = struct.unpack(">h", bytes(val_bytes))[0]
                        results.append(("raw_eeg", raw))
                else:                         # single-byte value
                    if i < len(payload):
                        val = payload[i]; i += 1
                        if code == 0x04:
                            results.append(("attention", val))
                        elif code == 0x05:
                            results.append(("meditation", val))
        return results


# ─── SERIAL READER THREAD ─────────────────────────────────────────────────────
class TGAMReader(threading.Thread):
    """
    Background thread: reads serial, parses ThinkGear, pushes raw_eeg samples
    into sample_queue as floats (microvolts — TGAM raw unit ≈ 0.1 µV/LSB).
    """
    UV_PER_LSB = 0.1   # approximate; TGAM datasheet varies

    def __init__(self, port: str, sample_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.port         = port
        self.q            = sample_queue
        self.stop_event   = stop_event
        self.parser       = ThinkGearParser()
        self.samples_read = 0
        self.signal_quality = 200  # 0=good, 200=no contact

    def run(self):
        import serial
        try:
            ser = serial.Serial(self.port, TGAM_BAUD, timeout=0.1)
        except Exception as e:
            print(f"[TGAM] Serial open failed: {e}")
            print(f"[TGAM] Available ports:")
            import serial.tools.list_ports
            for p in serial.tools.list_ports.comports():
                print(f"       {p.device}  {p.description}")
            self.stop_event.set()
            return

        print(f"[TGAM] Connected on {self.port} @ {TGAM_BAUD} baud")
        while not self.stop_event.is_set():
            try:
                chunk = ser.read(64)
                if not chunk:
                    continue
                results = self.parser.feed(chunk)
                for code, val in results:
                    if code == "raw_eeg":
                        uv = val * self.UV_PER_LSB
                        self.q.put(uv)
                        self.samples_read += 1
            except Exception as e:
                print(f"[TGAM] Read error: {e}")
                break
        ser.close()
        print("[TGAM] Serial closed.")


# ─── SIMULATE TGAM (for testing without hardware) ────────────────────────────
class TGAMSimulator(threading.Thread):
    """Generates synthetic alpha + noise EEG at 512 Hz for testing without hardware."""
    def __init__(self, sample_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.q          = sample_queue
        self.stop_event = stop_event

    def run(self):
        print("[TGAM] Running in SIMULATION mode (no hardware connected)")
        t      = 0
        dt     = 1.0 / TGAM_SRATE
        while not self.stop_event.is_set():
            # Synthetic EEG: alpha (10 Hz) + beta (20 Hz) + noise
            alpha = 15.0  * np.sin(2 * np.pi * 10  * t)
            beta  =  5.0  * np.sin(2 * np.pi * 20  * t)
            gamma =  2.0  * np.sin(2 * np.pi * 40  * t)
            noise =  8.0  * np.random.randn()
            uv    = alpha + beta + gamma + noise
            self.q.put(float(uv))
            t += dt
            time.sleep(dt)


# ─── FIF BUILDER ──────────────────────────────────────────────────────────────
def build_fif(samples_uv: np.ndarray, filepath: str):
    """
    Takes Fp1 samples (microvolts, 256 Hz after downsampling),
    creates an MNE Raw object with 3D montage, saves as .fif.
    """
    data = samples_uv.reshape(1, -1) * 1e-6   # MNE uses Volts
    info = mne.create_info(ch_names=["Fp1"], sfreq=ZUNA_SRATE, ch_types=["eeg"])
    raw  = mne.io.RawArray(data, info, verbose=False)
    # Apply 3D electrode positions from standard 10-20
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore", verbose=False)
    raw.save(filepath, overwrite=True, verbose=False)
    return raw


# ─── ZUNA PIPELINE ────────────────────────────────────────────────────────────
def run_zuna(epoch_id: int, raw_fp1: np.ndarray, use_gpu: bool = False):
    """
    Full ZUNA pipeline: 1 Fp1 channel → 64 channel reconstruction.
    Returns MNE Raw of reconstructed 64-channel signal, or None on failure.
    """
    from zuna import preprocessing, inference, pt_to_fif as zuna_pt_to_fif

    # Clean previous epoch files
    for d in [RAW_FIF_DIR, PT_INPUT_DIR, PT_OUTPUT_DIR, FIF_OUTPUT_DIR, FIF_FILTER_DIR]:
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    fif_path = os.path.join(RAW_FIF_DIR, f"epoch_{epoch_id:04d}.fif")
    build_fif(raw_fp1, fif_path)

    gpu_device = 0 if use_gpu else ""

    try:
        preprocessing(
            input_dir           = RAW_FIF_DIR,
            output_dir          = PT_INPUT_DIR,
            apply_notch_filter  = True,
            apply_highpass_filter = True,
            apply_average_reference = False,   # only 1 channel, skip avg ref
            target_channel_count = ZUNA_TARGET_CH,
            save_preprocessed_fif = True,
            preprocessed_fif_dir  = FIF_FILTER_DIR,
            n_jobs              = 1,
        )
        inference(
            input_dir           = PT_INPUT_DIR,
            output_dir          = PT_OUTPUT_DIR,
            gpu_device          = gpu_device,
            tokens_per_batch    = 50000,
            data_norm           = None,
            diffusion_sample_steps = 20,       # fewer steps = faster, good enough for live viz
        )
        zuna_pt_to_fif(
            input_dir  = PT_OUTPUT_DIR,
            output_dir = FIF_OUTPUT_DIR,
        )
    except Exception as e:
        print(f"[ZUNA] Pipeline error: {e}")
        return None

    # Load reconstructed .fif
    fif_files = [f for f in os.listdir(FIF_OUTPUT_DIR) if f.endswith(".fif")]
    if not fif_files:
        print("[ZUNA] No output .fif found")
        return None

    out_path = os.path.join(FIF_OUTPUT_DIR, fif_files[0])
    try:
        raw_out = mne.io.read_raw_fif(out_path, preload=True, verbose=False)
        return raw_out
    except Exception as e:
        print(f"[ZUNA] Failed to load output .fif: {e}")
        return None


# ─── LIVE VISUALIZER ──────────────────────────────────────────────────────────
class LiveVisualizer:
    def __init__(self, rolling_sec: int = 10):
        self.rolling_n = rolling_sec * TGAM_SRATE
        self.raw_buffer = deque(maxlen=self.rolling_n)
        self.last_reconstructed = None   # MNE Raw from ZUNA
        self.epoch_count = 0
        self.attention_val = 0
        self._setup_figure()

    def _setup_figure(self):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 9), facecolor="#0d0d0d")
        self.fig.canvas.manager.set_window_title("TGAM + ZUNA Live EEG")
        gs = gridspec.GridSpec(3, 3, figure=self.fig,
                               hspace=0.45, wspace=0.35,
                               left=0.06, right=0.97, top=0.93, bottom=0.06)

        # Top-left: raw Fp1 rolling window
        self.ax_raw = self.fig.add_subplot(gs[0, :2])
        self.ax_raw.set_facecolor("#111111")
        self.ax_raw.set_title("Raw EEG — Fp1 (rolling 10 s)", color="white", fontsize=10)
        self.ax_raw.set_xlabel("Time (s)", color="#888888")
        self.ax_raw.set_ylabel("µV", color="#888888")
        self.ax_raw.tick_params(colors="#888888")
        for spine in self.ax_raw.spines.values():
            spine.set_color("#333333")
        self.line_raw, = self.ax_raw.plot([], [], color="#00e5ff", lw=0.6)

        # Top-right: PSD of Fp1
        self.ax_psd = self.fig.add_subplot(gs[0, 2])
        self.ax_psd.set_facecolor("#111111")
        self.ax_psd.set_title("PSD — Fp1", color="white", fontsize=10)
        self.ax_psd.set_xlabel("Hz", color="#888888")
        self.ax_psd.set_ylabel("dB", color="#888888")
        self.ax_psd.tick_params(colors="#888888")
        for spine in self.ax_psd.spines.values():
            spine.set_color("#333333")
        self.line_psd, = self.ax_psd.plot([], [], color="#ff6f00", lw=1)
        # Band markers
        for fq, label, c in [(4,"δ","#4fc3f7"),(8,"θ","#81c784"),
                              (13,"α","#ffb74d"),(30,"β","#f06292"),(50,"γ","#ce93d8")]:
            self.ax_psd.axvline(fq, color=c, alpha=0.4, lw=0.8, ls="--")
            self.ax_psd.text(fq+0.5, 0.95, label, transform=self.ax_psd.get_xaxis_transform(),
                             color=c, fontsize=7, va="top")

        # Middle: 64ch topomap
        self.ax_topo = self.fig.add_subplot(gs[1, 1])
        self.ax_topo.set_facecolor("#0d0d0d")
        self.ax_topo.set_title("ZUNA 64ch Topomap (last epoch)", color="white", fontsize=10)
        self._topo_obj = None

        # Middle-left: attention/meditation bar
        self.ax_bar = self.fig.add_subplot(gs[1, 0])
        self.ax_bar.set_facecolor("#111111")
        self.ax_bar.set_title("eSense Meters", color="white", fontsize=10)
        self.ax_bar.set_xlim(0, 100)
        self.ax_bar.tick_params(colors="#888888")
        for spine in self.ax_bar.spines.values():
            spine.set_color("#333333")
        self.bar_attention  = self.ax_bar.barh(["Attention", "Meditation"], [0, 0],
                                                color=["#00e5ff", "#76ff03"])

        # Middle-right: status
        self.ax_status = self.fig.add_subplot(gs[1, 2])
        self.ax_status.set_facecolor("#0d0d0d")
        self.ax_status.axis("off")
        self.txt_status = self.ax_status.text(0.05, 0.85, "Waiting for ZUNA...",
                                               transform=self.ax_status.transAxes,
                                               color="white", fontsize=9,
                                               va="top", fontfamily="monospace")

        # Bottom: reconstructed channel time-series
        self.ax_chans = []
        n_display = min(len(DISPLAY_CHANNELS), 3)
        for i in range(n_display):
            ax = self.fig.add_subplot(gs[2, i])
            ax.set_facecolor("#111111")
            ax.set_title(f"ZUNA: {DISPLAY_CHANNELS[i]}", color="white", fontsize=9)
            ax.tick_params(colors="#888888")
            for spine in ax.spines.values():
                spine.set_color("#333333")
            self.ax_chans.append(ax)
        self.lines_chans = [ax.plot([], [], color="#76ff03", lw=0.7)[0] for ax in self.ax_chans]

        plt.show(block=False)
        plt.pause(0.05)

    def push_raw_sample(self, uv: float, attention: int = 0, meditation: int = 0):
        self.raw_buffer.append(uv)
        self.attention_val  = attention
        self.meditation_val = meditation if hasattr(self, "meditation_val") else 0

    def update_raw_plot(self):
        if len(self.raw_buffer) < 2:
            return
        arr = np.array(self.raw_buffer)
        t   = np.arange(len(arr)) / TGAM_SRATE
        self.line_raw.set_data(t, arr)
        self.ax_raw.set_xlim(t[0], t[-1])
        lo, hi = arr.min(), arr.max()
        pad = max((hi - lo) * 0.1, 1.0)
        self.ax_raw.set_ylim(lo - pad, hi + pad)

        # PSD via Welch
        if len(arr) >= TGAM_SRATE * 2:
            from scipy.signal import welch
            f, pxx = welch(arr, fs=TGAM_SRATE, nperseg=min(len(arr), 512))
            mask = f < 60
            pxx_db = 10 * np.log10(pxx[mask] + 1e-12)
            self.line_psd.set_data(f[mask], pxx_db)
            self.ax_psd.set_xlim(0, 60)
            self.ax_psd.set_ylim(pxx_db.min() - 5, pxx_db.max() + 5)

    def update_zuna_plot(self, raw_reconstructed):
        """Update topomap and channel time-series from ZUNA output MNE Raw."""
        self.last_reconstructed = raw_reconstructed
        self.epoch_count += 1

        data = raw_reconstructed.get_data(units="uV")  # shape: (n_ch, n_times)
        info = raw_reconstructed.info
        ch_names = raw_reconstructed.ch_names

        # ── Topomap: mean power per channel ──
        mean_power = np.mean(data ** 2, axis=1)
        self.ax_topo.clear()
        self.ax_topo.set_facecolor("#0d0d0d")
        self.ax_topo.set_title(f"ZUNA 64ch Topomap (epoch {self.epoch_count})",
                                color="white", fontsize=10)
        try:
            mne.viz.plot_topomap(
                mean_power,
                info,
                axes=self.ax_topo,
                show=False,
                cmap="RdBu_r",
                vlim=(np.percentile(mean_power, 5), np.percentile(mean_power, 95)),
                sensors=True,
                contours=4,
            )
        except Exception as e:
            self.ax_topo.text(0.5, 0.5, f"Topo error:\n{e}",
                              ha="center", va="center", color="red",
                              transform=self.ax_topo.transAxes, fontsize=7)

        # ── Channel time-series ──
        t = np.arange(data.shape[1]) / raw_reconstructed.info["sfreq"]
        for i, (ax, line) in enumerate(zip(self.ax_chans, self.lines_chans)):
            ch = DISPLAY_CHANNELS[i] if i < len(DISPLAY_CHANNELS) else None
            if ch and ch in ch_names:
                idx = ch_names.index(ch)
                sig = data[idx]
                line.set_data(t, sig)
                ax.set_xlim(t[0], t[-1])
                ax.set_ylim(sig.min() - 1, sig.max() + 1)
            ax.figure.canvas.draw_idle()

        # ── Status ──
        self.txt_status.set_text(
            f"Epochs processed : {self.epoch_count}\n"
            f"Channels out     : {len(ch_names)}\n"
            f"Duration         : {data.shape[1] / ZUNA_SRATE:.1f} s\n"
            f"Attention        : {self.attention_val}\n"
            f"ZUNA ok ✓"
        )

    def render(self):
        try:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        except Exception:
            pass


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TGAM → ZUNA live EEG visualizer")
    parser.add_argument("--port",       type=str,  default=None,
                        help="Serial port, e.g. /dev/tty.usbserial-1410. Omit to simulate.")
    parser.add_argument("--gpu",        action="store_true",
                        help="Use GPU for ZUNA inference (default: CPU)")
    parser.add_argument("--steps",      type=int,  default=20,
                        help="ZUNA diffusion steps (fewer = faster, default 20)")
    parser.add_argument("--simulate",   action="store_true",
                        help="Force simulation mode even if port is given")
    args = parser.parse_args()

    use_gpu     = args.gpu
    stop_event  = threading.Event()
    sample_q    = queue.Queue()
    esense_q    = queue.Queue(maxsize=10)   # attention/meditation values

    # ── Start capture ──
    if args.port and not args.simulate:
        reader = TGAMReader(args.port, sample_q, stop_event)
    else:
        if not args.port:
            print("[INFO] No --port specified. Listing available serial ports:")
            try:
                import serial.tools.list_ports
                for p in serial.tools.list_ports.comports():
                    print(f"       {p.device}  ({p.description})")
            except Exception:
                pass
        reader = TGAMSimulator(sample_q, stop_event)
    reader.start()

    viz = LiveVisualizer(rolling_sec=10)

    # Collect samples for one epoch
    epoch_buf   = []
    epoch_id    = 0
    n_needed    = EPOCH_SEC * TGAM_SRATE        # 2560 samples @ 512 Hz → 5 s
    downsample_factor = TGAM_SRATE // ZUNA_SRATE  # 2×

    print(f"[INFO] Collecting {n_needed} samples per epoch ({EPOCH_SEC}s @ {TGAM_SRATE} Hz)…")
    print(f"[INFO] ZUNA target: {ZUNA_TARGET_CH} channels | GPU: {use_gpu}")
    print("[INFO] Close the plot window or press Ctrl+C to stop.")

    zuna_result_q = queue.Queue(maxsize=1)

    def zuna_worker():
        """Runs ZUNA in a background thread so UI stays responsive."""
        while not stop_event.is_set():
            try:
                epoch_data, eid = zuna_result_q.get(timeout=1.0)
            except queue.Empty:
                continue
            print(f"[ZUNA] Processing epoch {eid} ({len(epoch_data)} samples)…")
            # Downsample 512→256 Hz via simple decimation (ZUNA preprocessor will filter)
            ds = np.array(epoch_data[::downsample_factor])
            t0 = time.time()
            raw_out = run_zuna(eid, ds, use_gpu=use_gpu)
            elapsed = time.time() - t0
            if raw_out is not None:
                print(f"[ZUNA] Epoch {eid} done in {elapsed:.1f}s — "
                      f"{len(raw_out.ch_names)} channels reconstructed")
                viz.update_zuna_plot(raw_out)
            else:
                print(f"[ZUNA] Epoch {eid} failed.")

    zuna_thread = threading.Thread(target=zuna_worker, daemon=True)
    zuna_thread.start()

    attention  = 0
    meditation = 0
    plot_tick  = 0

    try:
        while plt.fignum_exists(viz.fig.number):
            # Drain the sample queue
            drained = 0
            while not sample_q.empty() and drained < 256:
                uv = sample_q.get_nowait()
                epoch_buf.append(uv)
                viz.push_raw_sample(uv, attention, meditation)
                drained += 1

            # Update raw plot every ~100 ms (50 samples @ 512 Hz)
            plot_tick += drained
            if plot_tick >= 50:
                viz.update_raw_plot()
                plot_tick = 0

            # When we have a full epoch, send to ZUNA (non-blocking)
            if len(epoch_buf) >= n_needed:
                epoch_data = epoch_buf[:n_needed]
                epoch_buf  = epoch_buf[n_needed:]   # keep remainder
                epoch_id  += 1
                if zuna_result_q.empty():
                    zuna_result_q.put((epoch_data, epoch_id))
                else:
                    print(f"[ZUNA] Skipping epoch {epoch_id} — still processing previous")

            viz.render()
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping…")

    stop_event.set()
    print(f"[INFO] Processed {epoch_id} epochs total. Workspace: {WORKDIR}")
    shutil.rmtree(WORKDIR, ignore_errors=True)


if __name__ == "__main__":
    main()
