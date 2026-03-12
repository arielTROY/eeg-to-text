"""
Focus Engine Test — runs audio engine + simulated EEG states, no headset needed.
Plays each brain state for 8s through Mac speakers so you can hear the difference.
Then runs the focus benchmark (reaction time test only, quick version).
"""
import time, sys, json, os, random
import numpy as np

# Patch SPEAKER_MODE before importing
os.environ["SPEAKER_MODE"] = "1"

print("=" * 60)
print("🧠 ADHD FOCUS ENGINE — Speaker Test")
print("=" * 60)
print()

# ── Test 1: Audio Engine Layers ──────────────────────────────────────────
print("▶ Starting audio engine on iMac Speakers...")
print("  (Put volume at 40-50%)")
print()

import sounddevice as sd
from adhd_focus_engine import AudioEngine, DOSE_PROFILES

audio = AudioEngine()
audio.start()
print("✓ Audio stream opened: 44100 Hz, stereo, 512 block")
print()

results = {}

states = [
    ("unfocused", 5.2, 2.0, 8.0, 20.0, 0.0, 0.005,
     "High θ/β=5.2 → 40Hz gamma clicks + SR noise + 18Hz isochronic"),
    ("anxious", 1.3, 6.0, 15.0, 8.0, -1.5, 0.008,
     "Low θ/β=1.3 → alpha monaural 10Hz + heavy pink noise + warm pad"),
    ("focused", 2.8, 5.5, 12.0, 10.0, 0.5, 0.020,
     "θ/β=2.8 in zone → maintenance drone + light 14Hz SMR binaural"),
]

doses = ["5mg", "10mg", "20mg"]

print("━" * 60)
print(f"{'DOSE':>6} │ {'STATE':>10} │ {'TBR':>5} │ {'AUDIO RESPONSE':40}")
print("━" * 60)

for dose_key in doses:
    audio.set_dose(dose_key)
    for state, tbr, smr, snr, f_alpha, theta_ph, tgc, desc in states:
        audio.set_state(state, tbr, smr, snr, f_alpha, theta_ph, tgc)

        # Collect what's active
        layers = []
        if audio.gamma_click_vol > 0.001: layers.append(f"γ40={audio.gamma_click_vol:.2f}")
        if audio.binaural_vol > 0.001:    layers.append(f"beat={audio.binaural_beat:.0f}Hz")
        if audio.pink_vol > 0.001:        layers.append(f"SR={audio.pink_vol:.2f}")
        if audio.iso_vol > 0.001:         layers.append(f"iso={audio.iso_freq:.0f}Hz")
        if audio.pad_vol > 0.001:         layers.append(f"pad={audio.pad_vol:.2f}")
        if audio.dmn_interrupt_vol > 0.01:layers.append("DMN!")

        layer_str = " ".join(layers)
        print(f"{dose_key:>6} │ {state:>10} │ {tbr:>5.1f} │ {layer_str:40}")

        results[f"{dose_key}_{state}"] = {
            "dose": dose_key,
            "state": state,
            "tbr": tbr,
            "gamma_click_vol": round(audio.gamma_click_vol, 3),
            "binaural_beat_hz": round(audio.binaural_beat, 1),
            "binaural_vol": round(audio.binaural_vol, 3),
            "pink_noise_vol": round(audio.pink_vol, 3),
            "iso_freq_hz": round(audio.iso_freq, 1),
            "iso_vol": round(audio.iso_vol, 3),
            "pad_vol": round(audio.pad_vol, 3),
            "dmn_interrupt": round(audio.dmn_interrupt_vol, 3),
            "active_layers": layer_str,
        }

        # Play for 3s so user can hear
        time.sleep(3)

print("━" * 60)
print()

# ── Test 2: Play each state at 10mg for 8s with narration ────────────────
print("▶ Demo: 10mg dose, cycling through states (8s each)...")
print("  Listen for the audio changes!")
print()

audio.set_dose("10mg")

demo_states = [
    ("unfocused", 5.0, 2.0, 8.0, 18.0, 0.0, 0.003),
    ("focused",   2.5, 5.0, 14.0, 10.0, 0.5, 0.025),
    ("anxious",   1.2, 7.0, 18.0, 6.0, -1.2, 0.010),
    ("focused",   2.8, 5.5, 12.0, 10.0, 0.5, 0.022),
]

for i, (state, tbr, smr, snr, fa, tp, tgc) in enumerate(demo_states):
    audio.set_state(state, tbr, smr, snr, fa, tp, tgc)
    emoji = {"unfocused": "🟡", "focused": "🟢", "anxious": "🔴"}[state]
    print(f"  {emoji} [{i+1}/4] {state.upper():10} TBR={tbr:.1f}  (playing 8s...)")

    # Trigger rewards in focused state
    if state == "focused":
        time.sleep(3)
        audio.trigger_reward("focus")
        print(f"         🏆 Focus reward chime!")
        time.sleep(3)
        audio.trigger_reward("tgc")
        print(f"         🔗 TGC coupling reward!")
        time.sleep(2)
    else:
        time.sleep(8)

print()

# ── Test 3: Quick RT Variability Test (console version) ──────────────────
print("▶ Quick Reaction Time Test (10 trials)")
print("  Press ENTER as fast as you can when you see GO!")
print("  (Tests your focus consistency — CV < 0.15 = good)")
print()

import select

rts = []
for trial in range(10):
    wait = random.uniform(1.5, 3.5)
    sys.stdout.write(f"  Trial {trial+1:2d}/10: Wait...")
    sys.stdout.flush()
    time.sleep(wait)
    sys.stdout.write(" GO! → ")
    sys.stdout.flush()
    t0 = time.perf_counter()
    input()  # wait for ENTER
    rt = time.perf_counter() - t0
    rts.append(rt * 1000)
    print(f"            {rt*1000:.0f} ms")

rts_arr = np.array(rts)
mean_rt = float(np.mean(rts_arr))
std_rt = float(np.std(rts_arr))
cv = std_rt / mean_rt if mean_rt > 0 else 0

print()
print("━" * 60)
print("📊 REACTION TIME RESULTS")
print("━" * 60)
print(f"  Mean RT:      {mean_rt:.0f} ms")
print(f"  Std Dev:      {std_rt:.0f} ms")
print(f"  Min/Max:      {np.min(rts_arr):.0f} / {np.max(rts_arr):.0f} ms")
print(f"  CV:           {cv:.3f}  {'✅ good (<0.15)' if cv < 0.15 else '⚠️ ADHD-like (>0.20)' if cv > 0.20 else '↔ borderline'}")
print(f"  All RTs:      {[f'{x:.0f}' for x in rts]}")
print("━" * 60)
print()

# ── Summary table ────────────────────────────────────────────────────────
print("=" * 60)
print("📋 FULL AUDIO LAYER SUMMARY")
print("=" * 60)
print()
print(f"{'Dose':>6} │ {'State':>10} │ {'γ40Hz':>6} │ {'Beat':>7} │ {'SR':>6} │ {'Iso':>6} │ {'Pad':>5} │ {'DMN':>4}")
print("─" * 70)
for key, r in results.items():
    print(f"{r['dose']:>6} │ {r['state']:>10} │ {r['gamma_click_vol']:>6.3f} │ "
          f"{r['binaural_beat_hz']:>5.0f}Hz │ {r['pink_noise_vol']:>6.3f} │ "
          f"{r['iso_freq_hz']:>4.0f}Hz │ {r['pad_vol']:>5.3f} │ {r['dmn_interrupt']:>4.2f}")
print("─" * 70)
print()
print("γ40Hz = MIT GENUS 40Hz click volume")
print("Beat  = Monaural beat frequency (entrainment target)")
print("SR    = Stochastic resonance pink noise (calibrated to neural SNR)")
print("Iso   = Isochronic tone pulse rate (arousal)")
print("Pad   = Ambient drone volume (reward circuit)")
print("DMN   = Default mode network interrupt")
print()

# Stop audio
audio.stop()
print("✓ Audio engine stopped")
print("Done! Run again with the EEG headset for real-time brain-adaptive mode.")
