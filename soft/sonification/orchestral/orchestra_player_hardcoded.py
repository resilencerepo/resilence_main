
"""
Orchestral Layer Player — Mix / Harmony / Tune→Harmony (absolute pitch + final lock)

- MIX: random instruments per section, your per-layer fades/gains/mutes, staggered entries, master envelope
- HARMONY: forces sustain sources, detects each layer's pitch, shifts to absolute chord targets (major/minor),
           and GUARANTEES a clean chord by locking the LAST seconds to the exact target
- TUNE→HARMONY: begins slightly detuned per layer, progressively converges to the same absolute chord targets,
                and also locks the tail to the exact targets

No external DSP libs required (uses numpy + soundfile + sounddevice).

Run:
  pip install sounddevice soundfile numpy
  python orchestra_player_hardcoded.py /path/to/dataset_root --mode mix
  python orchestra_player_hardcoded.py /path/to/dataset_root --mode harmony
  python orchestra_player_hardcoded.py /path/to/dataset_root --mode tune2harmony
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
import sounddevice as sd
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import argparse
import re

# -------------------------- FILETYPES & SR --------------------------
AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".aifc"}
SR = 44100


# -------------------- MIX MODE MASTER ENVELOPE ---------------------
MASTER_FADE_IN_S  = 1.0
MASTER_SUSTAIN_S  = 4.0
MASTER_FADE_OUT_S = 2.0
MASTER_MAXVOL     = 0.8

# ---------------------- HARMONY MASTER ENVELOPE --------------------
HARMONY_MASTER_FADE_IN_S  = MASTER_FADE_IN_S
HARMONY_MASTER_SUSTAIN_S  = MASTER_SUSTAIN_S
HARMONY_MASTER_FADE_OUT_S = MASTER_FADE_OUT_S
HARMONY_MASTER_MAXVOL     = MASTER_MAXVOL

# ---------------------- TUNE→HARMONY MASTER ------------------------
TUNE2HARM_MASTER_FADE_IN_S  = MASTER_FADE_IN_S
TUNE2HARM_MASTER_SUSTAIN_S  = MASTER_SUSTAIN_S
TUNE2HARM_MASTER_FADE_OUT_S = MASTER_FADE_OUT_S
TUNE2HARM_MASTER_MAXVOL     = MASTER_MAXVOL

# ----------------------------- LAYERS -------------------------------
@dataclass
class LayerConfig:
    gain: float = 1.0
    fade_in_s: float = 0.05
    fade_out_s: float = 0.05
    mute: bool = False

# User-provided per-layer settings
LAYER_CFG: Dict[str, LayerConfig] = {
    "S":      LayerConfig(gain=0.3, fade_in_s=0.10, fade_out_s=0.20, mute=True),
    "A":      LayerConfig(gain=0.4, fade_in_s=0.20, fade_out_s=0.20, mute=True),
    "T":      LayerConfig(gain=0.5, fade_in_s=0.05, fade_out_s=0.10, mute=False),
    "B":      LayerConfig(gain=0.5, fade_in_s=0.20, fade_out_s=0.30, mute=False),
    "Melody": LayerConfig(gain=0.7, fade_in_s=0.05, fade_out_s=0.05, mute=False),
}

# Entry staggering
#ENTRY_OFFSET_MIN_S = 0.25
#ENTRY_OFFSET_MAX_S = 1.25
ENTRY_OFFSET_MIN_S = 0.0
ENTRY_OFFSET_MAX_S = 0.0


# --------------------------- HARMONY SETUP --------------------------
HARMONY_CHORD_TYPE = "major"  # or "minor"
HARMONY_ROOT = "C4"           # choose the root note (e.g., "C4", "A3", "D#3")

FORCE_SUSTAIN_FOR_HARMONY   = True
FORCE_SUSTAIN_FOR_TUNE2HARM = True

# Voicings in semitones relative to root
HARMONY_VOICING_MAJOR_ST = {"B": 0, "T": 7, "A": 4, "S": 12, "Melody": 19}
HARMONY_VOICING_MINOR_ST = {"B": 0, "T": 7, "A": 3, "S": 12, "Melody": 15}

# Tune→Harmony: opening spread & morph windows
TUNE_START_DETUNE_RANGE_CENTS = (-35, 35)
TUNE2HARM_WINDOW_S = 0.05
TUNE2HARM_XFADE_S  = 0.006

# -------- Final "lock" — guarantee the ending is clean triad ----------
LOCK_LAST_S  = 0.9     # how many seconds at the END to force to exact targets
LOCK_XFADE_S = 0.03    # crossfade into the locked tail
DEBUG_PITCH  = False   # print detected->target info per layer

# ---------------------------- UTILITIES ----------------------------
def intro_outro_envelope(duration_s: float, sr: int, fade_in_s: float, sustain_s: float, fade_out_s: float, max_volume: float = 1.0) -> np.ndarray:
    total = int(duration_s * sr)
    env = np.zeros(total, dtype=np.float32)
    assert abs(duration_s - (fade_in_s + sustain_s + fade_out_s)) < 1e-6, "Master duration mismatch"
    n_fade_in = int(fade_in_s * sr)
    n_sustain = int(sustain_s * sr)
    n_fade_out = total - (n_fade_in + n_sustain)
    if n_fade_in > 0:
        env[:n_fade_in] = np.linspace(0.0, max_volume, num=n_fade_in, endpoint=False, dtype=np.float32)
    if n_sustain > 0:
        env[n_fade_in:n_fade_in+n_sustain] = max_volume
    if n_fade_out > 0:
        env[n_fade_in+n_sustain:] = np.linspace(max_volume, 0.0, num=n_fade_out, endpoint=True, dtype=np.float32)
    return env


def list_instruments(articulation_dir: Path) -> List[Path]:
    if not articulation_dir.exists():
        return []
    return [p for p in articulation_dir.iterdir() if p.is_dir()]


def gather_audio_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def read_audio_mono_resampled(path: Path, target_sr: int) -> np.ndarray:
    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = data.mean(axis=1)
    if sr != target_sr:
        x_old = np.linspace(0, 1, num=len(mono), endpoint=False, dtype=np.float32)
        x_new = np.linspace(0, 1, num=int(len(mono) * (target_sr / sr)), endpoint=False, dtype=np.float32)
        mono = np.interp(x_new, x_old, mono).astype(np.float32)
    peak = np.max(np.abs(mono)) if mono.size > 0 else 1.0
    if peak > 0:
        mono = mono / max(1.0, peak)
    return mono


# ---- Pitch helpers ----
NOTE2SEMITONE = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
def note_to_midi(note: str) -> int:
    m = re.match(r'^([A-Ga-g][#b]?)(-?\d+)$', note.strip())
    if not m:
        raise ValueError(f"Bad note name: {note}")
    name = m.group(1).capitalize()
    octv = int(m.group(2))
    semi = NOTE2SEMITONE[name]
    return 12 * (octv + 1) + semi  # MIDI: A4=69

def midi_to_hz(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

def cents_between(f1: float, f2: float) -> float:
    if f1 <= 0 or f2 <= 0:
        return 0.0
    return 1200.0 * np.log2(f2 / f1)


def estimate_pitch_hz(x: np.ndarray, sr: int, fmin=55.0, fmax=1400.0) -> float:
    """Autocorrelation pitch estimate on a centered window."""
    n = x.size
    if n < sr // 20:
        return 0.0
    win_n = min(n, int(0.4 * sr))
    start = max(0, (n - win_n)//2)
    seg = x[start:start+win_n].astype(np.float32)
    if not np.any(seg):
        return 0.0
    seg = seg - np.mean(seg)
    size = 1
    while size < (2 * seg.size):
        size <<= 1
    fft = np.fft.rfft(seg, size)
    ac = np.fft.irfft(fft * np.conj(fft))[:seg.size]
    mx = float(np.max(np.abs(ac))) + 1e-9
    ac = ac / mx
    lag_min = max(1, int(sr / fmax))
    lag_max = min(seg.size - 1, int(sr / fmin))
    if lag_max <= lag_min:
        return 0.0
    lag = lag_min + int(np.argmax(ac[lag_min:lag_max+1]))
    if 1 <= lag < seg.size - 1:
        y0, y1, y2 = ac[lag-1], ac[lag], ac[lag+1]
        denom = (y0 - 2*y1 + y2)
        if abs(denom) > 1e-9:
            delta = 0.5 * (y0 - y2) / denom
            lag = lag + float(delta)
    freq = sr / lag if lag > 0 else 0.0
    return float(max(0.0, freq))


def apply_detune_full_window(buffer: np.ndarray, cents: float) -> np.ndarray:
    if abs(cents) < 1e-3:
        return buffer.copy()
    factor = 2.0 ** (cents / 1200.0)
    n_new = max(1, int(round(buffer.size / factor)))
    x_old = np.linspace(0, 1, num=buffer.size, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0, 1, num=n_new, endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, buffer).astype(np.float32)


def detune_preserve_length(x: np.ndarray, cents: float) -> np.ndarray:
    if x.size == 0 or abs(cents) < 1e-3:
        return x.copy()
    shifted = apply_detune_full_window(x, cents)
    x_old = np.linspace(0, 1, num=shifted.size, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0, 1, num=x.size, endpoint=False, dtype=np.float32)
    return np.interp(x_new, x_old, shifted).astype(np.float32)


def progressive_detune_to_target(x: np.ndarray, sr: int, cents_start: float, cents_end: float,
                                 win_s: float, xfade_s: float) -> np.ndarray:
    """Progressively morph pitch from start cents to end cents (duration preserved)."""
    n = x.size
    if n == 0:
        return x
    win = max(1, int(win_s * sr))
    xf  = max(1, int(xfade_s * sr))
    hop = max(1, win - xf)

    out = np.zeros(n, dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)
    # raised-cosine window
    if win > 1:
        t = np.arange(win, dtype=np.float32)
        win_env = 0.5 * (1 - np.cos(2 * np.pi * t / (win - 1)))
    else:
        win_env = np.ones(1, dtype=np.float32)

    i = 0
    while i < n:
        start = i
        end = min(n, i + win)
        seg = x[start:end]
        if seg.size < win:
            seg = np.pad(seg, (0, win - seg.size)).astype(np.float32)
        center = (start + end) / 2.0
        alpha = center / max(1.0, n - 1.0)
        cents_here = (1.0 - alpha) * cents_start + alpha * cents_end
        proc = detune_preserve_length(seg, cents_here)[:end - start]
        w = win_env[:proc.size]
        out[start:end] += proc * w
        wsum[start:end] += w
        i += hop

    nz = wsum > 1e-6
    out[nz] /= wsum[nz]
    out[~nz] = 0.0
    return out


def crossfade_replace_tail(buf: np.ndarray, new_tail: np.ndarray, xfade_s: float, sr: int) -> np.ndarray:
    """Crossfade into a new tail to guarantee final tuning."""
    n = buf.size
    m = new_tail.size
    if m >= n:
        return new_tail[-n:].astype(np.float32)
    xfade = max(1, int(xfade_s * sr))
    start = n - m
    xf_start = max(start, n - m - xfade)
    xf_len = min(xfade, n - xf_start, m)
    out = buf.copy()
    if xf_start > 0:
        out[:xf_start] = buf[:xf_start]
    if xf_len > 0:
        a = np.linspace(1.0, 0.0, xf_len, dtype=np.float32)
        b = 1.0 - a
        out[xf_start:xf_start+xf_len] = buf[xf_start:xf_start+xf_len] * a + new_tail[:xf_len] * b
    tail_start = xf_start + xf_len
    out[tail_start:] = new_tail[xf_len:xf_len + (n - tail_start)]
    return out.astype(np.float32)


# -------------------- Dataset helpers (choose sources) --------------
def choose_satb_instrument(root: Path, section: str, force_sustain: bool = False) -> Optional[Path]:
    sec_dir = root / section
    if not sec_dir.exists():
        return None
    if force_sustain:
        sustain_dir = sec_dir / "sustain"
        instruments = list_instruments(sustain_dir)
        if instruments:
            return random.choice(instruments)
    # fallback / normal
    sustain_dir = sec_dir / "sustain"
    instruments = list_instruments(sustain_dir)
    if not instruments:
        arts = [p for p in sec_dir.iterdir() if p.is_dir()]
        all_inst = []
        for a in arts:
            all_inst.extend(list_instruments(a))
        instruments = all_inst
    if not instruments:
        return None
    return random.choice(instruments)


def choose_melody_instrument(root: Path, force_sustain: bool = False) -> Tuple[str, Optional[Path]]:
    mel_dir = root / "Melody"
    art_choice = "sustain" if force_sustain else random.choice(["pizzicato", "sustain"])
    art_dir = mel_dir / art_choice
    instruments = list_instruments(art_dir)
    if not instruments and not force_sustain:
        other = "sustain" if art_choice == "pizzicato" else "pizzicato"
        alt_dir = mel_dir / other
        instruments = list_instruments(alt_dir)
        art_choice = other if instruments else art_choice
    if not instruments:
        return art_choice, None
    return art_choice, random.choice(instruments)


def build_layer_stream(instrument_dir: Path, target_sr: int, target_len: int) -> np.ndarray:
    files = gather_audio_files(instrument_dir)
    if not files:
        return np.zeros(target_len, dtype=np.float32)
    random.shuffle(files)
    buf_list: List[np.ndarray] = []
    total = 0
    i = 0
    while total < target_len and i < max(8, len(files) * 2):
        f = files[i % len(files)]
        arr = read_audio_mono_resampled(f, target_sr)
        if arr.size == 0:
            i += 1
            continue
        buf_list.append(arr)
        total += arr.size
        i += 1
    if not buf_list:
        return np.zeros(target_len, dtype=np.float32)
    cat = np.concatenate(buf_list, axis=0)
    if cat.size < target_len:
        reps = int(np.ceil(target_len / cat.size)) if cat.size > 0 else 1
        cat = np.tile(cat, reps)
    return cat[:target_len].astype(np.float32)


def apply_layer_envelope_in_place(layer: np.ndarray, sr: int, fade_in_s: float, fade_out_s: float) -> None:
    n = layer.size
    fi = int(max(0, fade_in_s) * sr)
    fo = int(max(0, fade_out_s) * sr)
    if fi > 0:
        fi = min(fi, n)
        layer[:fi] *= np.linspace(0.0, 1.0, num=fi, endpoint=True, dtype=np.float32)
    if fo > 0:
        fo = min(fo, n)
        tail = layer[-fo:]
        layer[-fo:] = tail * np.linspace(1.0, 0.0, num=fo, endpoint=True, dtype=np.float32)


# ------------------------- CORE RENDERER ---------------------------
def render_layers(dataset_root: Path,
                  duration_s: float,
                  master_fade_in: float,
                  master_sustain: float,
                  master_fade_out: float,
                  master_maxvol: float,
                  force_sustain: bool = False,
                  absolute_targets_hz: Optional[Dict[str, float]] = None,
                  progressive_start_cents: Optional[Dict[str, float]] = None,
                  prog_win_s: float = 0.05,
                  prog_xf_s: float = 0.006) -> np.ndarray:
    """Render with optional absolute targets and/or progressive start->target morph."""
    n = int(duration_s * SR)
    master = np.zeros(n, dtype=np.float32)

    sections = ["S", "A", "T", "B"]
    satb_instruments = {sec: choose_satb_instrument(dataset_root, sec, force_sustain=force_sustain) for sec in sections}
    mel_art, mel_inst = choose_melody_instrument(dataset_root, force_sustain=force_sustain)

    # ---- Build each layer ----
    layers: List[np.ndarray] = []
    layer_keys: List[str] = []

    for sec in sections:
        cfg = LAYER_CFG.get(sec, LayerConfig())
        inst_dir = satb_instruments[sec]
        if inst_dir is None or cfg.mute:
            layers.append(np.zeros(n, dtype=np.float32))
        else:
            buf = build_layer_stream(inst_dir, SR, n)

            # Progressive morph from start detune -> 0 cents (pre-align), if requested
            if progressive_start_cents and sec in progressive_start_cents:
                buf = progressive_detune_to_target(buf, SR, progressive_start_cents[sec], 0.0, prog_win_s, prog_xf_s)

            # Absolute alignment to target frequency, if provided
            if absolute_targets_hz and sec in absolute_targets_hz:
                est = estimate_pitch_hz(buf, SR)
                tgt = float(absolute_targets_hz[sec])
                cents = cents_between(est, tgt) if est > 0 and tgt > 0 else 0.0
                buf = detune_preserve_length(buf, cents)

                # Final tail lock
                lock_n = int(LOCK_LAST_S * SR)
                if 0 < lock_n < buf.size:
                    tail = buf[-lock_n:]
                    est_tail = estimate_pitch_hz(tail, SR)
                    cents_tail = cents_between(est_tail, tgt) if est_tail > 0 and tgt > 0 else 0.0
                    locked_tail = detune_preserve_length(tail, cents_tail)
                    buf = crossfade_replace_tail(buf, locked_tail, LOCK_XFADE_S, SR)
                    if DEBUG_PITCH:
                        print(f"[LOCK] {sec}: tail {est_tail:.1f} Hz -> {tgt:.1f} Hz (cents {cents_tail:+.1f})")

            layers.append(buf)
        layer_keys.append(sec)

    # Melody
    cfg_m = LAYER_CFG.get("Melody", LayerConfig())
    if mel_inst is None or cfg_m.mute:
        layers.append(np.zeros(n, dtype=np.float32))
    else:
        buf = build_layer_stream(mel_inst, SR, n)

        if progressive_start_cents and "Melody" in progressive_start_cents:
            buf = progressive_detune_to_target(buf, SR, progressive_start_cents["Melody"], 0.0, prog_win_s, prog_xf_s)

        if absolute_targets_hz and "Melody" in absolute_targets_hz:
            est = estimate_pitch_hz(buf, SR)
            tgt = float(absolute_targets_hz["Melody"])
            cents = cents_between(est, tgt) if est > 0 and tgt > 0 else 0.0
            buf = detune_preserve_length(buf, cents)

            lock_n = int(LOCK_LAST_S * SR)
            if 0 < lock_n < buf.size:
                tail = buf[-lock_n:]
                est_tail = estimate_pitch_hz(tail, SR)
                cents_tail = cents_between(est_tail, tgt) if est_tail > 0 and tgt > 0 else 0.0
                locked_tail = detune_preserve_length(tail, cents_tail)
                buf = crossfade_replace_tail(buf, locked_tail, LOCK_XFADE_S, SR)
                if DEBUG_PITCH:
                    print(f"[LOCK] Melody: tail {est_tail:.1f} Hz -> {tgt:.1f} Hz (cents {cents_tail:+.1f})")

        layers.append(buf)
    layer_keys.append("Melody")

    # ---- Entry order & offsets ----
    order = list(range(len(layers)))
    random.shuffle(order)
    first = order[0]
    offsets = [0] * len(layers)
    offsets[first] = 0
    for idx in order[1:]:
        off_s = random.uniform(ENTRY_OFFSET_MIN_S, ENTRY_OFFSET_MAX_S)
        offsets[idx] = int(min(n - 1, off_s * SR))

    # ---- Mix with per-layer envelopes & gains ----
    for i, layer in enumerate(layers):
        off = offsets[i]
        if off >= n:
            continue
        remain = n - off
        src = layer[:remain].copy()
        key = layer_keys[i]
        cfg = LAYER_CFG.get(key, LayerConfig())
        apply_layer_envelope_in_place(src, SR, cfg.fade_in_s, cfg.fade_out_s)
        src *= float(cfg.gain)
        master[off:off+remain] += src

    # Normalize pre-master
    peak = np.max(np.abs(master)) if master.size > 0 else 1.0
    if peak > 1.0:
        master = master / peak

    # Apply master envelope
    env = intro_outro_envelope(duration_s, SR, master_fade_in, master_sustain, master_fade_out, master_maxvol)
    master *= env

    # Safety headroom
    peak2 = np.max(np.abs(master)) if master.size > 0 else 1.0
    if peak2 > 0.99:
        master = master / (peak2 / 0.98)

    return master


# ------------------------------ Targets ----------------------------
def target_map_for_voicing(root_note: str, chord_type: str) -> Dict[str, float]:
    root_midi = note_to_midi(root_note)
    if chord_type.lower() == "minor":
        st = HARMONY_VOICING_MINOR_ST
    else:
        st = HARMONY_VOICING_MAJOR_ST
    return {k: midi_to_hz(root_midi + int(st.get(k, 0))) for k in ["B","T","A","S","Melody"]}


# -------------------------------- API ------------------------------

def build_intro_outro_buffer(dataset_root: str | Path) -> np.ndarray:
    duration_s = MASTER_FADE_IN_S + MASTER_SUSTAIN_S + MASTER_FADE_OUT_S
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    mix = render_layers(
        root,
        duration_s,
        MASTER_FADE_IN_S, MASTER_SUSTAIN_S, MASTER_FADE_OUT_S, MASTER_MAXVOL,
        force_sustain=False,
    )
    return mix.astype(np.float32)


def build_harmony_buffer(dataset_root: str | Path) -> np.ndarray:
    duration_s = HARMONY_MASTER_FADE_IN_S + HARMONY_MASTER_SUSTAIN_S + HARMONY_MASTER_FADE_OUT_S
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    targets_hz = target_map_for_voicing(HARMONY_ROOT, HARMONY_CHORD_TYPE)
    mix = render_layers(
        root,
        duration_s,
        HARMONY_MASTER_FADE_IN_S, HARMONY_MASTER_SUSTAIN_S, HARMONY_MASTER_FADE_OUT_S, HARMONY_MASTER_MAXVOL,
        force_sustain=FORCE_SUSTAIN_FOR_HARMONY,
        absolute_targets_hz=targets_hz,
    )
    return mix.astype(np.float32)


def build_tune2harmony_buffer(dataset_root: str | Path) -> np.ndarray:
    duration_s = TUNE2HARM_MASTER_FADE_IN_S + TUNE2HARM_MASTER_SUSTAIN_S + TUNE2HARM_MASTER_FADE_OUT_S
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    targets_hz = target_map_for_voicing(HARMONY_ROOT, HARMONY_CHORD_TYPE)
    start_map: Dict[str, float] = {k: random.uniform(*TUNE_START_DETUNE_RANGE_CENTS) for k in ["S","A","T","B","Melody"]}

    mix = render_layers(
        root,
        duration_s,
        TUNE2HARM_MASTER_FADE_IN_S, TUNE2HARM_MASTER_SUSTAIN_S, TUNE2HARM_MASTER_FADE_OUT_S, TUNE2HARM_MASTER_MAXVOL,
        force_sustain=FORCE_SUSTAIN_FOR_TUNE2HARM,
        absolute_targets_hz=targets_hz,
        progressive_start_cents=start_map,
        prog_win_s=TUNE2HARM_WINDOW_S,
        prog_xf_s=TUNE2HARM_XFADE_S,
    )
    return mix.astype(np.float32)


def build_intro_outro_buffer(dataset_root: str | Path) -> np.ndarray:
    duration_s = MASTER_FADE_IN_S + MASTER_SUSTAIN_S + MASTER_FADE_OUT_S
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    mix = render_layers(
        root,
        duration_s,
        MASTER_FADE_IN_S, MASTER_SUSTAIN_S, MASTER_FADE_OUT_S, MASTER_MAXVOL,
        force_sustain=False,
    )
    return mix.astype(np.float32)


def trigger_intro_outro(dataset_root: str | Path) -> None:
    mix = build_intro_outro_buffer(dataset_root)
    sd.play(mix, samplerate=SR, blocking=False)
    sd.sleep(int(mix.shape[0] / SR * 1000 + 500))


def trigger_harmony(dataset_root: str | Path) -> None:
    mix = build_harmony_buffer(dataset_root)
    sd.play(mix, samplerate=SR, blocking=False)
    sd.sleep(int(mix.shape[0] / SR * 1000 + 500))


def trigger_tune2harmony(dataset_root: str | Path) -> None:
    mix = build_tune2harmony_buffer(dataset_root)
    sd.play(mix, samplerate=SR, blocking=False)
    sd.sleep(int(mix.shape[0] / SR * 1000 + 500))

def play_orchestra(
    dataset_root: str | Path,
    mode: str = "mix",
    fade_in_s: float = 1.0,
    sustain_s: float = 2.0,
    fade_out_s: float = 2.0,
    max_volume: float = 0.8,
    root_note: str = "C4",
    chord_type: str = "major",
    layer_gains: Optional[Dict[str, float]] = None,
    layer_mutes: Optional[Dict[str, bool]] = None,
    lock_last_s: Optional[float] = None,
    debug_pitch: Optional[bool] = None,
) -> None:
    """
    Programmatic entry point so the whole orchestral engine can be treated
    as a single "layer" by an external main script.
    """

    # Use the globals already defined at the top of this file
    global MASTER_FADE_IN_S, MASTER_SUSTAIN_S, MASTER_FADE_OUT_S, MASTER_MAXVOL
    global HARMONY_MASTER_FADE_IN_S, HARMONY_MASTER_SUSTAIN_S, HARMONY_MASTER_FADE_OUT_S, HARMONY_MASTER_MAXVOL
    global TUNE2HARM_MASTER_FADE_IN_S, TUNE2HARM_MASTER_SUSTAIN_S, TUNE2HARM_MASTER_FADE_OUT_S, TUNE2HARM_MASTER_MAXVOL
    global HARMONY_ROOT, HARMONY_CHORD_TYPE
    global LOCK_LAST_S, DEBUG_PITCH

    # 1) Update master envelope from arguments
    MASTER_FADE_IN_S  = float(fade_in_s)
    MASTER_SUSTAIN_S  = float(sustain_s)
    MASTER_FADE_OUT_S = float(fade_out_s)
    MASTER_MAXVOL     = float(max_volume)

    # Keep harmony & tune2harmony in sync with the same envelope
    HARMONY_MASTER_FADE_IN_S  = MASTER_FADE_IN_S
    HARMONY_MASTER_SUSTAIN_S  = MASTER_SUSTAIN_S
    HARMONY_MASTER_FADE_OUT_S = MASTER_FADE_OUT_S
    HARMONY_MASTER_MAXVOL     = MASTER_MAXVOL

    TUNE2HARM_MASTER_FADE_IN_S  = MASTER_FADE_IN_S
    TUNE2HARM_MASTER_SUSTAIN_S  = MASTER_SUSTAIN_S
    TUNE2HARM_MASTER_FADE_OUT_S = MASTER_FADE_OUT_S
    TUNE2HARM_MASTER_MAXVOL     = MASTER_MAXVOL

    # 2) Update harmony target
    HARMONY_ROOT = root_note
    HARMONY_CHORD_TYPE = chord_type

    # 3) Optional: update lock + debug
    if lock_last_s is not None:
        LOCK_LAST_S = float(lock_last_s)
    if debug_pitch is not None:
        DEBUG_PITCH = bool(debug_pitch)

    # 4) Optional: per-section gains/mutes (you *can* ignore these in YAML if not needed)
    if layer_gains:
        for k, g in layer_gains.items():
            if k in LAYER_CFG:
                LAYER_CFG[k].gain = float(g)

    if layer_mutes:
        for k, m in layer_mutes.items():
            if k in LAYER_CFG:
                LAYER_CFG[k].mute = bool(m)

    # 5) Call the existing triggers (same ones you use from CLI)
    if mode == "mix":
        trigger_intro_outro(dataset_root)
    elif mode == "harmony":
        trigger_harmony(dataset_root)
    elif mode == "tune2harmony":
        trigger_tune2harmony(dataset_root)
    else:
        raise ValueError(f"Unknown mode: {mode}")




# --------------------------------- CLI -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hardcoded orchestral player (absolute harmony + final lock).")
    ap.add_argument("dataset_root", type=str, help="Path to dataset root")
    ap.add_argument("--mode", type=str, default="mix", choices=["mix", "harmony", "tune2harmony"], help="Play mode")
    args = ap.parse_args()

    if args.mode == "mix":
        trigger_intro_outro(args.dataset_root)
    elif args.mode == "harmony":
        trigger_harmony(args.dataset_root)
    else:
        trigger_tune2harmony(args.dataset_root)
