# -*- coding: utf-8 -*-
"""
Messiaen AudioLayer for sounddevice central mixer.

Public contract used by your main:
- create_layer(sample_rate=...)
- class implements render(frames)->(frames,2) float32
- set_base_gain(gain, state="")
- debug_print(base_gain)

CSV schema expected (VariantIndex_portable.csv):
  variant_id, path, segment_id, entry_anchor_ms, mid_anchor_ms, exit_anchor_ms,
  tempo_factor, pitch_semitones, state_suggestion, phase_affinity,
  calm_final, centroid, rms

`path` is usually just the filename (e.g., seg_0001__t00__p0.wav)
living under VARIANTS_ROOT/Variants/.
"""

import os, math, csv, time, threading, random
from typing import Optional, List

import numpy as np
import soundfile as sf  # pip install soundfile

# --------------------- small utils ---------------------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def eqpow(a_lin: float) -> float:
    """Equal-power amplitude curve."""
    return math.sqrt(clamp(float(a_lin), 0.0, 1.0))

# --------------------- config defaults ---------------------

DEFAULTS = dict(
    MEAS_PATH = "sonification.txt",  # file with: time, br, std, presence, moon
    VARIANTS_ROOT = os.path.join("assets", "CalmPool"),  # contains "Variants/.."
    CSV_PATH = os.path.join("assets", "CalmPool", "Metadata", "VariantIndex_portable.csv"),
    CALM_MIN = 0.70,             # only load calm variants ≥ this
    CROSSFADE_MS = 650,          # AB crossfade time
    BASE_DWELL_S = 8.0,          # min time before switching at an anchor
    GAIN_SLEW_S = 0.05,          # slew for main-cap smoothing
    SEED = None,                 # set an int for reproducibility
)

# Allow env overrides without touching code
MEAS_PATH      = os.getenv("MESS_MEAS_PATH", DEFAULTS["MEAS_PATH"])
CSV_PATH       = os.getenv("MESS_CSV", DEFAULTS["CSV_PATH"])
VARIANTS_ROOT  = os.getenv("MESS_ROOT", DEFAULTS["VARIANTS_ROOT"])
CALM_MIN       = float(os.getenv("MESS_CALM_MIN", DEFAULTS["CALM_MIN"]))
CROSSFADE_MS   = int(os.getenv("MESS_XFADE_MS", DEFAULTS["CROSSFADE_MS"]))
BASE_DWELL_S   = float(os.getenv("MESS_BASE_DWELL_S", DEFAULTS["BASE_DWELL_S"]))
GAIN_SLEW_S    = float(os.getenv("MESS_GAIN_SLEW_S", DEFAULTS["GAIN_SLEW_S"]))
SEED           = os.getenv("MESS_SEED", DEFAULTS["SEED"])

# --------------------- data structures ---------------------

class Variant:
    __slots__ = ("path_abs","family","anchors","dur_s","calm_score","tempo_hint")
    def __init__(self, path_abs: str, family: str, anchors: List[float],
                 dur_s: float, calm_score: float, tempo_hint: Optional[float]):
        self.path_abs = path_abs
        self.family = family              # inhale|exhale|either
        self.anchors = anchors            # fractions [0..1], sorted
        self.dur_s = float(dur_s)
        self.calm_score = float(calm_score)
        self.tempo_hint = tempo_hint

class PhaseClock:
    """Simple inhale/exhale phase clock that keeps ticking sample-accurately."""
    def __init__(self, sr: int):
        self.sr = sr
        self.pos = 0.0
        self.phase = "exhale"
        self._last_br = 10.0
    def tick(self, samples: int, br_used: float):
        br = clamp(br_used if br_used > 0 else self._last_br, 4.0, 35.0)
        self._last_br = br
        period_s = 60.0 / br
        step = (samples / self.sr) / period_s
        self.pos = (self.pos + step) % 1.0
        self.phase = "inhale" if self.pos < 0.4 else "exhale"
        return self.phase

# --------------------- main AudioLayer ---------------------

class MessiaenLayer:
    """
    Pull-based AudioLayer for sounddevice.
    - Two disk readers for AB crossfades.
    - Selection driven by breathe file tailer (1 Hz): br_used, std, presence.
    - Audibility is controlled by main via set_base_gain() -> this is a CAP.
    """

    # ---------- construction ----------

    def __init__(self, sample_rate: int):
        if SEED is not None:
            try: random.seed(int(SEED))
            except: pass

        self.sr = int(sample_rate)

        # gain cap from main (slewed)
        self._cap = 0.0
        self._cap_target = 0.0
        self._cap_step = 1.0 / max(1, int(GAIN_SLEW_S * self.sr))

        # internal musical intensity (0..1), independent of main cap
        self._inten = 1.0
        self._inten_t0 = time.perf_counter()
        self._inten_t1 = self._inten_t0
        self._inten_v0 = self._inten
        self._inten_v1 = self._inten

        # Breath state (shared with tailer thread)
        self._lock = threading.Lock()
        self.presence = 0
        self.std = 0.2
        self.br_used = 10.0
        self._epi = 1.0
        self._last_presence = 0

        # Readers A/B
        self._rA = None  # type: Optional[sf.SoundFile]
        self._rB = None  # type: Optional[sf.SoundFile]
        self._slot = 0   # 0 means A is current, 1 means B is current
        self._posA = 0
        self._posB = 0
        self._xfade_samps = int(self.sr * (CROSSFADE_MS / 1000.0))
        self._xfade_rem = 0

        # Current variant and anchors
        self._cur: Optional[Variant] = None
        self._cur_played_samps = 0
        self._next_anchor_idx = 0     # set when we actually pick a Variant

        # Pool
        self._pool: List[Variant] = []
        self._by_fam = {"inhale": [], "exhale": [], "either": []}
        self._load_pool()

        # Phase clock (sample-accurate)
        self._pc = PhaseClock(self.sr)

        # Tailer thread (1 Hz)
        self._stop = False
        self._tailer = threading.Thread(target=self._tailer_loop, daemon=True)
        self._tailer.start()

    # ---------- public API expected by main ----------

    def set_base_gain(self, gain: float, state: str = ""):
        """Main sets the *maximum* allowed gain. We slew toward it to avoid zipper noise."""
        self._cap_target = clamp(gain, 0.0, 1.0)

    def render(self, frames: int) -> np.ndarray:
        """Return (frames,2) float32 buffer mixed from current readers, with internal crossfades."""
        out = np.zeros((frames, 2), dtype=np.float32)
        """
        # 1) slew the main cap toward its target
        tgt_cap = float(self._cap_target)
        if abs(tgt_cap - self._cap) > self._cap_step:
            self._cap += self._cap_step if tgt_cap > self._cap else -self._cap_step
        else:
            self._cap = tgt_cap

        # 2) step the layer’s own internal intensity envelope
        inten_shaped = self._step_intensity()  # 0..1 with equal-power easing inside

        # 3) FINAL GAIN = CAP × INTENSITY  (strict cap; never exceeds the main’s value)
        layer_gain = float(self._cap) * float(inten_shaped)
        """
        # 1) slew the main cap toward its target
        tgt_cap = float(self._cap_target)
        if abs(tgt_cap - self._cap) > self._cap_step:
            self._cap += self._cap_step if tgt_cap > self._cap else -self._cap_step
        else:
            self._cap = tgt_cap
        # 1) follow the main program’s cap exactly (main already interpolates over FADE_DUR)
        self._cap = float(self._cap_target)
        inten_shaped = self._step_intensity()    # keep your musical envelope
        layer_gain   = float(self._cap) * float(inten_shaped)  # never exceeds the main’s cap


        remaining = frames
        write_pos = 0

        while remaining > 0:
            # Ensure we have a current file loaded
            if self._cur is None or (self._rA is None and self._rB is None):
                self._start_next_variant()

            # How many samples can we pull this sub-iteration
            chunk = remaining

            if self._slot == 0:
                # A is current
                a = self._read_into(self._rA, chunk)
                if a is None:
                    self._start_next_variant(force=True)
                    continue
                if self._xfade_rem > 0 and self._rB is not None:
                    b = self._read_into(self._rB, chunk)
                    if b is None:
                        b = np.zeros_like(a)
                    # equal-power crossfade envelope across this chunk
                    take = min(self._xfade_rem, a.shape[0], b.shape[0])
                    if take > 0:
                        env = self._xfade_env(take)[:, None]  # shape (take, 1) to match stereo
                        a[:take, :] = a[:take, :] * (1.0 - env) + b[:take, :] * env

                    self._xfade_rem -= take
                    if self._xfade_rem <= 0:
                        self._close_A()
                        self._slot = 1

                    self._xfade_rem -= take
                    if self._xfade_rem <= 0:
                        self._close_A()
                        self._slot = 1
                out[write_pos:write_pos+a.shape[0], :] += a * layer_gain
                n = a.shape[0]
                write_pos += n
                remaining -= n
                self._cur_played_samps += n
            else:
                # B is current
                a = self._read_into(self._rB, chunk)
                if a is None:
                    self._start_next_variant(force=True)
                    continue
                if self._xfade_rem > 0 and self._rA is not None:
                    b = self._read_into(self._rA, chunk)
                    if b is None:
                        b = np.zeros_like(a)
                    take = min(self._xfade_rem, a.shape[0], b.shape[0])
                    if take > 0:
                        env = self._xfade_env(take)[:, None]
                        a[:take, :] = a[:take, :] * (1.0 - env) + b[:take, :] * env
                    self._xfade_rem -= take
                    if self._xfade_rem <= 0:
                        self._close_B()
                        self._slot = 0

                    self._xfade_rem -= take
                    if self._xfade_rem <= 0:
                        self._close_B()
                        self._slot = 0
                out[write_pos:write_pos+a.shape[0], :] += a * layer_gain
                n = a.shape[0]
                write_pos += n
                remaining -= n
                self._cur_played_samps += n

            # Check anchors / dwell to see if we should schedule a switch
            self._maybe_schedule_switch(frames=n)

        return out

    def debug_print(self, base_gain: float):
        cur_name = os.path.basename(self._cur.path_abs) if self._cur else "-"
        prog = 0.0
        if self._cur:
            total = max(1, int(self._cur.dur_s * self.sr))
            prog = self._cur_played_samps / total
        readers = []
        readers.append(f"A:{'on' if self._rA is not None else 'off'}")
        readers.append(f"B:{'on' if self._rB is not None else 'off'}")
        xf = self._xfade_rem / max(1, self._xfade_samps) if self._xfade_samps > 0 else 0.0
        print(
            f"[messiaen] base={base_gain:.2f} cap_now={self._cap:.2f} inten={self._inten:.2f} "
            f"slot={'A' if self._slot==0 else 'B'} readers={','.join(readers)} xfade={xf:.2f} "
            f"pool={len(self._pool)} playing={cur_name} prog={prog:.2f} "
            f"pres={self.presence} br={self.br_used:.1f} std={self.std:.2f} phase={self._pc.phase}"
        )

    # ---------- intensity helpers ----------

    def _set_intensity(self, tgt: float, dur_s: float):
        now = time.perf_counter()
        self._inten_t0 = now
        self._inten_t1 = now + max(0.001, float(dur_s))
        self._inten_v0 = self._inten
        self._inten_v1 = clamp(tgt, 0.0, 1.0)

    def _step_intensity(self) -> float:
        now = time.perf_counter()
        if now >= self._inten_t1:
            self._inten = self._inten_v1
            return self._inten
        x = (now - self._inten_t0) / (self._inten_t1 - self._inten_t0)
        shaped = 0.5 - 0.5 * math.cos(math.pi * x)  # equal-power (cosine)
        self._inten = self._inten_v0 + (self._inten_v1 - self._inten_v0) * shaped
        return self._inten

    # ---------- pool loading / selection ----------

    def _load_pool(self):
        """
        Loads variants from CSV with columns:
          path, calm_final, entry_anchor_ms, mid_anchor_ms, exit_anchor_ms,
          phase_affinity (inhale|exhale|either), tempo_factor
        `path` can be just a filename under VARIANTS_ROOT/Variants/.
        """
        base_dir = os.path.dirname(os.path.abspath(CSV_PATH))
        root_dir = os.path.abspath(VARIANTS_ROOT)

        def resolve_path(p: str) -> Optional[str]:
            """Robust file resolver for filename-only paths under Variants/."""
            if not p:
                return None
            p = str(p).strip().strip('"').strip("'")
            if not p:
                return None
            # absolute?
            if os.path.isabs(p) and os.path.isfile(p):
                return p
            fname = p.replace("\\", "/").split("/")[-1]
            candidates = [
                os.path.join(root_dir, "Variants", fname),
                os.path.join(root_dir, fname),
                os.path.join(base_dir, "Variants", fname),
                os.path.join(base_dir, fname),
            ]
            for cand in candidates:
                cand = os.path.normpath(cand)
                if os.path.isfile(cand):
                    return cand
            return None

        def ms_list_to_fractions(anchors_ms: list, dur_s: float) -> list:
            if dur_s <= 0:
                return [0.0, 0.5, 0.9]
            out = []
            for v in anchors_ms:
                try:
                    ms = float(v)
                    if ms <= 0: continue
                    frac = ms / (dur_s * 1000.0)
                    if 0.0 <= frac <= 1.0:
                        out.append(frac)
                except:
                    pass
            out = sorted(set(out))
            return out or [0.0, 0.5, 0.9]

        total = kept = below = missing = bad = 0
        self._pool = []
        self._by_fam = {"inhale": [], "exhale": [], "either": []}

        if not os.path.isfile(CSV_PATH):
            raise FileNotFoundError(f"[Messiaen] CSV not found: {CSV_PATH}")

        with open(CSV_PATH, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                total += 1
                try:
                    calm = row.get("calm_final", "")
                    calm = float(calm) if str(calm).strip() != "" else 0.0
                    if calm < CALM_MIN:
                        below += 1
                        continue

                    path_field = row.get("path", "")
                    path_abs = resolve_path(path_field)
                    if not path_abs:
                        missing += 1
                        continue

                    # duration
                    try:
                        with sf.SoundFile(path_abs) as fp:
                            dur_s = len(fp) / float(fp.samplerate)
                    except:
                        dur_s = 10.0

                    anchors_ms = [
                        row.get("entry_anchor_ms", ""),
                        row.get("mid_anchor_ms", ""),
                        row.get("exit_anchor_ms", ""),
                    ]
                    anchors = ms_list_to_fractions(anchors_ms, dur_s)

                    fam = (row.get("phase_affinity") or "either").strip().lower()
                    if fam not in ("inhale", "exhale", "either"):
                        fam = "either"

                    tempo_hint = None
                    if (row.get("tempo_factor") or "").strip():
                        try: tempo_hint = float(row["tempo_factor"])
                        except: tempo_hint = None

                    v = Variant(
                        path_abs=path_abs,
                        family=fam,
                        anchors=anchors,
                        dur_s=dur_s,
                        calm_score=calm,
                        tempo_hint=tempo_hint,
                    )
                    self._pool.append(v)
                    self._by_fam[fam].append(v)
                    kept += 1
                except Exception:
                    bad += 1
                    continue

        print(
            f"[Messiaen] pool stats: total={total} kept={kept} "
            f"below_thr={below} missing_files={missing} bad_rows={bad} (thr={CALM_MIN})"
        )
        print(f"[Messiaen] loaded {kept} calm variants (≥ {CALM_MIN}).")

    def _pick_family_pool(self, phase: str) -> List[Variant]:
        if phase == "inhale" and self._by_fam["inhale"]: return self._by_fam["inhale"]
        if phase == "exhale" and self._by_fam["exhale"]: return self._by_fam["exhale"]
        return (self._by_fam["either"] or []) + self._by_fam["inhale"] + self._by_fam["exhale"]

    def _pick_next(self, phase: str) -> Optional[Variant]:
        cand = self._pick_family_pool(phase) or self._pool
        if not cand: return None
        if self.br_used < 12.0:
            pref = [v for v in cand if (v.tempo_hint or 1.0) <= 1.0]
            cand = pref or cand
        return random.choice(cand)

    # ---------- reading / crossfading ----------

    def _open_reader(self, path: str) -> Optional[sf.SoundFile]:
        try:
            return sf.SoundFile(path, mode="r")
        except Exception:
            return None

    def _read_into(self, reader: Optional[sf.SoundFile], frames: int) -> Optional[np.ndarray]:
        if reader is None:
            return None
        try:
            data = reader.read(frames, dtype="float32", always_2d=True)
            if data.size == 0:
                return None
            if data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)
            elif data.shape[1] > 2:
                data = data[:, :2]
            return data
        except Exception:
            return None

    def _xfade_env(self, n: int) -> np.ndarray:
        x = np.linspace(0.0, 1.0, num=max(1, n), endpoint=False, dtype=np.float32)
        return (np.sin(0.5 * np.pi * x) ** 2).astype(np.float32)

    def _close_A(self):
        if self._rA is not None:
            try: self._rA.close()
            except: pass
            self._rA = None
            self._posA = 0

    def _close_B(self):
        if self._rB is not None:
            try: self._rB.close()
            except: pass
            self._rB = None
            self._posB = 0

    def _start_next_variant(self, force: bool=False):
        new_slot = 1 - self._slot
        phase = self._pc.phase
        v = self._pick_next(phase)
        if not v:
            self._close_A(); self._close_B()
            self._cur = None
            self._xfade_rem = 0
            return

        r = self._open_reader(v.path_abs)
        if r is None:
            return

        if new_slot == 0:
            self._close_A()
            self._rA = r
            self._posA = 0
        else:
            self._close_B()
            self._rB = r
            self._posB = 0

        both_ready = (self._rA is not None and self._rB is not None)
        if both_ready and (self._cur is not None) and not force:
            self._xfade_rem = self._xfade_samps
        else:
            self._xfade_rem = 0
            self._slot = new_slot

        self._cur = v
        self._cur_played_samps = 0
        # Skip anchor at 0.0 if present
        self._next_anchor_idx = 1 if len(v.anchors) > 1 else 0

    # ---------- selection timing ----------

    def _maybe_schedule_switch(self, frames: int):
        if self._cur is None:
            return

        # advance the internal phase clock
        self._pc.tick(frames, self.br_used)

        # dwell time shaped by STD (steady → longer)
        s = clamp(self.std, 0.0, 1.0)
        dwell_factor = 1.4 - 0.6 * s  # 1.4..0.8
        dwell_needed = int(self.sr * BASE_DWELL_S * dwell_factor)

        # progress within current file
        dur_samps = int(max(1, self._cur.dur_s * self.sr))
        prog = self._cur_played_samps / dur_samps

        # anchor readiness
        ready_anchor = False
        if self._next_anchor_idx < len(self._cur.anchors):
            if prog >= self._cur.anchors[self._next_anchor_idx]:
                ready_anchor = True

        at_end = prog >= 0.999

        if (self._cur_played_samps >= dwell_needed and ready_anchor) or at_end:
            self._start_next_variant()

    # ---------- 1 Hz tailer ----------

    def _tailer_loop(self):
        last_line = None
        epi = 1.0
        while not self._stop:
            try:
                with open(MEAS_PATH, "r", encoding="utf-8") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
            except FileNotFoundError:
                time.sleep(1.0); continue

            if not lines:
                time.sleep(1.0); continue

            line = lines[-1]
            if line != last_line:
                last_line = line
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    try:
                        br = float(parts[1]); std = float(parts[2]); presence = int(parts[3])
                    except:
                        br, std, presence = 10.0, 0.2, 0

                    if presence == 1 and self._last_presence == 0:
                        epi = random.uniform(1.5, 2.0)
                    elif presence == 0 and self._last_presence == 1:
                        epi = 1.0

                    br_used = clamp(br * epi, 4.0, 35.0)

                    with self._lock:
                        self.presence = presence
                        self.std = clamp(std, 0.0, 1.0)
                        self.br_used = br_used
                        self._epi = epi
                        self._last_presence = presence
            time.sleep(1.0)

    # ---------- teardown ----------

    def close(self):
        self._stop = True
        try:
            self._tailer.join(timeout=1.0)
        except:
            pass
        self._close_A(); self._close_B()

# --------------- factory for your main ----------------

def create_layer(sample_rate: int = 44100):
    return MessiaenLayer(sample_rate=sample_rate)

