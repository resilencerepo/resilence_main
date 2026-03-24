# -*- coding: utf-8 -*-
# layers/breathing_layer.py
"""
BreathingLayer (multi-sublayer version)
- All four bins play simultaneously.
- The *active* bin (chosen by BR) fades up to internal gain=1.0;
  the other bins fade down to 0.0.
- No BR->volume mapping beyond selecting the active bin.
- Main base_gain (cap) from the app is the overall ceiling.
- Per-file equal-power fade-in/out + seamless auto-advance per bin.

Expected layout:
audio/
  breathing/
    BIN1/*.wav
    BIN2/*.wav
    BIN3/*.wav
    BIN4/*.wav
"""

import os, math, random
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

from layers.loop_layer import AudioLayer


# ---------- utilities ----------

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x
def eqpow(a): return math.sqrt(clamp(float(a), 0.0, 1.0))  # equal-power amplitude


# ---------- config ----------

AUDIO_ROOT = os.path.join("audio", "breathing")

# bin splits (bpm) + hysteresis margin to reduce flapping
SPLIT_12 = 11.0
SPLIT_23 = 14.0
SPLIT_34 = 18.0
HYST = 0.7  # set 0.0 for no hysteresis

# fades & scheduling
CROSSFADE_MS   = 250   # between successive files inside *each bin*
FADE_IN_MS     = 200   # per-file
FADE_OUT_MS    = 800   # per-file tail
SCHEDULE_AHEAD = 1.0   # sec before end to queue next file

# smoothing toward main cap
CAP_SLEW_S = 0.10

# smoothing for per-bin internal gain (to/from 1.0)
BIN_GAIN_SLEW_S = 0.35

# reproducibility (None = random)
SEED = None


# ---------- hysteretic bin selector ----------

class HystereticBins:
    def __init__(self, s12, s23, s34, margin):
        self.s12, self.s23, self.s34, self.m = float(s12), float(s23), float(s34), float(margin)
        self.current = "BIN2"  # reasonable default mid-band
    def update(self, br: float) -> str:
        b = self.current
        s12, s23, s34, m = self.s12, self.s23, self.s34, self.m
        if b == "BIN1":
            if br >= s12 + m: b = "BIN2"
        elif b == "BIN2":
            if br <  s12 - m: b = "BIN1"
            elif br >= s23 + m: b = "BIN3"
        elif b == "BIN3":
            if br <  s23 - m: b = "BIN2"
            elif br >= s34 + m: b = "BIN4"
        else:  # BIN4
            if br <  s34 - m: b = "BIN3"
        self.current = b
        return b


# ---------- one bin voice (loops its own folder) ----------

class BinVoice:
    def __init__(self, bin_name: str, sr: int, files: List[str]):
        self.name = bin_name
        self.sr = int(sr)
        self.files = files[:]  # list of wav paths
        self.idx = 0

        self.r_cur: Optional[sf.SoundFile] = None
        self.r_new: Optional[sf.SoundFile] = None
        self.len_cur = 0
        self.pos_cur = 0
        self.len_new = 0
        self.pos_new = 0

        self.fade_in = int(self.sr * (FADE_IN_MS/1000.0))
        self.fade_out = int(self.sr * (FADE_OUT_MS/1000.0))
        self.sched_before_end = int(self.sr * SCHEDULE_AHEAD)
        self.xfade = int(self.sr * (CROSSFADE_MS/1000.0))
        self.xfade_rem = 0

        # internal per-bin gain (we’ll smooth toward target)
        self.g_now = 0.0
        self.g_tgt = 0.0
        self.g_step = 1.0 / max(1, int(BIN_GAIN_SLEW_S * self.sr))

        # start immediately if any content
        self._ensure_reader(force=True)

    def file_count(self) -> int:
        return len(self.files)

    # public API per render tick
    def set_target(self, to_one: bool):
        self.g_tgt = 1.0 if to_one else 0.0

    def render(self, frames: int) -> np.ndarray:
        out = np.zeros((frames, 2), dtype=np.float32)

        # smooth bin gain toward target
        if abs(self.g_tgt - self.g_now) > self.g_step:
            self.g_now += self.g_step if self.g_tgt > self.g_now else -self.g_step
        else:
            self.g_now = self.g_tgt

        if self.r_cur is None and self.r_new is None:
            # try to start something if empty
            self._ensure_reader(force=True)
            if self.r_cur is None and self.r_new is None:
                return out  # no files present

        remaining = frames
        write = 0

        while remaining > 0:
            n = remaining

            # ensure we have a current reader
            if self.r_cur is None:
                # promote new to current if possible
                if self.r_new is not None:
                    self.r_cur, self.len_cur, self.pos_cur = self.r_new, self.len_new, self.pos_new
                    self.r_new = None; self.len_new = 0; self.pos_new = 0
                    self.xfade_rem = 0
                else:
                    self._ensure_reader(force=True)
                    if self.r_cur is None:
                        # still nothing
                        break

            a = self._read_with_env(self.r_cur, self.pos_cur, self.len_cur, n)
            if a is None:
                # hit EOF unexpectedly -> rotate immediately
                self._queue_next(force=True)
                continue
            self.pos_cur += a.shape[0]

            # crossfade with r_new if we’re in transition
            if self.xfade_rem > 0 and self.r_new is not None:
                b = self._read_with_env(self.r_new, self.pos_new, self.len_new, a.shape[0])
                if b is None:
                    b = np.zeros_like(a)
                take = min(self.xfade_rem, a.shape[0])
                env = self._xfade_env(take)[:, None]
                a[:take, :] = a[:take, :] * (1.0 - env) + b[:take, :] * env
                self.pos_new += take
                self.xfade_rem -= take
                if self.xfade_rem <= 0:
                    # new becomes current
                    if self.r_cur is not None:
                        try: self.r_cur.close()
                        except: pass
                    self.r_cur, self.len_cur, self.pos_cur = self.r_new, self.len_new, self.pos_new
                    self.r_new = None; self.len_new = 0; self.pos_new = 0

            out[write:write+a.shape[0], :] += a
            write += a.shape[0]
            remaining -= a.shape[0]

            # schedule next near tail
            self._maybe_queue_next()

        # apply *bin* internal gain (equal-power)
        out *= eqpow(self.g_now)
        return out

    # internals

    def _pick_next_path(self) -> Optional[str]:
        if not self.files:
            return None
        p = self.files[self.idx % len(self.files)]
        self.idx += 1
        return p

    def _open_reader(self, path: str) -> Optional[sf.SoundFile]:
        try:
            f = sf.SoundFile(path, mode="r")
            if f.samplerate != self.sr:
                f.close(); return None
            return f
        except Exception:
            return None

    def _ensure_reader(self, force: bool):
        if self.r_cur is None:
            path = self._pick_next_path()
            if not path: return
            r = self._open_reader(path)
            if r is None: return
            self.r_cur = r
            self.len_cur = len(r)
            self.pos_cur = 0

    def _queue_next(self, force: bool):
        # open next into r_new, schedule crossfade if we have both
        path = self._pick_next_path()
        if not path: return
        r = self._open_reader(path)
        if r is None: return
        self.r_new = r
        self.len_new = len(r)
        self.pos_new = 0
        if self.r_cur is not None and not force:
            self.xfade_rem = self.xfade
        else:
            # hard switch immediately
            if self.r_cur is not None:
                try: self.r_cur.close()
                except: pass
            self.r_cur, self.len_cur, self.pos_cur = self.r_new, self.len_new, self.pos_new
            self.r_new = None; self.len_new = 0; self.pos_new = 0
            self.xfade_rem = 0

    def _maybe_queue_next(self):
        if self.r_cur is None: return
        # schedule a new file a bit before the end (or to guarantee fade-out tail)
        remain = max(0, self.len_cur - self.pos_cur)
        tail_need = max(self.sched_before_end, self.fade_out)
        if remain <= tail_need and self.r_new is None:
            self._queue_next(force=False)

    def _read_with_env(self, reader: sf.SoundFile, pos: int, total: int, frames: int) -> Optional[np.ndarray]:
        try:
            data = reader.read(frames, dtype="float32", always_2d=True)
        except Exception:
            return None
        if data.size == 0:
            return None
        # up/down-mix stereo
        if data.shape[1] == 1:
            data = np.repeat(data, 2, axis=1)
        elif data.shape[1] > 2:
            data = data[:, :2]

        n = data.shape[0]
        start = pos
        end = pos + n

        # equal-power fade-in
        if self.fade_in > 0 and start < self.fade_in:
            take = min(n, self.fade_in - start)
            if take > 0:
                env = self._eqpow_ramp(start, take, self.fade_in)[:, None]
                data[:take, :] *= env

        # equal-power fade-out
        if self.fade_out > 0 and total > 0:
            tail_start = max(0, total - self.fade_out)
            if end > tail_start:
                s_in = max(0, tail_start - start)
                take = min(n - s_in, max(0, total - tail_start))
                if take > 0:
                    pos_in_fade = (start + s_in) - tail_start
                    env = 1.0 - self._eqpow_ramp(pos_in_fade, take, self.fade_out)
                    data[s_in:s_in+take, :] *= env[:, None]

        return data

    @staticmethod
    def _eqpow_ramp(start: int, length: int, total: int) -> np.ndarray:
        if total <= 1: return np.ones((length,), dtype=np.float32)
        x0 = clamp(start / float(total), 0.0, 1.0)
        x1 = clamp((start + length) / float(total), 0.0, 1.0)
        x = np.linspace(x0, x1, num=length, endpoint=False, dtype=np.float32)
        return (np.sin(0.5 * math.pi * x) ** 2).astype(np.float32)

    @staticmethod
    def _xfade_env(n: int) -> np.ndarray:
        if n <= 0: return np.zeros((0,), dtype=np.float32)
        x = np.linspace(0.0, 1.0, num=n, endpoint=False, dtype=np.float32)
        return (np.sin(0.5 * math.pi * x) ** 2).astype(np.float32)


# ---------- layer that mixes all four BinVoice ----------

class BreathingLayer(AudioLayer):
    def __init__(self, sample_rate: int = 44100):
        if SEED is not None:
            try: random.seed(int(SEED))
            except: pass

        self.sample_rate = int(sample_rate)
        self.sr = int(sample_rate)
        self.state = "NO_PRESENCE"

        # parent handles mixer contract; we don't use its file loader
        super().__init__(name="breathing", audio_dir=None, sample_rate=self.sample_rate)

        # base cap (from main) with smoothing
        self._cap_now = 0.0          # the actual cap used this callback
        self._cap_tgt = 0.0          # target set by main
        self._cap_slew_s = 1.0 / max(1, int(CAP_SLEW_S * self.sample_rate))

        self._dbg_samples = 0

        # current BR (effective, if you pass br_eff)
        self._br = 0.0

        # prepare file lists per bin
        per_bin: Dict[str, List[str]] = {f"BIN{i}": [] for i in (1,2,3,4)}
        for bn in per_bin.keys():
            folder = os.path.join(AUDIO_ROOT, bn)
            if os.path.isdir(folder):
                files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".wav")]
                files.sort()
                per_bin[bn] = files
            print(f"[breathing] {bn}: {len(per_bin[bn])} file(s)")

        # construct four voices
        self.voices: Dict[str, BinVoice] = {
            bn: BinVoice(bn, self.sr, per_bin[bn]) for bn in per_bin.keys()
        }

        # hysteretic bin chooser
        self._chooser = HystereticBins(SPLIT_12, SPLIT_23, SPLIT_34, HYST)
        self._active = self._chooser.update(self._br)
        self._apply_bin_targets()

    # ---- your helper you added (kept here, but we use the hysteretic chooser) ----
    def _set_active_bin_from_br(self, br_eff: float):
        s12, s23, s34 = SPLIT_12, SPLIT_23, SPLIT_34
        if br_eff < s12: target = "BIN1"
        elif br_eff < s23: target = "BIN2"
        elif br_eff < s34: target = "BIN3"
        else: target = "BIN4"
        if target != self._active:
            self._active = target
            self._apply_bin_targets()

    # ---- API expected by main ----
    def set_base_gain(self, base_gain: float, state: str | None = None):
        # main sets ONLY the layer cap target; we apply it once after summing all bins
        self._cap_tgt = max(0.0, min(1.0, float(base_gain)))
        if state is not None:
            self.state = state

    def update_from_tick(self, br: float, presence: int, eclipse: Optional[int] = None):
        """
        Call this once per *new* line (outer tick).
        Pass br = effective BR (already epi-multiplied if you use epi).
        We keep decks running silently in NO_PRESENCE; main should set cap=0 there.
        """
        self._br = float(br)
        # Use hysteresis-based chooser for stability:
        new_bin = self._chooser.update(self._br)
        if new_bin != self._active:
            self._active = new_bin
            self._apply_bin_targets()

    def _apply_bin_targets(self):
        for name, v in self.voices.items():
            v.set_target(to_one=(name == self._active))

        # ---- audio callback pull ----
    def render(self, frames: int) -> np.ndarray:
        # --- time-based slew of the cap toward target (per audio block) ---
        if self._cap_slew_s <= 0:
            self._cap_now = float(self._cap_tgt)
        else:
            frac = frames / (self._cap_slew_s * self.sample_rate)  # portion of slew window covered by this block
            if frac < 0.0: 
                frac = 0.0
            elif frac > 1.0:
                frac = 1.0
            if self._cap_now < self._cap_tgt:
                self._cap_now = min(self._cap_tgt, self._cap_now + frac)
            elif self._cap_now > self._cap_tgt:
                self._cap_now = max(self._cap_tgt, self._cap_now - frac)

        # --- sum all bin voices (each has its own internal fades) ---
        mix = np.zeros((frames, 2), dtype="float32")
        for voice in self.voices.values():
            buf = voice.render(frames)
            if buf is not None and buf.size:
                mix += buf

        # --- STRICT LINEAR CAP (this *must* obey main’s ceiling) ---
        cap = float(self._cap_now)
        if cap <= 1e-6:
            # keep returning silence until the cap rises
            return np.zeros_like(mix)

        out = mix * cap

        # --- optional soft limiter ---
        peak = float(np.max(np.abs(out)))
        if peak > 1.0:
            out /= (peak + 1e-9)

        # --- lightweight debug every ~0.5 s (optional; remove if too chatty) ---
        if not hasattr(self, "_dbg_samples"):
            self._dbg_samples = 0
        self._dbg_samples += frames
        if self._dbg_samples >= int(0.5 * self.sample_rate):
            pre_rms = float(np.sqrt(np.mean(np.maximum(mix**2, 1e-12))))
            post_rms = float(np.sqrt(np.mean(np.maximum(out**2, 1e-12))))
            print(f"[breathing] cap_now={self._cap_now:.2f} cap_tgt={self._cap_tgt:.2f} "
                f"preRMS={pre_rms:.3f} postRMS={post_rms:.3f} active={self._active}")
            self._dbg_samples = 0

        return out

    def debug_print(self, base_gain: float):
        print(
            f"[breathing] active={self._active}, state={self.state}, "
            f"cap_tgt={self._cap_tgt:.2f}, cap_now={self._cap_now:.2f}, "
            f"br={self._br:.2f}  "
            + " ".join(f"{k}:{v.file_count()}" for k, v in self.voices.items())
        )


# ---------- factory ----------

def create_layer(sample_rate: int = 44100):
    return BreathingLayer(sample_rate=sample_rate)
