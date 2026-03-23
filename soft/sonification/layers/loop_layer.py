# layers/loop_layer.py
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import soundfile as sf


@dataclass
class AudioLayer:
    name: str
    audio_dir: Optional[Path]
    sample_rate: int = 44100

    buffers: List[np.ndarray] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)

    # gain handling
    base_gain: float = 0.0        # from main (max allowed)
    local_factor: float = 1.0     # internal [0..1] factor
    target_gain: float = 0.0      # = base_gain * local_factor
    current_gain: float = 0.0     # actually applied
    smoothing_time: float = 0.5   # seconds to approach target_gain
    state: str = "NO_PRESENCE"


    def __post_init__(self):
        if self.audio_dir is not None:
            self.audio_dir = Path(self.audio_dir).resolve()

        if self.audio_dir is None or not self.audio_dir.exists():
            print(f"[{self.name}] WARNING: audio dir {self.audio_dir} missing. Layer will be silent.")
            return

        wav_files = sorted(self.audio_dir.glob("*.wav"))
        if not wav_files:
            print(f"[{self.name}] WARNING: no .wav in {self.audio_dir}. Layer will be silent.")
            return

        print(f"[{self.name}] Found {len(wav_files)} wav file(s) in {self.audio_dir}:")
        for f in wav_files:
            print(f"   - {f.name}")
            data, sr = sf.read(str(f), dtype="float32")
            if sr != self.sample_rate:
                print(
                    f"[{self.name}] WARNING: {f.name} has sr={sr}, "
                    f"expected {self.sample_rate}. Skipping this file."
                )
                continue

            # Ensure stereo
            if data.ndim == 1:
                data = np.stack([data, data], axis=-1)
            elif data.shape[1] == 1:
                data = np.repeat(data, 2, axis=1)

            self.buffers.append(data)
            self.positions.append(0)

        if not self.buffers:
            print(f"[{self.name}] No usable wav files after sr check. Layer will be silent.")

    # --------- called by main each tick ---------
    def set_gain(self, gain: float, state: Optional[str] = None):
        self.target_gain = float(gain)
        if state is not None:
            self.state = state

    # --------- called by main each tick ---------
    def set_base_gain(self, base_gain: float, state: Optional[str] = None):
        """Main sets the maximum allowed gain for this layer."""
        self.base_gain = float(base_gain)
        if state is not None:
            self.state = state
        # recompute target with current local factor
        self.target_gain = self.base_gain * self.local_factor

    # --------- called by the layer's own logic (optional) ---------
    def set_local_factor(self, factor: float):
        """
        Internal modulation inside the layer.
        Must be in [0, 1] so we never exceed base_gain.
        """
        f = max(0.0, min(1.0, float(factor)))
        self.local_factor = f
        self.target_gain = self.base_gain * self.local_factor


    # --------- called by mixer each audio callback ---------
    def render(self, frames: int) -> np.ndarray:
        """
        Returns a (frames, 2) stereo buffer with internal gain smoothing.
        """
        if not self.buffers:
            return np.zeros((frames, 2), dtype="float32")

        # Compute block-level smoothing towards target_gain
        if self.smoothing_time <= 0.0 or self.current_gain == self.target_gain:
            # no smoothing: jump directly
            gain_curve = np.full(frames, self.target_gain, dtype="float32")
            self.current_gain = self.target_gain
        else:
            # how much of the remaining difference we can cover in this block
            # depending on smoothing_time (seconds)
            # fraction of smoothing interval covered by this block:
            frac = frames / (self.smoothing_time * self.sample_rate)
            frac = min(max(frac, 0.0), 1.0)

            new_gain = self.current_gain + (self.target_gain - self.current_gain) * frac
            # per-sample ramp between current_gain and new_gain
            gain_curve = np.linspace(self.current_gain, new_gain, frames, dtype="float32")
            self.current_gain = new_gain

        out = np.zeros((frames, 2), dtype="float32")

        for i, buf in enumerate(self.buffers):
            pos = self.positions[i]
            length = buf.shape[0]

            if length == 0:
                continue

            if pos + frames <= length:
                chunk = buf[pos:pos + frames]
                self.positions[i] = pos + frames
                if self.positions[i] >= length:
                    self.positions[i] = 0
            else:
                n1 = length - pos
                n2 = frames - n1
                chunk = np.vstack((buf[pos:], buf[:n2]))
                self.positions[i] = n2

            out += chunk

        # apply smoothed gain curve (per-sample)
        out *= gain_curve[:, None]
        return out

    def debug_print(self, base_gain: float):
        print(
            f"[{self.name}] threads={len(self.buffers)}, "
            f"state={self.state}, base={base_gain:.2f}, "
            f"local_factor={self.local_factor:.2f}, "
            f"target={self.target_gain:.2f}, current={self.current_gain:.2f}"
        )
