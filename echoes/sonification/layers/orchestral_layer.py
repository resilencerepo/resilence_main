# orchestral_layer.py
from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys

import numpy as np
import soundfile as sf

# -------------------------------------------------------------------
# Path setup
# -------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent  # .../sonification

# orchestral is inside /sonification/orchestral
ORCH_ENGINE_ROOT = PROJECT_ROOT / "orchestral"
#ORCH_DATASET_ROOT = ORCH_ENGINE_ROOT / "dataset_root"
ORCH_DATASET_ROOT = ORCH_ENGINE_ROOT / "dataset_root_44100"

orchestra_backend = None

if ORCH_ENGINE_ROOT.exists():
    sys.path.insert(0, str(ORCH_ENGINE_ROOT))
    try:
        import orchestra_player_hardcoded as oph  # type: ignore
        orchestra_backend = oph
        print(f"[orchestral] Using orchestra_player_hardcoded from {ORCH_ENGINE_ROOT}")
    except ImportError as e:
        print(f"[orchestral] WARNING: Could not import orchestra_player_hardcoded: {e}")
else:
    print(f"[orchestral] WARNING: Orchestral engine dir not found: {ORCH_ENGINE_ROOT}")


# -------------------------------------------------------------------
# Simple fallback: single wav from sonification/audio/orchestral
# (Used only if orchestra_player_hardcoded or dataset_root are missing.)
# -------------------------------------------------------------------

FALLBACK_AUDIO_DIR = PROJECT_ROOT / "audio" / "orchestral"


def _load_fallback_file() -> Optional[tuple[np.ndarray, int]]:
    if not FALLBACK_AUDIO_DIR.exists():
        print(f"[orchestral] Fallback audio dir not found: {FALLBACK_AUDIO_DIR}")
        return None

    wavs = sorted(FALLBACK_AUDIO_DIR.glob("*.wav"))
    if not wavs:
        print(f"[orchestral] No .wav files found in fallback dir: {FALLBACK_AUDIO_DIR}")
        return None

    path = wavs[0]
    print(f"[orchestral] Fallback will use: {path.name}")
    data, sr = sf.read(str(path), dtype="float32")

    # Ensure stereo for mixer compatibility
    if data.ndim == 1:
        data = np.stack([data, data], axis=-1)
    elif data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)

    return data.astype(np.float32), int(sr)


# -------------------------------------------------------------------
# OrchestralLayer abstraction used by the central mixer
# -------------------------------------------------------------------

class OrchestralLayer:
    """
    Mixer-friendly orchestral event layer (Option A).

    Backend mode (preferred):
      - intro()/outro() build internal buffers via orchestra_player_hardcoded.*build_*_buffer()
      - render(frames) returns chunks to be mixed by the central mixer
      - DOES NOT call sd.play / DOES NOT open its own stream

    Fallback mode:
      - uses a pre-rendered stereo wav buffer and streams it via render(frames)
      - also does not call sd.play (so it’s still mixer-safe)
    """

    def __init__(self, sr: int = 48000, channels: int = 2):
        self.backend = orchestra_backend
        self.dataset_root = ORCH_DATASET_ROOT if ORCH_DATASET_ROOT.exists() else None

        self.sr = int(sr)
        self.channels = int(channels)

        # Playback state (mono buffer for backend; stereo buffer for fallback)
        self._buf: Optional[np.ndarray] = None   # (N,) mono OR (N,2) stereo
        self._pos: int = 0
        self._playing: bool = False

        # Fallback buffer (stereo)
        self.fallback_buffer: Optional[np.ndarray] = None
        self.fallback_sr: Optional[int] = None

        # Decide mode
        if self.backend is not None and self.dataset_root is not None:
            # Validate backend expectations for SR if available
            backend_sr = getattr(self.backend, "SR", None)
            if backend_sr is not None and int(backend_sr) != self.sr:
                raise ValueError(
                    f"[orchestral] SR mismatch: backend SR={backend_sr} vs mixer SR={self.sr}. "
                    f"Unify SRs (recommended: 48000)."
                )

            # Require the new buffer-returning API
            if not hasattr(self.backend, "build_intro_outro_buffer"):
                raise AttributeError(
                    "[orchestral] orchestra_player_hardcoded is missing build_intro_outro_buffer(). "
                    "Add build_*_buffer() functions (Option A refactor)."
                )

            print(f"[orchestral] Backend + dataset_root found: {self.dataset_root}")
            self.mode = "backend"

        else:
            print("[orchestral] Using fallback mode (pre-rendered wav buffer, if available).")
            fb = _load_fallback_file()
            if fb is not None:
                self.fallback_buffer, self.fallback_sr = fb
            self.mode = "fallback"

    # ---------------------------- event triggers -------------------------
    def _start_backend_event(self, kind: str) -> None:
        if self.backend is None or self.dataset_root is None:
            print("[orchestral] Backend event requested but backend/dataset_root missing.")
            return

        # You can switch outro to a different buffer builder later if you like.
        if kind == "intro":
            buf_mono = self.backend.build_intro_outro_buffer(self.dataset_root)
        elif kind == "outro":
            buf_mono = self.backend.build_intro_outro_buffer(self.dataset_root)
        else:
            raise ValueError(f"Unknown backend event kind: {kind}")

        if buf_mono.ndim != 1:
            # render_layers returns mono (N,). If you ever change it, handle it here.
            buf_mono = np.asarray(buf_mono).reshape(-1)

        self._buf = buf_mono.astype(np.float32)
        self._pos = 0
        self._playing = True

    def _start_fallback_event(self, label: str) -> None:
        if self.fallback_buffer is None or self.fallback_sr is None:
            print(f"[ORCH] {label} (fallback) – no audio loaded, silent.")
            self._buf = None
            self._playing = False
            return

        if int(self.fallback_sr) != self.sr:
            raise ValueError(
                f"[orchestral] Fallback wav SR={self.fallback_sr} vs mixer SR={self.sr}. "
                "Use a wav at the mixer SR or resample offline."
            )

        self._buf = self.fallback_buffer  # stereo (N,2)
        self._pos = 0
        self._playing = True

    # ---------------------------- public API ------------------------
    def intro(self) -> None:
        """Called when presence changes 0 → 1."""
        if self.mode == "backend":
            print("[ORCH] INTRO (buffered, backend)")
            self._start_backend_event("intro")
        else:
            print("[ORCH] INTRO (buffered, fallback)")
            self._start_fallback_event("INTRO")

    def outro(self) -> None:
        """Called when presence changes 1 → 0."""
        if self.mode == "backend":
            print("[ORCH] OUTRO (buffered, backend)")
            self._start_backend_event("outro")
        else:
            print("[ORCH] OUTRO (buffered, fallback)")
            self._start_fallback_event("OUTRO")

    def stop(self) -> None:
        """Hard stop (silence immediately)."""
        self._buf = None
        self._pos = 0
        self._playing = False

    def is_playing(self) -> bool:
        return bool(self._playing)

    def render(self, frames: int) -> np.ndarray:
        """
        Mixer callback pulls audio from here.

        Returns float32 array shape (frames, channels).
        """
        out = np.zeros((frames, self.channels), dtype=np.float32)

        if not self._playing or self._buf is None:
            return out

        buf = self._buf

        # Backend buffer is mono (N,)
        if buf.ndim == 1:
            n = buf.shape[0]
            end = min(n, self._pos + frames)
            chunk = buf[self._pos:end]
            self._pos = end

            L = chunk.size
            out[:L, 0] = chunk
            if self.channels > 1:
                out[:L, 1] = chunk

        # Fallback buffer is stereo (N,2)
        else:
            n = buf.shape[0]
            end = min(n, self._pos + frames)
            chunk = buf[self._pos:end, :self.channels]
            self._pos = end

            L = chunk.shape[0]
            out[:L, :chunk.shape[1]] = chunk

        if self._pos >= (buf.shape[0] if buf.ndim == 1 else buf.shape[0]):
            self._playing = False

        return out


def create_layer(sr: int = 48000, channels: int = 2) -> OrchestralLayer:
    """Factory used by your sonification/mixer system."""
    return OrchestralLayer(sr=sr, channels=channels)
