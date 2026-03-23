# sonification_engine.py
"""
Local sonification engine.
Wraps all audio/state logic. No Streamlit, no API, no server.

Public API:
    engine = SonificationEngine()
    engine.load_uploaded_scenario(lines)   # list of raw text lines
    engine.load_internal_demo()            # uses INTERNAL_DEMO_PATH
    engine.apply_config(calm_threshold, epi, gain_presets)
    engine.start()
    engine.stop(immediate=True)
    engine.get_snapshot()                  # -> dict with current display state
    engine.compute_internal_demo_summary() # -> str (one symbol per second)
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd

# ─────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# The built-in demo file — place alongside this script
INTERNAL_DEMO_PATH = _HERE / "internal_demo_80s.txt"

SAMPLE_RATE = 44100
BLOCK_SIZE  = 512

TICK_SECONDS = 1.0    # one data-line per second
MAIN_SECONDS = 0.05   # inner-loop update interval

CONFIRM_TICKS = 2

FADE_DUR = {
    "PRESENCE":     5.0,
    "ECLIPSE":      2.0,
    "CALM_EXCITED": 4.0,
}

DEFAULT_GAIN_PRESETS: Dict[str, Dict[str, float]] = {
    "NO_PRESENCE": {"messiaen": 0.0, "breathing": 0.0, "ocean": 0.0, "birds": 0.0},
    "ECLIPSE":     {"messiaen": 0.0, "breathing": 0.0, "ocean": 0.4, "birds": 0.0},
    "CALM":        {"messiaen": 0.7, "breathing": 0.7, "ocean": 0.1, "birds": 0.15},
    "EXCITED":     {"messiaen": 1.0, "breathing": 0.05,"ocean": 0.05,"birds": 0.4},
}


# ─────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────

def parse_line(line: str) -> Optional[Dict]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        return None
    try:
        time_str       = parts[0]
        breathing_rate = float(parts[1])
        breathing_dev  = float(parts[2])
        presence       = int(parts[3])
    except ValueError:
        return None
    try:
        moon = int(parts[4])
    except ValueError:
        moon = 1  # treat as no eclipse
    return {
        "time": time_str,
        "breathing_rate": breathing_rate,
        "breathing_dev":  breathing_dev,
        "presence":       presence,
        "moon":           moon,
    }


def compute_raw_state(presence: int, moon: Optional[int],
                      br_eff: float, threshold: float) -> str:
    if presence == 0:
        return "NO_PRESENCE"
    if moon == 0:
        return "ECLIPSE"
    return "CALM" if br_eff < threshold else "EXCITED"


def classify_transition(old: str, new: str) -> Optional[str]:
    if old == new:
        return None
    if old == "NO_PRESENCE" or new == "NO_PRESENCE":
        return "PRESENCE"
    if old == "ECLIPSE" or new == "ECLIPSE":
        return "ECLIPSE"
    return "CALM_EXCITED"


def _interp_gains(g_from: Dict[str, float], g_to: Dict[str, float],
                  alpha: float) -> Dict[str, float]:
    return {k: (1 - alpha) * g_from[k] + alpha * g_to[k] for k in g_from}


# ─────────────────────────────────────────────
# Layer imports — graceful fallback if missing
# ─────────────────────────────────────────────

def _try_import_layers():
    """
    Returns (messiaen_layer, breathing_layer, ocean_layer, birds_layer, orchestral_layer)
    as instances, or None per layer if import fails.
    """
    layers = {}
    try:
        import layers.messiaen_layer as _m
        layers["messiaen"] = _m.create_layer(sample_rate=SAMPLE_RATE)
    except Exception as e:
        print(f"[engine] messiaen layer unavailable: {e}")
        layers["messiaen"] = None

    try:
        import layers.breathing_layer as _b
        layers["breathing"] = _b.create_layer(sample_rate=SAMPLE_RATE)
    except Exception as e:
        print(f"[engine] breathing layer unavailable: {e}")
        layers["breathing"] = None

    try:
        import layers.ocean_layer as _o
        layers["ocean"] = _o.create_layer(sample_rate=SAMPLE_RATE)
    except Exception as e:
        print(f"[engine] ocean layer unavailable: {e}")
        layers["ocean"] = None

    try:
        import layers.birds_layer as _bi
        layers["birds"] = _bi.create_layer(sample_rate=SAMPLE_RATE)
    except Exception as e:
        print(f"[engine] birds layer unavailable: {e}")
        layers["birds"] = None

    try:
        import layers.orchestral_layer as _orch
        layers["orchestral"] = _orch.create_layer(sr=SAMPLE_RATE, channels=2)
    except Exception as e:
        print(f"[engine] orchestral layer unavailable: {e}")
        layers["orchestral"] = None

    return layers


# ─────────────────────────────────────────────
# Silent dummy layer (used when real one missing)
# ─────────────────────────────────────────────

class _SilentLayer:
    def __init__(self, name: str):
        self.name = name
    def set_base_gain(self, gain: float, state: str = ""):
        pass
    def render(self, frames: int) -> np.ndarray:
        return np.zeros((frames, 2), dtype=np.float32)
    def debug_print(self, _: float):
        pass
    # breathing compat
    def update_from_tick(self, br: float, presence: int, eclipse=None):
        pass


class _SilentOrchestrallayer:
    def intro(self): pass
    def outro(self): pass
    def stop(self): pass
    def is_playing(self) -> bool: return False
    def render(self, frames: int) -> np.ndarray:
        return np.zeros((frames, 2), dtype=np.float32)


# ─────────────────────────────────────────────
# Central Mixer (local sounddevice stream)
# ─────────────────────────────────────────────

class _CentralMixer:
    def __init__(self, all_layers: Dict):
        self._layers = all_layers
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=2,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, outdata, frames, time_info, status):
        mix = np.zeros((frames, 2), dtype=np.float32)
        for layer in self._layers.values():
            mix += layer.render(frames)
        # soft clip
        np.clip(mix, -1.0, 1.0, out=mix)
        outdata[:] = mix

    def start(self):
        self._stream.start()

    def stop(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


# ─────────────────────────────────────────────
# Main Engine
# ─────────────────────────────────────────────

class SonificationEngine:
    """
    Thread-safe local sonification engine.
    Call apply_config() at any time (even while playing) to update params.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Scenario data (list of parsed dicts)
        self._scenario: List[Dict] = []
        self._is_demo: bool = False

        # Config (can be updated live)
        self._calm_threshold: float = 28.0
        self._epi: float = 1.0
        self._gain_presets: Dict[str, Dict[str, float]] = {
            s: dict(v) for s, v in DEFAULT_GAIN_PRESETS.items()
        }

        # Playback state (written by _run_loop, read by get_snapshot)
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._mixer: Optional[_CentralMixer] = None

        # Snapshot (updated each tick)
        self._snapshot: Dict = self._empty_snapshot()

    # ── public API ──────────────────────────────────────────────────────

    def load_uploaded_scenario(self, lines: List[str]) -> int:
        """Parse raw lines into scenario. Returns number of valid rows."""
        parsed = [parse_line(l) for l in lines if l.strip()]
        parsed = [p for p in parsed if p is not None]
        with self._lock:
            self._scenario = parsed
            self._is_demo  = False
        return len(parsed)

    def load_internal_demo(self) -> int:
        """Load the built-in demo file. Returns number of valid rows."""
        if not INTERNAL_DEMO_PATH.exists():
            raise FileNotFoundError(
                f"Internal demo file not found: {INTERNAL_DEMO_PATH}\n"
                "Place internal_demo_80s.txt next to sonification_engine.py"
            )
        with open(INTERNAL_DEMO_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        parsed = [parse_line(l) for l in lines if l.strip()]
        parsed = [p for p in parsed if p is not None]
        with self._lock:
            self._scenario = parsed
            self._is_demo  = True
        return len(parsed)

    def apply_config(self, calm_threshold: float, epi: float,
                     gain_presets: Dict[str, Dict[str, float]]):
        with self._lock:
            self._calm_threshold = float(calm_threshold)
            self._epi = float(epi)
            self._gain_presets = {s: dict(v) for s, v in gain_presets.items()}

    def start(self):
        """Start playback in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self, immediate: bool = True):
        """Stop playback."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_snapshot(self) -> Dict:
        with self._lock:
            return dict(self._snapshot)

    def compute_internal_demo_summary(self) -> str:
        """
        Compute the one-symbol-per-second summary line for the internal demo.
        Uses raw state (no confirmation). Transitions shown as ~.
        """
        with self._lock:
            scenario  = list(self._scenario)
            threshold = self._calm_threshold
            epi       = self._epi

        if not scenario:
            return ""

        symbols = []
        prev_raw: Optional[str] = None

        for row in scenario:
            br_eff   = row["breathing_rate"] * epi
            raw      = compute_raw_state(row["presence"], row["moon"], br_eff, threshold)
            if prev_raw is None:
                prev_raw = raw

            if raw != prev_raw:
                symbols.append("~")
            else:
                sym = {"NO_PRESENCE": "_", "CALM": "C",
                       "EXCITED": "E", "ECLIPSE": "M"}.get(raw, "?")
                symbols.append(sym)

            prev_raw = raw

        return "".join(symbols)

    @property
    def is_playing(self) -> bool:
        return self._running

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty_snapshot() -> Dict:
        return {
            "presence":          0,
            "moon":              1,
            "breathing_rate":    0.0,
            "br_eff":            0.0,
            "calm_threshold":    28.0,
            "epi":               1.0,
            "confirmed_state":   "NO_PRESENCE",
            "transitioning":     False,
            "from_state":        "NO_PRESENCE",
            "to_state":          "NO_PRESENCE",
            "gains":             {k: 0.0 for k in ("messiaen","breathing","ocean","birds","orchestral")},
            "tick_index":        0,
            "finished":          False,
        }

    def _run_loop(self):
        # Build layers
        raw_layers = _try_import_layers()

        def _wrap(name, layer, fallback_cls):
            if layer is None:
                return fallback_cls(name)
            return layer

        import layers.breathing_layer as _bmod  # noqa
        layers_dict = {
            "messiaen":   _wrap("messiaen",   raw_layers.get("messiaen"),   _SilentLayer),
            "breathing":  _wrap("breathing",  raw_layers.get("breathing"),  _SilentLayer),
            "ocean":      _wrap("ocean",      raw_layers.get("ocean"),      _SilentLayer),
            "birds":      _wrap("birds",      raw_layers.get("birds"),      _SilentLayer),
            "orchestral": raw_layers.get("orchestral") or _SilentOrchestrallayer(),
        }

        mixer = _CentralMixer(layers_dict)
        mixer.start()
        self._mixer = mixer

        # ── state machine ──
        pending_state: Optional[str] = None
        pending_count: int = 0
        confirmed_state: str = "NO_PRESENCE"

        transition_active:    bool  = False
        transition_start:     float = 0.0
        transition_duration:  float = 0.0
        from_state:           str   = "NO_PRESENCE"
        to_state:             str   = "NO_PRESENCE"

        last_presence_raw: int   = 0
        br_eff:            float = 0.0
        last_data:         Optional[Dict] = None

        # Snapshot helpers
        def _update_snapshot(**kwargs):
            with self._lock:
                self._snapshot.update(kwargs)

        # ── scenario playback ──
        with self._lock:
            scenario = list(self._scenario)

        tick_index = 0

        for row in scenario:
            if not self._running:
                break

            tick_index += 1
            tick_start = time.time()

            # ── read config snapshot ──
            with self._lock:
                threshold = self._calm_threshold
                epi       = self._epi
                presets   = {s: dict(v) for s, v in self._gain_presets.items()}

            data = row
            last_data = data

            # ── update breathing layer ──
            bl = layers_dict["breathing"]
            if hasattr(bl, "update_from_tick"):
                bl.update_from_tick(
                    br=data["breathing_rate"],
                    presence=data["presence"],
                    eclipse=data["moon"],
                )

            presence = data["presence"]

            if last_presence_raw == 0 and presence == 1:
                pass  # epi comes from slider now
            elif presence == 0:
                pass
            last_presence_raw = presence

            if presence == 1:
                br_eff = data["breathing_rate"] * epi
            else:
                br_eff = 0.0

            # update breathing layer with br_eff
            if hasattr(bl, "update_from_tick"):
                bl.update_from_tick(br=br_eff, presence=presence, eclipse=data["moon"])

            raw_state = compute_raw_state(
                presence=presence,
                moon=data["moon"],
                br_eff=br_eff,
                threshold=threshold,
            )

            # ── confirmation logic ──
            if not transition_active:
                old_confirmed = confirmed_state

                if raw_state == confirmed_state:
                    pending_state = None
                    pending_count = 0
                elif pending_state is None or raw_state != pending_state:
                    pending_state = raw_state
                    pending_count = 1
                else:
                    pending_count += 1
                    if pending_count >= CONFIRM_TICKS:
                        confirmed_state = raw_state
                        pending_state   = None
                        pending_count   = 0

                if confirmed_state != old_confirmed:
                    t_type = classify_transition(old_confirmed, confirmed_state)
                    if t_type:
                        transition_active    = True
                        transition_start     = tick_start
                        transition_duration  = FADE_DUR[t_type]
                        from_state           = old_confirmed
                        to_state             = confirmed_state

                        orch = layers_dict["orchestral"]
                        if t_type == "PRESENCE":
                            if old_confirmed == "NO_PRESENCE":
                                orch.intro()
                            elif confirmed_state == "NO_PRESENCE":
                                orch.outro()

            # ── inner loop ──
            last_gains: Dict[str, float] = {}
            orch_gain: float = 0.0

            while True:
                now     = time.time()
                elapsed = now - tick_start
                if elapsed >= TICK_SECONDS:
                    break

                # re-read config (allows live slider updates)
                with self._lock:
                    threshold = self._calm_threshold
                    epi       = self._epi
                    presets   = {s: dict(v) for s, v in self._gain_presets.items()}

                # finish transition?
                if transition_active:
                    if now >= transition_start + transition_duration:
                        transition_active = False
                        confirmed_state   = to_state

                # compute gains
                if not transition_active:
                    gains  = dict(presets[confirmed_state])
                    alpha  = None
                    orch_gain = 0.0
                else:
                    alpha  = (now - transition_start) / transition_duration
                    alpha  = max(0.0, min(1.0, alpha))
                    gains  = _interp_gains(presets[from_state], presets[to_state], alpha)
                    # orchestral: 1 only during PRESENCE transitions
                    t_type = classify_transition(from_state, to_state)
                    orch_gain = 1.0 if t_type == "PRESENCE" else 0.0

                last_gains = gains

                # push gains to layers
                for name in ("messiaen", "breathing", "ocean", "birds"):
                    layer = layers_dict[name]
                    layer.set_base_gain(gains[name], state=confirmed_state)

                # push orchestral gain
                orch = layers_dict["orchestral"]
                if hasattr(orch, "set_base_gain"):
                    orch.set_base_gain(orch_gain)

                # update snapshot
                all_gains = dict(last_gains)
                all_gains["orchestral"] = orch_gain
                _update_snapshot(
                    presence=data["presence"],
                    moon=data["moon"],
                    breathing_rate=data["breathing_rate"],
                    br_eff=br_eff,
                    calm_threshold=threshold,
                    epi=epi,
                    confirmed_state=confirmed_state,
                    transitioning=transition_active,
                    from_state=from_state,
                    to_state=to_state,
                    gains=all_gains,
                    tick_index=tick_index,
                    finished=False,
                )

                sleep_remaining = TICK_SECONDS - (time.time() - tick_start)
                if sleep_remaining <= 0:
                    break
                time.sleep(min(MAIN_SECONDS, sleep_remaining))

        # done
        _update_snapshot(finished=True)
        self._running = False

        # silence all layers
        for name in ("messiaen", "breathing", "ocean", "birds"):
            layers_dict[name].set_base_gain(0.0)

        # short wait for audio to fade, then stop stream
        time.sleep(0.1)
        mixer.stop()
        self._mixer = None
