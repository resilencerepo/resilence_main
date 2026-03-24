# sonification.py
"""
    "EXCITED": {
        "messiaen": 0.4,
        "breathing": 0.2,
        "ocean": 0.1,
        "birds": 0.6,
    },
"""
from __future__ import annotations

import time
import random
from typing import Dict, Optional

import numpy as np
import sounddevice as sd

import layers.messiaen_layer as messiaen_mod
import layers.breathing_layer as breathing_mod
import layers.ocean_layer as ocean_mod
import layers.birds_layer as birds_mod
import layers.orchestral_layer as orchestral_mod
from layers.loop_layer import AudioLayer

# ----------------- GLOBALS -----------------
TICK_SECONDS = 1.0      # how often we read a new line from the file
MAIN_SECONDS = 0.25     # how often we update gains / transitions inside each tick


SAMPLE_RATE = 44100
BLOCK_SIZE = 512  # audio block size for sounddevice

CALM_THRESHOLD = 28.0  # threshold on br_eff (breathing_rate * epi)
CONFIRM_TICKS = 2      # state must appear twice (per new data) to confirm

FADE_DUR = {
    "PRESENCE":     5.0,  # to/from NO_PRESENCE
    "ECLIPSE":      2.0,  # to/from ECLIPSE
    "CALM_EXCITED": 4.0,  # CALM <-> EXCITED
}

GAIN_PRESETS = {
    "NO_PRESENCE": {
        "messiaen": 0.0,
        "breathing": 0.0,
        "ocean": 0.0,
        "birds": 0.0,
    },
    "ECLIPSE": {
        "messiaen": 0.0,
        "breathing": 0.0,
        "ocean": 0.4,
        "birds": 0.0,
    },
    "CALM": {
        "messiaen": 0.7,
        "breathing": 0.7,
        "ocean": 0.1,
        "birds": 0.15,
    },
    "EXCITED": {
        "messiaen": 1.0,
        "breathing": 0.05,
        "ocean": 0.05,
        "birds": 0.4,
    },
}

# 2-tick confirmation state variables (per new data)
pending_state: Optional[str] = None
pending_count: int = 0
confirmed_state: str = "NO_PRESENCE"


# --------------- HELPERS -------------------
def parse_line(line: str) -> Optional[Dict]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 5:
        return None

    try:
        time_str = parts[0]
        breathing_rate = float(parts[1])
        breathing_dev = float(parts[2])
        presence = int(parts[3])
    except ValueError:
        return None

    try:
        moon = int(parts[4])
    except ValueError:
        moon = None  # treat as "no eclipse"

    return {
        "time": time_str,
        "breathing_rate": breathing_rate,
        "breathing_dev": breathing_dev,
        "presence": presence,
        "moon": moon,
    }


def compute_raw_state(presence: int, moon: Optional[int], br_eff: float, threshold: float) -> str:
    if presence == 0:
        return "NO_PRESENCE"
    if moon == 0:
        return "ECLIPSE"
    if br_eff < threshold:
        return "CALM"
    else:
        return "EXCITED"


def update_state_with_confirmation(raw_state: str) -> str:
    """
    Called once per *new line* (once per second).
    Needs CONFIRM_TICKS consecutive raw_state values before changing confirmed_state.
    """
    global pending_state, pending_count, confirmed_state

    if raw_state == confirmed_state:
        pending_state = None
        pending_count = 0
        return confirmed_state

    if pending_state is None or raw_state != pending_state:
        pending_state = raw_state
        pending_count = 1
        return confirmed_state  # still old state

    pending_count += 1
    if pending_count >= CONFIRM_TICKS:
        confirmed_state = raw_state
        pending_state = None
        pending_count = 0

    return confirmed_state


def classify_transition(old: str, new: str) -> Optional[str]:
    if old == new:
        return None
    if old == "NO_PRESENCE" or new == "NO_PRESENCE":
        return "PRESENCE"
    if old == "ECLIPSE" or new == "ECLIPSE":
        return "ECLIPSE"
    return "CALM_EXCITED"


# --------------- CENTRAL MIXER -----------------
class CentralMixer:
    """
    Central audio mixer using sounddevice.
    It pulls audio from each AudioLayer according to their current gains.
    """
    def __init__(self, layers: Dict[str, AudioLayer], sample_rate: int, block_size: int):
        self.layers = layers
        self.sample_rate = sample_rate
        self.block_size = block_size

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=2,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, outdata, frames, time_info, status):
        if status:
            print("[MIXER] Status:", status)
        mix = np.zeros((frames, 2), dtype="float32")
        for layer in self.layers.values():
            mix += layer.render(frames)
            ############################3
        outdata[:] = mix

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()


# --------------- MAIN LOOP -----------------
def main():
    global confirmed_state
    print("[ENTRY] Running sonification main()")

    # ---- Create layers ----
    messiaen_layer: AudioLayer = messiaen_mod.create_layer(sample_rate=SAMPLE_RATE)
    breathing_layer: AudioLayer = breathing_mod.create_layer(sample_rate=SAMPLE_RATE)
    ocean_layer: AudioLayer = ocean_mod.create_layer(sample_rate=SAMPLE_RATE)
    birds_layer: AudioLayer = birds_mod.create_layer(sample_rate=SAMPLE_RATE)
    orchestral_layer: AudioLayer = orchestral_mod.create_layer(sr=SAMPLE_RATE, channels=2)

    # Only continuous, gain-controlled layers go through the mixer:
    layers = {
        "messiaen": messiaen_layer,
        "breathing": breathing_layer,
        "ocean": ocean_layer,
        "birds": birds_layer,
        "orchestral": orchestral_layer,   # ✅ now goes through the central mixer
    }




    # ---- Start central mixer ----
    mixer = CentralMixer(layers, sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    mixer.start()
    print("[MAIN] Central mixer started")

    log_path = "sonification.txt"
    print(f"[MAIN] Tailing {log_path} ... (Ctrl+C to stop)\n")

    # Transition tracking
    transition_active = False
    transition_start = 0.0
    transition_duration = 0.0
    from_state = confirmed_state
    to_state = confirmed_state

    # EPI / breathing tracking
    current_epi: Optional[float] = None
    last_presence_raw = 0
    br_eff: float = 0.0
    last_data: Optional[Dict] = None

    tick_index = 0  # outer ticks, once per second
    last_raw_line: Optional[str] = None

    try:
        with open(log_path, "r") as f:
            # initial: read first line if exists
            # (we'll treat absence like "no data yet")
            while True:
                tick_index += 1
                tick_start = time.time()

                # --------- READ ONE NEW LINE (once per second) ---------
                pos = f.tell()
                line = f.readline()
                is_new = True

                if not line:
                    # no new line → reuse last one
                    f.seek(pos)
                    if last_raw_line is None:
                        print(f"[TICK {tick_index}] Waiting for first line in {log_path}...")
                        # still run an inner loop with no data? here we just sleep the whole second
                        time.sleep(TICK_SECONDS)
                        continue
                    raw_line = last_raw_line
                    is_new = False
                else:
                    raw_line = line.strip()
                    last_raw_line = raw_line
                    is_new = True

                # --------- PARSE AND UPDATE STATE (ONLY IF is_new) ---------
                if is_new:
                    data = parse_line(raw_line)
                    if not data:
                        print(f"[TICK {tick_index}] Malformed line, skipping: {raw_line}")
                        # still wait one second before trying next line
                        time.sleep(TICK_SECONDS)
                        continue

                    last_data = data
                    # tell breathing layer the latest breathing rate each new line
                    breathing_layer.update_from_tick(
                        br=data["breathing_rate"],
                        presence=data["presence"],
                        eclipse=data["moon"],  # ignored for bin logic; main caps handle eclipse
                    )

                    presence = data["presence"]

                    # EPI logic: constant during presence=1 episode
                    if last_presence_raw == 0 and presence == 1:
                        current_epi = random.uniform(1.5, 2.0)
                    elif presence == 0:
                        current_epi = None
                    last_presence_raw = presence

                    if presence == 1 and current_epi is not None:
                        br_eff = data["breathing_rate"] * current_epi
                    else:
                        br_eff = 0.0
                    # after computing br_eff and state:
                    breathing_layer.update_from_tick(br=br_eff, presence=presence, eclipse=data["moon"])

                    if isinstance(breathing_layer, breathing_mod.BreathingLayer):
                        breathing_layer.update_from_tick(br=br_eff, presence=presence, eclipse=data["moon"])



                    raw_state = compute_raw_state(
                        presence=presence,
                        moon=data["moon"],
                        br_eff=br_eff,
                        threshold=CALM_THRESHOLD,
                    )

                    if not transition_active:
                        old_state = confirmed_state
                        current_state = update_state_with_confirmation(raw_state)

                        if current_state != old_state:
                            t_type = classify_transition(old_state, current_state)
                            if t_type is not None:
                                transition_active = True
                                transition_start = tick_start
                                transition_duration = FADE_DUR[t_type]
                                from_state = old_state
                                to_state = current_state

                                print(
                                    f"[TICK {tick_index}] NEW TRANSITION: "
                                    f"{old_state} → {current_state} ({t_type})"
                                )

                                if t_type == "PRESENCE":
                                    if old_state == "NO_PRESENCE":
                                        orchestral_layer.intro()
                                    elif current_state == "NO_PRESENCE":
                                        orchestral_layer.outro()
                                    else:
                                        print("   [ORCH] PRESENCE transition but unclear direction")

                # If not is_new, we keep last_data, br_eff, and any active transition.
                # Now we run the INNER FASTER LOOP for up to TICK_SECONDS seconds.

                last_gains: Dict[str, float] = {}
                last_alpha: Optional[float] = None
                last_t_type: Optional[str] = None
                inner_step = 0

                while True:
                    now = time.time()
                    elapsed = now - tick_start
                    if elapsed >= TICK_SECONDS:
                        break  # done with this outer second

                    inner_step += 1

                    # --- Update transition progress based on time ---
                    if transition_active:
                        if now >= transition_start + transition_duration:
                            transition_active = False
                            confirmed_state = to_state

                    # --- Compute gains at this inner step ---
                    if not transition_active:
                        gains = GAIN_PRESETS[confirmed_state]
                        alpha = None
                        t_type = None
                    else:
                        alpha = (now - transition_start) / transition_duration
                        alpha = max(0.0, min(1.0, alpha))
                        gains = {}
                        for layer_name in GAIN_PRESETS[from_state]:
                            g_from = GAIN_PRESETS[from_state][layer_name]
                            g_to = GAIN_PRESETS[to_state][layer_name]
                            gains[layer_name] = (1 - alpha) * g_from + alpha * g_to
                        t_type = classify_transition(from_state, to_state)

                    last_gains = gains
                    last_alpha = alpha
                    last_t_type = t_type

                    # --- Update layers with current gains (every MAIN_SECONDS) ---
                    state_for_layers = confirmed_state
                    for name, layer in layers.items():
                        if name == "orchestral":
                            continue
                        base_gain = gains[name]
                        layer.set_base_gain(base_gain, state=state_for_layers)

                        # no debug here to avoid too much spam

                    # Sleep until next inner step, but don't go past TICK_SECONDS
                    now2 = time.time()
                    elapsed2 = now2 - tick_start
                    remaining = TICK_SECONDS - elapsed2
                    if remaining <= 0:
                        break
                    sleep_time = min(MAIN_SECONDS, remaining)
                    time.sleep(sleep_time)

                # --------- DEBUG PRINT ONCE PER OUTER TICK ---------
                data_for_print = last_data if last_data is not None else {}
                presence_print = data_for_print.get("presence", "?")
                br_print = data_for_print.get("breathing_rate", 0.0)
                dev_print = data_for_print.get("breathing_dev", 0.0)
                moon_print = data_for_print.get("moon", "?")

                print(f"[TICK {tick_index}] line: {raw_line}  (is_new={is_new})")
                print(f"   presence         : {presence_print}")
                print(f"   breathing_rate   : {br_print:.2f}")
                print(f"   breathing_dev    : {dev_print:.3f}")
                print(f"   moon             : {moon_print}")
                print(f"   current_epi      : {current_epi}")
                print(f"   confirmed_state  : {confirmed_state}")
                print(f"   br_eff           : {br_eff:.2f}")
                print(f"   transition_active: {transition_active}")
                if transition_active and last_alpha is not None:
                    print(f"   transition: {from_state} → {to_state}, type={last_t_type}, alpha={last_alpha:.2f}")
                print(f"   gains (non-orch) : {last_gains}\n")

                # Optional: print per-layer summary once per second
                for name, layer in layers.items():
                    if name == "orchestral":
                        continue
                    base_gain = last_gains.get(name, 0.0)
                    layer.debug_print(base_gain)


    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        mixer.stop()
        print("[MAIN] Mixer stopped")


if __name__ == "__main__":
    main()
