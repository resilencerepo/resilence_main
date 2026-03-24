# main.py
from pathlib import Path
import yaml

from orchestra_player_hardcoded import play_orchestra

def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    here = Path(__file__).resolve().parent
    cfg_path = here / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")

    cfg = load_config(cfg_path)

    # Resolve dataset_root (can be None: then the orchestra module uses ORCHESTRA_DATASET or ./dataset_root)
    dataset_root_cfg = cfg.get("dataset_root", None)
    if dataset_root_cfg is not None:
        dataset_root = (here / dataset_root_cfg).resolve()
    else:
        dataset_root = None

    layers_cfg = cfg.get("layers", {})

    # ---------------- ORCHESTRAL LAYER (single on/off) ----------------
    orch_cfg = layers_cfg.get("orchestra", {})
    if orch_cfg.get("enabled", True):
        print("[MAIN] Orchestral layer: ON")

        mode       = orch_cfg.get("mode", "harmony")
        root_note  = orch_cfg.get("root_note", "C4")
        chord_type = orch_cfg.get("chord_type", "major")
        max_volume = float(orch_cfg.get("max_volume", 0.8))

        # fixed 1+2+2 envelope as before
        fade_in_s  = 1.0
        sustain_s  = 2.0
        fade_out_s = 2.0

        play_orchestra(
            dataset_root=dataset_root,
            mode=mode,
            fade_in_s=fade_in_s,
            sustain_s=sustain_s,
            fade_out_s=fade_out_s,
            max_volume=max_volume,
            root_note=root_note,
            chord_type=chord_type,
            # we don’t touch internal layer_gains/mutes here – treat as one block
            layer_gains=None,
            layer_mutes=None,
            lock_last_s=0.9,
            debug_pitch=False,
        )
    else:
        print("[MAIN] Orchestral layer: OFF")

    # ---------------- OTHER LAYERS (stubs) ----------------
    # breathing layer
    breathing_cfg = layers_cfg.get("breathing", {})
    if breathing_cfg.get("enabled", False):
        print("[MAIN] Breathing layer: ON (TODO: plug your breathing engine here)")

    # ambient layer
    ambient_cfg = layers_cfg.get("ambient", {})
    if ambient_cfg.get("enabled", False):
        print("[MAIN] Ambient layer: ON (TODO: plug your ambient engine here)")


if __name__ == "__main__":
    main()
