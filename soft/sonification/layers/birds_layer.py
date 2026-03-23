from pathlib import Path
from .loop_layer import AudioLayer

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
AUDIO_DIR = PROJECT_ROOT / "audio" / "birds"

def create_layer(sample_rate: int = 44100) -> AudioLayer:
    return AudioLayer("birds", AUDIO_DIR, sample_rate=sample_rate)
