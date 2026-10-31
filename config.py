import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"

for folder in [DATA_DIR, UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR]:
    os.makedirs(folder, exist_ok=True)

DB_PATH = str(DATA_DIR / "wildlife.db")

SPECIES_PRIORITY = {
    "human": {"score": 1.0, "is_endangered": False, "is_human": True},
    "person": {"score": 1.0, "is_endangered": False, "is_human": True},
    "tiger": {"score": 0.9, "is_endangered": True, "is_human": False},
    "lion": {"score": 0.9, "is_endangered": True, "is_human": False},
    "elephant": {"score": 0.85, "is_endangered": True, "is_human": False},
    "rhino": {"score": 0.9, "is_endangered": True, "is_human": False},
    "leopard": {"score": 0.85, "is_endangered": True, "is_human": False},
    "bear": {"score": 0.7, "is_endangered": False, "is_human": False},
    "deer": {"score": 0.3, "is_endangered": False, "is_human": False},
    "fox": {"score": 0.3, "is_endangered": False, "is_human": False},
    "bird": {"score": 0.2, "is_endangered": False, "is_human": False},
}

DEMO_CLASSES = ["person", "tiger", "lion", "elephant", "bear", "deer"]