"""Stable repository paths for AUBO tests at any directory depth."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples" / "phone_to_auboi10"
SRC = REPO_ROOT / "src"
