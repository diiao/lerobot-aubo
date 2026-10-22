"""Repository resource locations shared by AUBO modules and entry scripts."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "configs" / "aubo_i10"
EXAMPLE_ROOT = REPO_ROOT / "examples" / "phone_to_auboi10"
