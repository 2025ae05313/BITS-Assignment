"""Train + track + package (script version of notebook 02). See heartml.train."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heartml.train import main  # noqa: E402

if __name__ == "__main__":
    main()
