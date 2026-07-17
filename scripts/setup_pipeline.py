"""One-shot pipeline: download → enrich → index."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print(f"\n>>> {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    python = sys.executable
    size = sys.argv[1] if len(sys.argv) > 1 else "800"

    run([python, str(root / "scripts" / "download_dataset.py"), "--size", size])
    run([python, str(root / "scripts" / "enrich_metadata.py")])
    run([python, str(root / "run_index.py")])
    print("\nSetup complete. Try:")
    print(f'  {python} run_search.py "blue blazer in office setting"')


if __name__ == "__main__":
    main()
