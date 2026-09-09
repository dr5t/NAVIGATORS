#!/usr/bin/env python3
"""Run measured A–G replay ablations on a supplied recording. No generated scenarios."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import main  # type: ignore[import]

if __name__ == '__main__':
    args = sys.argv[1:]
    raise SystemExit(main(args if '--ablations' in args else ['--ablations', *args]))
