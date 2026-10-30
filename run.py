"""Portable entry point: always import this checkout, never an old editable install."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from journal_flow.cli import main

if __name__ == '__main__':
    main()
