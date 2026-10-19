"""Explicit stages for the frozen exploratory CMIN timing-policy comparison."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cmin_timing_study import main

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['prepare', 'encode', 'preflight', 'run', 'analyze'])
    main(p.parse_args().stage)
