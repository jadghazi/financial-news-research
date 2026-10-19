"""Explicit new namespace; CPU cache/train stages never overwrite version 1."""
import argparse
import os
import sys
import traceback
import time
import re
from pathlib import Path
sys.dont_write_bytecode=True
os.environ.setdefault('HF_HUB_DISABLE_XET','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.encoder_comparison import execute, write

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['cache','train'])
    parser.add_argument('--experiment',default='encoder_comparison_2026-10-12')
    args=parser.parse_args()
    if not re.fullmatch(r'[a-z0-9_-]+',args.experiment):parser.error('Invalid experiment name')
    out=ROOT/'artifacts/experiments'/args.experiment
    try:execute(ROOT,out,args.stage)
    except BaseException:
        out.mkdir(parents=True,exist_ok=True)
        write(out/f'failure_{args.stage}_{time.time_ns()}.json',{'traceback':traceback.format_exc()})
        raise
