"""New namespace; cache/train stages cannot overwrite old data or experiments."""
import argparse,os,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.headline_cap_experiment import execute
from src.encoder_comparison import write
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['cache','train']);args=parser.parse_args()
    out=ROOT/'artifacts/experiments/headline_cap_2026-10-12'
    try:execute(ROOT,out,args.stage)
    except BaseException:
        out.mkdir(parents=True,exist_ok=True);write(out/f'failure_{args.stage}_{time.time_ns()}.json',{'traceback':traceback.format_exc()});raise
