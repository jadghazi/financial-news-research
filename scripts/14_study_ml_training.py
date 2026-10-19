"""Run only a versioned downstream train/validation ML study on cached features."""
import argparse,os,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True;os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.ml_training_study import execute
from src.encoder_comparison import write
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','analyze']);args=parser.parse_args()
    out=ROOT/'artifacts/experiments/ml_training_study_2026-10-13'
    write(out/f'command_{args.stage}_{time.time_ns()}.json',{'argv':sys.argv,'stage':args.stage})
    try:execute(ROOT,out,args.stage)
    except BaseException:
        write(out/f'failure_{args.stage}_{time.time_ns()}.json',{'traceback':traceback.format_exc()});raise
