"""Execute only the isolated train/validation history comparison."""
import argparse,os,re,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.history_comparison import execute
from src.encoder_comparison import write
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','gates','train','analyze']);parser.add_argument('--experiment',default='history_comparison_2026-10-12');args=parser.parse_args()
    if not re.fullmatch(r'history_comparison_[a-z0-9_-]+',args.experiment):parser.error('Expected a history_comparison_ experiment name')
    out=ROOT/'artifacts/experiments'/args.experiment
    try:execute(ROOT,out,args.stage)
    except BaseException:
        out.mkdir(parents=True,exist_ok=True);write(out/f'failure_{args.stage}_{time.time_ns()}.json',{'traceback':traceback.format_exc()});raise
