"""Run only the isolated pipeline diagnosis; all models/caches already local."""
import argparse,os,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True;os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.pipeline_diagnosis import execute
from src.encoder_comparison import write
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['positive','flow','baselines','analyze']);args=parser.parse_args();out=ROOT/'artifacts/diagnostics/pipeline_diagnosis_2026-10-13'
    try:execute(ROOT,out,args.stage)
    except BaseException:
        out.mkdir(parents=True,exist_ok=True);write(out/f'failure_{args.stage}_{time.time_ns()}.json',{'traceback':traceback.format_exc()});raise
