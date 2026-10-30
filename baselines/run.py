"""Baselines: PnP-Flow, LS/LMMSE, RCFlow/SRF, PnP-CM."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from journal_flow.cli import main,parser

if __name__=='__main__':
    args=sys.argv[1:]
    if not any(x=='--method' or x.startswith('--method=') for x in args):
        args=[*args,'--method','pnpflow']
    if parser().parse_args(args).method=='flowadmm':
        raise SystemExit('FlowADMM is the main method: use main/run.py')
    main(args)
