"""Main method only: standard, threefold and robustness FlowADMM."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from journal_flow.cli import main

if __name__=='__main__':
    if any(x=='--method' or x.startswith('--method=') for x in sys.argv[1:]):
        raise SystemExit('main/run.py is FlowADMM only. Use baselines/run.py for other methods.')
    main([*sys.argv[1:],'--method','flowadmm'])
