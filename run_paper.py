"""Paper-method launcher; route new evaluation outputs into the repository ``results/`` tree.

The launcher only orchestrates the documented entry points:

* ``run.py`` (``journal_flow.cli``) for plan/check/evaluate/train/fit-stats
* ``timing/benchmark_all.py`` for receiver timing

Output location is portable and reproducible: everything defaults to
``<repo>/results`` and can be redirected with the ``KAPFLOW_OUTPUT_ROOT``
environment variable or by passing the underlying ``--output-root`` /
``--output`` flags explicitly.
"""
from pathlib import Path
import os
import subprocess
import sys

PROJECT = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get('KAPFLOW_OUTPUT_ROOT', PROJECT/'results'))
METHODS = {'kap':'flowadmm','pnpflow':'pnpflow','rcflow':'rcflow','srf':'srf','pnp-cm':'pnp-cm','ls':'ls','lmmse':'lmmse'}
USAGE = ('Usage: python run_paper.py {kap,pnpflow,rcflow,srf,pnp-cm,ls,lmmse} '
         '{plan,check,evaluate,train,fit-stats} [options]')

def main():
    args=sys.argv[1:]
    if not args or args[0] not in {*METHODS,'timing'}:
        print(USAGE)
        print('Timing: python run_paper.py timing [benchmark options]')
        print('Outputs default to %s; override with KAPFLOW_OUTPUT_ROOT or the child flags.' % OUTPUT)
        return 2
    method=args.pop(0)
    if method=='timing':
        if '--output' not in args:args+=['--output',str(OUTPUT/'timing')]
        command=[sys.executable,str(PROJECT/'timing/benchmark_all.py'),'--manifest',str(PROJECT/'timing/methods.json'),*args]
    else:
        if not args:args=['plan']
        if '--output-root' not in args:args+=['--output-root',str(OUTPUT)]
        if method=='srf' and '--conditions' not in args:args+=['--conditions','ckm']
        command=[sys.executable,str(PROJECT/'run.py'),*args,'--method',METHODS[method]]
    return subprocess.call(command,cwd=PROJECT)

if __name__=='__main__':raise SystemExit(main())
