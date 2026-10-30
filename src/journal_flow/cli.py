"""Explicit experiment plans, checkpoint guards and restartable evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
# Portable defaults: every path is resolved relative to this checkout and can be
# overridden from the command line or the environment.  Nothing is tied to the
# original development machine.
#   data root:        --data-root / CKMSRF_DATA_ROOT       default <repo>/data/ckm_srf_64x16_h5
#   checkpoints root: CKMSRF_CHECKPOINTS_ROOT              default <repo>/checkpoints
DATA_DEFAULT = str(ROOT/'data'/'ckm_srf_64x16_h5')
CHECKPOINTS_DEFAULT = ROOT/'checkpoints'
METHODS = ('flowadmm', 'pnpflow', 'rcflow', 'srf', 'pnp-cm', 'ls', 'lmmse')


def registry():
    return {r['id']: r for r in json.loads((ROOT/'configs/checkpoints.json').read_text(encoding='utf-8'))}


def artifact_path(record):
    """Resolve a registry target below the (optionally relocated) checkpoints root."""
    target = Path(record['target'])
    root = Path(os.environ.get('CKMSRF_CHECKPOINTS_ROOT', CHECKPOINTS_DEFAULT))
    if target.parts and target.parts[0] == 'checkpoints':
        return root.joinpath(*target.parts[1:])
    return ROOT/target


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def positive(value):
    n=int(value)
    if n<1: raise argparse.ArgumentTypeError('must be a positive integer')
    return n


def parser():
    p=argparse.ArgumentParser(description='FlowADMM main methods and research baselines')
    p.add_argument('action',choices=['list','plan','check','evaluate','train','fit-stats'])
    p.add_argument('--method',choices=METHODS,default='flowadmm')
    p.add_argument('--track',choices=['standard','threefold','robustness'],default='standard')
    p.add_argument('--conditions',nargs='+',choices=['ckm','no_ckm'],default=['ckm','no_ckm'])
    p.add_argument('--folds',nargs='+',type=int,choices=[1,2,3],default=[1,2,3])
    p.add_argument('--priors',nargs='+',choices=['standard','robust'],default=['standard','robust'])
    p.add_argument('--condition-modes',nargs='+',choices=['clean','mismatch'],default=['clean','mismatch'])
    p.add_argument('--m',nargs='+',type=positive,default=[1],help='MC endpoint samples, any positive integer; NOT pilot length')
    p.add_argument('--seeds',nargs='+',type=int,default=None,help='Default: three inference seeds for standard flow methods, one otherwise')
    p.add_argument('--iterations',type=positive,default=40)
    p.add_argument('--data-root',type=Path,default=Path(os.environ.get('CKMSRF_DATA_ROOT',DATA_DEFAULT)))
    p.add_argument('--output-root',type=Path,default=None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--batch-size',type=positive,default=16)
    p.add_argument('--num-workers',type=int,default=4)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--snr-db',nargs='+',type=float,default=[-10,-5,0,5,10,15,20,25,30])
    p.add_argument('--max-samples',type=positive)
    p.add_argument('--checkpoint',type=Path,help='Explicit checkpoint override, requires a single planned evaluation job')
    p.add_argument('--resume',action='store_true',help='Skip successfully completed identical jobs; retry incomplete jobs with log history')
    p.add_argument('--profile',type=Path,default=ROOT/'configs/hardware/gpu_24gb.yaml')
    p.add_argument('--run-id',default=None)
    p.add_argument('--init-checkpoint',type=Path)
    p.add_argument('--resume-checkpoint',type=Path)
    return p


def common(a):
    group='main' if a.method=='flowadmm' else 'baselines'
    args=['--data-root',str(a.data_root.resolve()),'--run-root',str(ROOT/group/'runs'),
          '--device',a.device,'--precision','fp32','--num-workers',str(a.num_workers),
          '--set','data.backend=plain_h5','--set','hardware.use_accelerate=false',
          '--set',f'hardware.eval_batch_size={a.batch_size}','--set','training.validation_seed=9102']
    return args


def prior_id(cond,fold,robust=False):
    return f'robust_{cond}' if robust else f'straight_{cond}'+(f'_fold{fold}' if fold else '')


def make_jobs(a):
    if a.output_root is None:
        a.output_root=ROOT/('main' if a.method=='flowadmm' else 'baselines')/'results'
    if a.track=='robustness' and a.method!='flowadmm':
        raise ValueError('robustness track is defined for FlowADMM')
    if a.track=='threefold' and a.method not in ['flowadmm','pnpflow','ls','lmmse']:
        raise ValueError('No recorded threefold training for this baseline')
    if a.method=='srf' and 'no_ckm' in a.conditions:
        raise ValueError('Recorded SRF baseline is CKM-conditioned; specify --conditions ckm')
    if a.method not in ['flowadmm','pnpflow'] and a.m!=[1]:
        raise ValueError('M applies only to flowadmm/pnpflow')
    reg=registry(); jobs=[]
    folds=a.folds if a.track=='threefold' else [0]
    conditions=a.conditions if a.method not in ['ls','lmmse'] else ['ckm']
    seeds=a.seeds or ([10009102,10009103,10009104] if a.track=='standard' and a.method in ['flowadmm','pnpflow'] else [10009102])
    for fold in folds:
        for cond in conditions:
            for prior in (a.priors if a.track=='robustness' else ['standard']):
                for mode in (a.condition_modes if a.track=='robustness' else ['clean']):
                    for m in a.m:
                        for seed in seeds:
                            pid=prior_id(cond,fold,prior=='robust')
                            if a.method in ['rcflow','srf','pnp-cm']:
                                pid=f"{'pnp_cm' if a.method=='pnp-cm' else a.method}_{cond}"
                            ckpt=artifact_path(reg[pid])
                            label=f'{a.track}/{a.method}/'+(f'holdout_{fold:02d}/' if fold else '')+f'{cond}/{prior}/{mode}/M{m}/seed_{seed}'
                            out=a.output_root.resolve()/label
                            module='ckmsrf.evaluate_classical' if a.method in ['ls','lmmse'] else 'ckmsrf.evaluate'
                            cmd=[sys.executable,'-m',module]+common(a)
                            cmd += ['--split',a.split,'--snr-db',*map(str,a.snr_db),'--output-dir',str(out),'--subspace-metrics']
                            requirements=[pid]
                            if a.method in ['ls','lmmse']:
                                cmd+=['--methods',a.method,'--protocol-checkpoint',str(ckpt)]
                                if a.method=='lmmse':
                                    sid='lmmse'+(f'_fold{fold}' if fold else '')
                                    cmd+=['--statistics',str(artifact_path(reg[sid]))]
                                    requirements.append(sid)
                            else:
                                sampler={'flowadmm':'flow-admm','pnpflow':'pnp-flow','rcflow':'srf-rc','srf':'srf-rc','pnp-cm':'pnp-cm'}[a.method]
                                cmd+=['--checkpoint',str(ckpt),'--sampler',sampler]
                                if a.method in ['flowadmm','pnpflow']:
                                    cmd+=['--pnp-iterations',str(a.iterations),'--flow-initialization','ls',
                                          '--flow-latent-std','1','--flow-latent-seed',str(seed)]
                                    if a.method=='flowadmm':
                                        tau,lo,hi,gamma=(.25,.2,.95,.5) if a.track=='robustness' else (.125,0.,.95,.25)
                                        cmd+=['--flow-admm-tau',str(tau),'--flow-admm-time-min',str(lo),
                                              '--flow-admm-time-max',str(hi),'--flow-admm-gamma',str(gamma),
                                              '--flow-admm-k-avg',str(m),'--flow-admm-k-avg-schedule','constant']
                                    else:
                                        cmd+=['--pnp-flow-step-scale','1.5','--pnp-flow-alpha','0.25',
                                              '--pnp-flow-time-min','0.2','--pnp-flow-time-max','1',
                                              '--pnp-flow-latent-samples',str(m)]
                                elif a.method=='pnp-cm':
                                    cmd+=['--pnp-iterations','4','--pnp-rho','1','--pnp-lambda','0.1']
                                else:
                                    cmd+=['--steps','20','--outer-loops','1','--pbg-steps','0','--fig-plus-mix','0']
                                cmd+=['--condition-mode',mode]
                                if a.track=='robustness':
                                    cmd+=['--condition-corruption-profile',str(ROOT/'configs/experiments/ckm_condition_mismatch_medium.yaml'),
                                          '--condition-corruption-seed','260211950']
                            if a.max_samples: cmd+=['--max-samples',str(a.max_samples)]
                            jobs.append(dict(label=label,command=cmd,output=str(out),artifacts=requirements,
                                             result_dir=str(out/a.method) if a.method in ['ls','lmmse'] else str(out),
                                             fold=fold,condition=cond,prior=prior,m=m,seed=seed,mode=mode))
    if a.checkpoint:
        if len(jobs)!=1: raise ValueError('--checkpoint requires exactly one job (select one condition, M, seed, prior, mode and fold)')
        cmd=jobs[0]['command']; flag='--protocol-checkpoint' if a.method in ['ls','lmmse'] else '--checkpoint'
        cmd[cmd.index(flag)+1]=str(a.checkpoint.resolve())
    return jobs


def check_jobs(jobs,a):
    import torch
    from ckmsrf.factory import build_model, parse_scene_ids
    from ckmsrf.evaluate import _load_checkpoint, _normalization, _load_model_weights, _prior_semantics
    cache={}
    for j in jobs:
        cmd=j['command']; flag='--protocol-checkpoint' if '--protocol-checkpoint' in cmd else '--checkpoint'
        path=Path(cmd[cmd.index(flag)+1])
        if not path.is_file(): raise FileNotFoundError(f'{path}\nSee CHECKPOINTS.md; robust no-CKM needs its own trained checkpoint.')
        if path not in cache:
            payload=_load_checkpoint(path); cfg=payload.get('config',{})
            model=build_model(cfg)
            _load_model_weights(model,payload,prefer_ema=True)
            _normalization(cfg,payload)
            cache[path]=(payload,cfg,sha256(path))
        payload,cfg,digest=cache[path]
        prior_method,_=_prior_semantics(cfg)
        required={'flowadmm':'straight_flow_matching','pnpflow':'straight_flow_matching',
                  'rcflow':'gmflow','srf':'gmflow','pnp-cm':'consistency'}.get(a.method)
        if required and prior_method!=required:
            raise ValueError(f'{path}: {a.method} requires {required}, found {prior_method}')
        cond=cfg.get('model',{}).get('condition',{})
        is_uncond=not cond.get('enabled',True) or cond.get('force_unconditional',False)
        if a.method not in ['ls','lmmse'] and is_uncond!=(j['condition']=='no_ckm'):
            raise ValueError(f'{path}: CKM condition does not match requested variant')
        is_mask=bool(cond.get('use_presence_mask',False))
        if a.track=='robustness' and is_mask!=(j['prior']=='robust'):
            raise ValueError(f'{path}: requested normal/robust architecture does not match')
        if j['fold']:
            def normalize_scene_id(value):
                return int(str(value).removeprefix('scene_'))
            expected={1,2,3}-{j['fold']}
            for key in ['train_scenes','validation_scenes']:
                actual={normalize_scene_id(x) for x in cfg.get('data',{}).get(key,[])}
                if actual!=expected: raise ValueError(f'{path}: {key}={actual}, expected {expected}')
            actual={normalize_scene_id(x) for x in cfg['data'].get('test_scenes',[])}
            if actual!={j['fold']}: raise ValueError(f'{path}: wrong held-out test scene')
        j['checkpoint_sha256']=digest
        expected_digest=registry()[j['artifacts'][0]].get('sha256')
        if expected_digest and not a.checkpoint and digest!=expected_digest:
            raise ValueError(f'{path}: differs from recorded paper checkpoint SHA256')
        if '--statistics' in cmd:
            from ckmsrf.evaluate_classical import _load_statistics, _check_normalization
            stat=Path(cmd[cmd.index('--statistics')+1])
            if not stat.is_file(): raise FileNotFoundError(stat)
            artifact=_load_statistics(stat,True)
            expected=set(parse_scene_ids(cfg['data'].get('train_scenes')) or ())
            actual=set(parse_scene_ids(artifact.get('config',{}).get('data',{}).get('train_scenes')) or ())
            if artifact.get('split')!='train' or not expected or actual!=expected:
                raise ValueError(f'{stat}: statistics must use the same source training scenes')
            _check_normalization(_normalization(cfg,payload),artifact)
            j['statistics_sha256']=sha256(stat)
    if a.device.startswith('cuda'):
        index=int(a.device.split(':')[1]) if ':' in a.device else 0
        if index<0 or index>=torch.cuda.device_count(): raise ValueError(f'{a.device}: only {torch.cuda.device_count()} GPUs visible to this process')
    if not a.data_root.is_dir(): raise FileNotFoundError(a.data_root)
    return jobs


def evaluate(jobs,a):
    env=os.environ.copy(); env['PYTHONPATH']=str(ROOT/'src')
    code_hash=hashlib.sha256(''.join(sha256(p) for p in sorted((ROOT/'src').rglob('*.py'))).encode()).hexdigest()
    for j in jobs:
        out=Path(j['output']); out.mkdir(parents=True,exist_ok=True)
        result_dir=Path(j['result_dir'])
        fingerprint=hashlib.sha256(json.dumps([j,code_hash],sort_keys=True).encode()).hexdigest()
        done=out/'curated_complete.json'; plan=out/'curated_job.json'
        if done.exists():
            saved=json.loads(done.read_text())
            if saved.get('fingerprint')!=fingerprint: raise ValueError(f'{out}: different settings; choose a new --output-root')
            if a.resume and (result_dir/'results.json').is_file() and (result_dir/'metrics.csv').is_file():
                print('SKIP',j['label'],flush=True); continue
            raise FileExistsError(f'{out}: completed; pass --resume to skip or choose a new output root')
        if plan.exists() and json.loads(plan.read_text())['fingerprint']!=fingerprint:
            raise ValueError(f'{out}: incomplete job has different settings; use another output root')
        plan.write_text(json.dumps({'fingerprint':fingerprint,**j},ensure_ascii=False,indent=2),encoding='utf-8')
        print('RUN',j['label'],flush=True)
        # OS locks are released automatically if this controller is interrupted.
        # The file remains as metadata and is safe to reuse on the next attempt.
        with (out/'curated.lock').open('a+b') as lock:
            lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
            if os.name=='nt':
                import msvcrt
                try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
                except OSError as exc: raise RuntimeError(f'{out}: another evaluator is running') from exc
            else:
                import fcntl
                try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError as exc: raise RuntimeError(f'{out}: another evaluator is running') from exc
            if done.exists():
                if a.resume: continue
                raise FileExistsError(f'{out}: completed by another invocation')
            with (out/'run.log').open('a',encoding='utf-8') as log:
                log.write('\nSTART '+datetime.now(timezone.utc).isoformat()+'\n'); log.flush()
                result=subprocess.run(j['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError(f'Job failed ({result.returncode}); see {out / "run.log"}')
            if not (result_dir/'results.json').is_file() or not (result_dir/'metrics.csv').is_file():
                raise RuntimeError(f'{out}: missing expected result files')
            done.write_text(json.dumps({'fingerprint':fingerprint,'completed_at':datetime.now(timezone.utc).isoformat()}),encoding='utf-8')


def training_config(a,cond,fold):
    if a.method in ['flowadmm','pnpflow']:
        if a.track=='threefold': return ROOT/f'main/configs/threefold/holdout_{fold:02d}_{cond}.yaml'
        return ROOT/f'main/configs/{a.track}/{cond}.yaml'
    key='pnp_cm' if a.method=='pnp-cm' else a.method
    return ROOT/f'baselines/configs/{key}_{cond}.yaml'


def main(argv=None):
    a=parser().parse_args(argv)
    if a.num_workers<0: raise ValueError('--num-workers must be >= 0')
    if a.track=='robustness' and a.method!='flowadmm': raise ValueError('robustness track belongs to FlowADMM')
    if a.track=='threefold' and a.method not in ['flowadmm','pnpflow','ls','lmmse']:
        raise ValueError('No recorded threefold model for this method')
    if a.action=='list':
        print(json.dumps(list(registry().values()),ensure_ascii=False,indent=2)); return
    if a.action=='train':
        if a.method in ['ls','lmmse']: raise ValueError('LS has no training; LMMSE uses fit-stats')
        if a.method=='srf' and 'no_ckm' in a.conditions: raise ValueError('SRF requires --conditions ckm')
        if a.resume_checkpoint and (len(a.conditions)>1 or (a.track=='threefold' and len(a.folds)>1)):
            raise ValueError('Resume a single training run at a time')
        env=os.environ.copy(); env['PYTHONPATH']=str(ROOT/'src')
        for fold in (a.folds if a.track=='threefold' else [0]):
            for cond in a.conditions:
                cmd=[sys.executable,'-m','ckmsrf.train','--config',str(training_config(a,cond,fold)),
                     '--profile',str(a.profile)]+common(a)
                if a.run_id: cmd+=['--run-id',a.run_id]
                if a.init_checkpoint: cmd+=['--init-checkpoint',str(a.init_checkpoint)]
                if a.resume_checkpoint: cmd+=['--resume',str(a.resume_checkpoint)]
                subprocess.run(cmd,cwd=ROOT,env=env,check=True)
        return
    if a.action=='fit-stats':
        reg=registry(); env=os.environ.copy(); env['PYTHONPATH']=str(ROOT/'src')
        for fold in (a.folds if a.track=='threefold' else [0]):
            sid='lmmse'+(f'_fold{fold}' if fold else '')
            dest=artifact_path(reg[sid])
            if dest.exists(): raise FileExistsError(f'{dest}: preserve existing statistics; use ckmsrf.fit_channel_stats --output for a new study')
            cmd=[sys.executable,'-m','ckmsrf.fit_channel_stats','--protocol-checkpoint',
                 str(artifact_path(reg[prior_id('ckm',fold)])),'--output',str(dest)]+common(a)
            subprocess.run(cmd,cwd=ROOT,env=env,check=True)
        return
    jobs=make_jobs(a)
    if a.action=='plan':
        print(json.dumps(jobs,ensure_ascii=False,indent=2)); return
    check_jobs(jobs,a)
    if a.action=='check': print(f'Validated {len(jobs)} jobs, weights, condition/fold metadata and runtime.'); return
    evaluate(jobs,a)
