"""Uniform resident-input receiver timing; no training or evaluator mutation."""
from __future__ import annotations
import argparse,copy,csv,hashlib,json,math,os,platform,random,sys,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO_ROOT=HERE.parent
# Import the single canonical package in <repo>/src.  timing/ deliberately keeps no
# private source copy, so the benchmark always measures the code under test.
sys.path.insert(0,str(REPO_ROOT/'src'))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from ckmsrf.evaluate import _load_checkpoint,_normalization,_load_model_weights,_prior_semantics
from ckmsrf.factory import build_model,build_dataset,load_or_build_index
from ckmsrf.physics import generate_observation
from ckmsrf.sampling import sample_ckmsrf
from ckmsrf.sampling_cm import sample_ckmcm_pnp
from ckmsrf.sampling_flow import sample_flow_admm,sample_pnp_flow
from ckmsrf.baselines import least_squares,gaussian_lmmse,select_statistics
from ckmsrf.evaluate_classical import _check_normalization

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writecsv(p,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def move(x,device):
    if torch.is_tensor(x):return x.to(device)
    if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
    return x
def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)
def generator(device,seed):return torch.Generator(device=device).manual_seed(seed)

def solve(spec,model,batch,cfg,device,seed,statistics):
    kind=spec['kind'];p=spec.get('parameters',{})
    if kind=='ls':return least_squares(batch['Y'],batch['P']),{}
    if kind=='lmmse':
        mean,cov=select_statistics(statistics,batch['scene_id'],scope='global')
        out=gaussian_lmmse(batch['Y'],batch['P'],batch['noise_var'],mean,cov,jitter=1e-6)
        return out,{}
    g=generator(device,seed)
    if kind=='rc':return sample_ckmsrf(model,batch,generator=g,**p)
    if kind=='cm':
        cc=cfg.get('consistency',{})
        return sample_ckmcm_pnp(model,batch,sigma_min=float(cc.get('sigma_min',.002)),sigma_max=float(cc.get('sigma_max',2)),**p)
    if kind=='pf':return sample_pnp_flow(model,batch,generator=g,**p)
    if kind=='fa':return sample_flow_admm(model,batch,generator=g,**p)
    raise ValueError(kind)

def aggregate(raw):
    cells=[]
    for method,scene,snr in sorted(set((r['method'],r['scene_id'],r['snr_db']) for r in raw)):
        rr=[r for r in raw if (r['method'],r['scene_id'],r['snr_db'])==(method,scene,snr)]
        n=sum(r['batch_size'] for r in rr)
        times=[r['receiver_ms']/r['batch_size'] for r in rr]
        inner=[r for r in rr if r['legacy_loop_ms']!='']
        cells.append(dict(method=method,scene_id=scene,snr_db=snr,timed_sample_evaluations=n,
            receiver_ms_per_sample=sum(r['receiver_ms'] for r in rr)/n,
            median_batch_amortized_ms_per_sample=float(np.median(times)),
            p95_batch_amortized_ms_per_sample=float(np.percentile(times,95)),
            legacy_loop_ms_per_sample=sum(r['legacy_loop_ms'] for r in inner)/sum(r['batch_size'] for r in inner) if inner else '',
            measured_calls=len(rr),nfe_per_recovery=rr[0]['nfe'],
            nmse_db=10*math.log10(max(sum(r['nmse_sum'] for r in rr)/n,1e-30))))
    overall=[]
    for method in sorted(set(r['method'] for r in cells)):
        cc=[r for r in cells if r['method']==method]
        n=sum(r['timed_sample_evaluations'] for r in cc)
        inner=[r for r in cc if r['legacy_loop_ms_per_sample']!='']
        overall.append(dict(method=method,scene_snr_cells=len(cc),
            macro_receiver_ms_per_sample=float(np.mean([r['receiver_ms_per_sample'] for r in cc])),
            micro_receiver_ms_per_sample=sum(r['receiver_ms_per_sample']*r['timed_sample_evaluations'] for r in cc)/n,
            legacy_macro_loop_ms_per_sample=float(np.mean([r['legacy_loop_ms_per_sample'] for r in inner])) if inner else '',
            nfe_per_recovery=cc[0]['nfe_per_recovery']))
    return cells,overall

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,default=HERE/'methods.json')
    ap.add_argument('--source-map',type=Path,help='JSON mapping method labels to moved checkpoint/statistics paths')
    ap.add_argument('--data-root',type=Path,default=Path(os.environ.get('CKMSRF_DATA_ROOT',REPO_ROOT/'data'/'ckm_srf_64x16_h5')))
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--samples-per-scene',type=int,default=32)
    ap.add_argument('--warmup',type=int,default=2)
    ap.add_argument('--repeats',type=int,default=1)
    ap.add_argument('--snr-db',type=float,nargs='+',default=list(range(-10,31,5)))
    ap.add_argument('--scenes',type=int,nargs='+',default=[1,2,3])
    ap.add_argument('--subcarrier',type=int,default=23)
    ap.add_argument('--snapshot',type=int,default=3)
    ap.add_argument('--methods',nargs='+')
    ap.add_argument('--extra-flow-m',type=int,nargs='*',default=[2,4,8,16])
    ap.add_argument('--seed',type=int,default=9102)
    ap.add_argument('--output',type=Path,default=HERE/'outputs')
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    if min(args.batch_size,args.samples_per_scene,args.repeats)<1 or args.warmup<0:ap.error('Invalid budget')
    if args.samples_per_scene%args.batch_size:ap.error('Use a multiple of batch size to avoid partial-batch bias')
    specs=json.loads(args.manifest.read_text(encoding='utf-8-sig'))['methods']
    if args.methods:
        unknown=set(args.methods)-{s['method'] for s in specs}
        if unknown:ap.error(f'Unknown methods: {unknown}')
        specs=[s for s in specs if s['method'] in args.methods]
    for m in args.extra_flow_m:
        if m<2:ap.error('Extra M must be >=2')
        for spec in list(specs):
            if spec['method'] in ['FA-U','FA-C']:
                extra=copy.deepcopy(spec);extra['method']+=f'-M{m}';extra['parameters']['k_avg']=m;specs.append(extra)
    overrides=json.loads(args.source_map.read_text()) if args.source_map else {}
    # Fail before timing when a required artifact is absent. Never silently drop a baseline.
    for s in specs:
        base=s['method'].split('-M')[0]
        s['checkpoint']=overrides.get(s['method'],overrides.get(base,s['checkpoint']))
        p=Path(s['checkpoint'])
        if not p.is_file():raise FileNotFoundError(f"{s['method']}: {p}; use --source-map")
        if s['kind']=='lmmse':
            s['statistics']=overrides.get('LMMSE-statistics',s['statistics'])
            if not Path(s['statistics']).is_file():raise FileNotFoundError(s['statistics'])
    if not args.data_root.is_dir():raise FileNotFoundError(args.data_root)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed);random.seed(args.seed);np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'run_manifest.json').exists():raise FileExistsError('Output already used; choose a fresh --output')
    env=dict(device=str(device),gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),platform=platform.platform(),
        cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),precision='fp32',tf32=False,
        timing_scope='synchronized wall time of full receiver call on resident device tensors; includes sampler diagnostics and initialization; excludes I/O, observation generation, metrics, loading weights and CPU-to-GPU transfer',
        torch_threads=1,ckmsrf_origin=str(Path(sys.modules['ckmsrf'].__file__).resolve()),
        budgets=vars(args),methods=specs,artifact_hashes={s['method']:digest(s['checkpoint']) for s in specs},
        source_hashes={str(p.relative_to(REPO_ROOT)):digest(p) for p in (REPO_ROOT/'src').rglob('*.py')},
        created_unix=time.time())
    (args.output/'run_manifest.json').write_text(json.dumps(env,default=str,indent=2),encoding='utf8')
    raw=[];identities={};counts=[0];start=time.perf_counter()
    with torch.no_grad():
        for spec in specs:
            name=spec['method'];payload=_load_checkpoint(Path(spec['checkpoint']))
            cfg=copy.deepcopy(payload['config'])
            cfg['paths']['data_root']=str(args.data_root)
            pm,po=_prior_semantics(cfg)
            if spec['kind'] in ['fa','pf'] and pm!='straight_flow_matching':raise ValueError(f'{name}: wrong prior type')
            if spec['kind']=='rc' and pm!='gmflow':raise ValueError(f'{name}: wrong prior type')
            if spec['kind']=='cm' and pm!=('diffusion' if name.startswith('Diff-') else 'consistency'):raise ValueError(f'{name}: wrong prior type')
            if spec['kind'] not in ['ls','lmmse']:
                cond=cfg.get('model',{}).get('condition',{})
                enabled=bool(cond.get('enabled',True)) and not bool(cond.get('force_unconditional',False))
                want='-U' not in name
                if enabled!=want:raise ValueError(f'{name}: checkpoint conditioning state differs')
            norm=_normalization(cfg,payload)
            model=None;statistics=None
            if spec['kind']=='lmmse':
                statistics=_load_checkpoint(Path(spec['statistics']))
                _check_normalization(norm,statistics)
                if statistics.get('split')!='train':raise ValueError('LMMSE statistics must be train-only')
                statistics=move(statistics,device)
            elif spec['kind']!='ls':
                model=build_model(cfg);_load_model_weights(model,payload,prefer_ema=True);model.to(device).eval()
            if args.check_only:
                print(f'CHECK OK {name}',flush=True);continue
            for scene in args.scenes:
                cell_cfg=copy.deepcopy(cfg);cell_cfg['data']['test_scenes']=[f'scene_{scene:02d}']
                cell_cfg['data']['test_sampling']='fixed';cell_cfg['data']['test_fixed_pairs']=[[args.subcarrier,args.snapshot]]
                index=load_or_build_index(cell_cfg,'test')
                ds=build_dataset(cell_cfg,'test',index=index,normalization=norm,include_observation=True)
                if len(ds)<args.samples_per_scene:raise ValueError(f'{scene}: insufficient test samples')
                ids=sorted(random.Random(args.seed+scene).sample(range(len(ds)),args.samples_per_scene))
                # Preload CPU data outside timed region. Homogeneous scenes and full batches.
                batches=list(DataLoader(Subset(ds,ids),batch_size=args.batch_size,num_workers=0,shuffle=False))
                ids_now=[(int(sc),int(tr),int(k),int(t)) for b in batches for sc,tr,k,t in zip(b['scene_id'],b['trajectory_id'],b['subcarrier_idx'],b['snapshot_idx'])]
                if scene in identities and identities[scene]!=ids_now:raise ValueError('Methods use different sample identities')
                identities[scene]=ids_now
                for si,snr in enumerate(args.snr_db):
                    gpu_batches=[]
                    for bi,host in enumerate(batches):
                        b=move(host,device)
                        obs,var=generate_observation(b['H'],b['P'],snr_db=snr,generator=generator(device,args.seed+scene*100000+si*1000+bi))
                        b.update(Y=obs,noise_var=var);gpu_batches.append(b)
                    if model is not None:
                        counts[0]=0
                        hook=model.register_forward_hook(lambda *a:counts.__setitem__(0,counts[0]+1))
                    # Count one probe outside timing; no forward hooks during measurements.
                    solve(spec,model,gpu_batches[0],cfg,device,args.seed,statistics)
                    nfe=counts[0] if model is not None else 0
                    if model is not None:hook.remove()
                    for wi in range(max(0,args.warmup-1)):solve(spec,model,gpu_batches[0],cfg,device,args.seed+wi+1,statistics)
                    sync(device)
                    for rep in range(args.repeats):
                        for bi,b in enumerate(gpu_batches):
                            counts[0]=0;sync(device);ts=time.perf_counter()
                            estimate,diag=solve(spec,model,b,cfg,device,10009102+si*100000+bi+rep*10000000,statistics)
                            sync(device);elapsed=(time.perf_counter()-ts)*1000
                            error=(estimate-b['H']).abs().square().flatten(1).sum(1)/b['H'].abs().square().flatten(1).sum(1).clamp_min(1e-12)
                            if not torch.isfinite(error).all():raise FloatingPointError(f'{name}: nonfinite prediction')
                            raw.append(dict(method=name,scene_id=scene,snr_db=snr,repeat=rep,batch=bi,batch_size=len(estimate),
                                receiver_ms=elapsed,legacy_loop_ms=float(diag['sampling_seconds'])*1000 if 'sampling_seconds' in diag else '',
                                nfe=nfe,nmse_sum=float(error.sum().cpu())))
                    writecsv(args.output/'batch_timings.csv',raw)
                    cells,overall=aggregate(raw);writecsv(args.output/'scene_snr_timings.csv',cells);writecsv(args.output/'method_summary.csv',overall)
                    last=next(r for r in cells if (r['method'],r['scene_id'],r['snr_db'])==(name,scene,snr))
                    print(f"{name} site={scene} snr={snr:g} receiver={last['receiver_ms_per_sample']:.4f} ms/sample elapsed={(time.perf_counter()-start)/60:.1f}min",flush=True)
                    del gpu_batches
                del batches,ds
            del model,statistics,payload
            if device.type=='cuda':torch.cuda.empty_cache()
    (args.output/'sample_identities.json').write_text(json.dumps(identities),encoding='utf8')
    if not args.check_only:
        cells,overall=aggregate(raw)
        assert len(cells)==len(specs)*len(args.scenes)*len(args.snr_db)
        for axis,outfile in [('scene_id','scene_summary.csv'),('snr_db','snr_summary.csv')]:
            rr=[]
            for method in sorted(set(r['method'] for r in cells)):
                for value in sorted(set(r[axis] for r in cells)):
                    cc=[r for r in cells if r['method']==method and r[axis]==value]
                    rr.append(dict(method=method,**{axis:value},receiver_ms_per_sample=float(np.mean([r['receiver_ms_per_sample'] for r in cc]))))
            writecsv(args.output/outfile,rr)
        ref_path=HERE/'accuracy_reference_m.csv'
        if ref_path.exists():
            ref=list(csv.DictReader(ref_path.open(encoding='utf-8-sig')));trade=[]
            for r in overall:
                if not r['method'].startswith('FA-'):continue
                base=r['method'].split('-M')[0];m=int(r['method'].split('-M')[1]) if '-M' in r['method'] else 1
                baseline=next(v for v in overall if v['method']==base)
                acc=next(v for v in ref if v['method']==base and int(v['M'])==m)
                acc1=next(v for v in ref if v['method']==base and int(v['M'])==1)
                trade.append(dict(method=base,M=m,NFE=r['nfe_per_recovery'],receiver_ms_per_sample=r['macro_receiver_ms_per_sample'],
                    time_ratio_to_M1=r['macro_receiver_ms_per_sample']/baseline['macro_receiver_ms_per_sample'],
                    archived_subset_nmse_db=acc['subset_nmse_db'],archived_full_nmse_db=acc['full_nmse_db'],
                    archived_subset_gain_over_M1_db=float(acc1['subset_nmse_db'])-float(acc['subset_nmse_db']),
                    timing_samples_per_scene=args.samples_per_scene,accuracy_subset_samples_per_snr=5997,
                    boundary='Archived accuracy cohort differs from timing subsample; no new full-test accuracy claim'))
            writecsv(args.output/'flowadmm_m_tradeoff.csv',trade)
        (args.output/'COMPLETED.json').write_text(json.dumps({'methods':len(specs),'cells':len(cells),'runtime_seconds':time.perf_counter()-start}),encoding='utf8')
    print(f'Done: {args.output}',flush=True)

if __name__=='__main__':main()
