"""Aggregate completed curated jobs; preserve full tests and optional post-hoc sensitivity separately."""
import argparse
import csv
import json
import math
from pathlib import Path
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={(1,41),(2,2094),(2,2844)}

def average(values):return sum(values)/len(values)
def db(value):return 10*math.log10(max(value,1e-300))
def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs='+',default=[ROOT/'main/results',ROOT/'baselines/results'])
    p.add_argument('--output',type=Path,default=ROOT/'analysis/current_summary')
    p.add_argument('--posthoc-exclusions',action='store_true',help='Explicit sensitivity-only mask; applies equally to all selected jobs')
    a=p.parse_args(argv)
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[]; pairing={};audit=[]
    for base in a.roots:
        for done in base.rglob('curated_complete.json'):
            job=json.loads((done.parent/'curated_job.json').read_text(encoding='utf-8'))
            result=Path(job['result_dir'])
            # Support moving the complete tree between computers.
            if not result.is_dir():
                result=done.parent
                if not (result/'results.json').exists():result=done.parent/job['label'].split('/')[1]
            per=result/'subspace_per_sample.csv'
            if not per.is_file():raise FileNotFoundError(f'{per}: subspace records required for paired summary')
            groups=defaultdict(list);fingerprints={}
            with per.open(encoding='utf-8-sig') as f:
                for r in csv.DictReader(f):
                    if a.posthoc_exclusions and (int(r['scene_id']),int(r['trajectory_id'])) in EXCLUDED:continue
                    key=tuple(r[k] for k in ['snr_db','scene_id','trajectory_id','snapshot_idx','subcarrier_idx','sample_seed'])
                    if key in fingerprints:raise ValueError(f'Duplicate sample in {per}: {key}')
                    fingerprints[key]=r['problem_sha256'];groups[float(r['snr_db'])].append(r)
            cmd=job['command'];split=cmd[cmd.index('--split')+1]
            pairkey=(job['fold'],split)
            if pairkey in pairing and pairing[pairkey]!=fingerprints:
                raise ValueError(f'Unpaired samples or observations: {per}. Summarize different cohorts separately.')
            pairing[pairkey]=fingerprints
            method=job['label'].split('/')[1]
            for snr,values in sorted(groups.items()):
                metrics={k:average([float(r[k]) for r in values]) for k in ['nmse_linear','e_obs','e_null']}
                rows.append(dict(track=job['label'].split('/')[0],method=method,condition=job['condition'],
                                 prior=job['prior'],mode=job['mode'],fold=job['fold'],M=job['m'],seed=job['seed'],
                                 snr_db=snr,num_samples=len(values),**metrics,nmse_db=db(metrics['nmse_linear'])))
            audit.append({'job':job['label'],'samples':len(fingerprints),'pairing':'exact_problem_sha256_match'})
    if not rows:raise ValueError('No completed curated jobs found')
    cohort='posthoc_sensitivity' if a.posthoc_exclusions else 'full_test'
    target=a.output/cohort;target.mkdir(exist_ok=True)
    write_csv(target/'per_seed_snr.csv',rows)
    dims=['track','method','condition','prior','mode','fold','M','snr_db']
    seeds=defaultdict(list)
    for r in rows:seeds[tuple(r[k] for k in dims)].append(r)
    per_snr=[]
    for key,rs in seeds.items():
        if len({r['seed'] for r in rs})!=len(rs):raise ValueError('Duplicate completed job/seed')
        metric={k:average([r[k] for r in rs]) for k in ['nmse_linear','e_obs','e_null']}
        per_snr.append({**dict(zip(dims,key)),'num_seeds':len(rs),**metric,'nmse_db':db(metric['nmse_linear'])})
    write_csv(target/'per_snr.csv',per_snr)
    folds=defaultdict(list)
    for r in per_snr:folds[tuple(r[k] for k in dims[:-1])].append(r)
    aggregates=[]
    for key,rs in folds.items():
        metric={k:average([r[k] for r in rs]) for k in ['nmse_linear','e_obs','e_null']}
        aggregates.append({**dict(zip(dims[:-1],key)),'num_snr':len(rs),**metric,'nmse_db':db(metric['nmse_linear'])})
    write_csv(target/'per_fold_aggregate.csv',aggregates)
    macro=defaultdict(list)
    for r in aggregates:
        if r['track']=='threefold':macro[tuple(r[k] for k in ['method','condition','prior','mode','M'])].append(r)
    macros=[]
    for key,rs in macro.items():
        if {r['fold'] for r in rs}!={1,2,3}:raise ValueError('Threefold macro requires all three folds')
        n=average([r['nmse_linear'] for r in rs])
        macros.append({**dict(zip(['method','condition','prior','mode','M'],key)),'nmse_linear':n,'nmse_db':db(n)})
    write_csv(target/'threefold_macro.csv',macros)
    (target/'pairing_audit.json').write_text(json.dumps({'cohort':cohort,'jobs':audit,
       'excluded_trajectories':sorted(EXCLUDED) if a.posthoc_exclusions else [],
       'scope':'Means are taken in linear scale: samples, inference seeds, SNR, then equal-weight folds. No training-seed uncertainty claim.'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(target)

if __name__=='__main__':main()
