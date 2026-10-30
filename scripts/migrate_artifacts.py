"""Migrate verified artifacts into the layout declared by ``configs/checkpoints.json``.

Default is a read-only inventory.  ``--apply`` copies only missing targets and
never overwrites a different file.  ``--mode symlink`` is optional for weights
that are kept in place.

The registry stores sanitized provenance strings (``runs/...``), so copying from
an actual training machine requires ``--source-map`` with
``{artifact_id: verified_local_path}``.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply',action='store_true')
    ap.add_argument('--mode',choices=['copy','symlink'],default='copy')
    ap.add_argument('--ids',nargs='+')
    ap.add_argument('--source-map',type=Path,help='Optional JSON {artifact_id: verified_local_path}')
    args=ap.parse_args(argv)
    records=json.loads((ROOT/'configs/checkpoints.json').read_text(encoding='utf-8'))
    overrides=json.loads(args.source_map.read_text(encoding='utf-8')) if args.source_map else {}
    report=[]; failures=0
    if args.ids and set(args.ids)-{r['id'] for r in records}:ap.error('Unknown artifact ID')
    for r in records:
        if args.ids and r['id'] not in args.ids:continue
        source=overrides.get(r['id'],r['source']); target=ROOT/r['target']
        # The destination comes from a project-owned registry, but still confine it.
        if not target.absolute().is_relative_to(ROOT.absolute()):raise ValueError(target)
        entry={'id':r['id'],'source':source,'target':str(target)}
        if not source:
            entry['status']='not_trained_no_source'; report.append(entry);print(r['id'],entry['status']);continue
        src=Path(source)
        if not src.is_absolute():src=ROOT/src
        if not src.is_file():
            entry['status']='missing_source';failures+=1
        else:
            h=digest(src);entry['sha256']=h
            if r.get('sha256') and h!=r['sha256']:
                entry['status']='source_hash_mismatch';failures+=1
            elif target.exists() or target.is_symlink():
                same=target.is_file() and digest(target)==h
                entry['status']='already_present_verified' if same else 'target_conflict_not_overwritten'
                failures+=not same
            elif args.apply:
                target.parent.mkdir(parents=True,exist_ok=True)
                if args.mode=='symlink':target.symlink_to(src.resolve())
                else:
                    temp=target.with_suffix(target.suffix+'.copying')
                    shutil.copy2(src,temp)
                    if digest(temp)!=h:raise IOError(f'Copy verification failed: {temp}')
                    temp.rename(target)
                entry['status']='copied_verified' if args.mode=='copy' else 'linked_verified'
            else:entry['status']='available_ready_to_copy'
        report.append(entry);print(r['id'],entry['status'])
    dest=ROOT/'analysis/artifact_check.json';dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Report:',dest)
    if failures:sys.exit(1)

if __name__=='__main__':main()
