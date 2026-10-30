"""Regression checks for real compatibility and experiment isolation."""
import copy
import json
from pathlib import Path
import pytest
import torch
import importlib.util
from ckmsrf.config import load_config,to_serializable
from ckmsrf.factory import build_model
from ckmsrf.evaluate import build_parser as evaluate_parser
from ckmsrf.evaluate_classical import build_parser as classical_parser
from journal_flow.cli import ROOT,parser,make_jobs,check_jobs,evaluate,registry
from test_data_synthetic import _make_scene,_make_shard


def args(*extra):
    return parser().parse_args(['plan','--device','cpu','--num-workers','0',*extra])


# Robustness checkpoints are not part of the released artifact set
# (see checkpoints/MANIFEST.json), so those planning cells are skipped instead of
# raising a KeyError on the registry.
ROBUST_SKIP=pytest.mark.skipif('robust_ckm' not in registry(),reason='robustness checkpoints are not released')


@pytest.mark.parametrize('method,track,extra,count',[
    ('flowadmm','standard',[],6),('flowadmm','standard',['--m','1','3','16'],18),
    ('flowadmm','threefold',[],6),
    pytest.param('flowadmm','robustness',['--conditions','ckm'],4,marks=ROBUST_SKIP),
    pytest.param('flowadmm','robustness',[],8,marks=ROBUST_SKIP),
    ('pnpflow','standard',[],6),('pnpflow','threefold',[],6),
    ('rcflow','standard',[],2),('srf','standard',['--conditions','ckm'],1),
    ('pnp-cm','standard',[],2),('ls','standard',[],1),('lmmse','threefold',[],3),
])
def test_commands_are_accepted_and_groups_are_separate(method,track,extra,count):
    jobs=make_jobs(args('--method',method,'--track',track,*extra))
    assert len(jobs)==count
    for j in jobs:
        p=classical_parser() if method in ['ls','lmmse'] else evaluate_parser()
        parsed=p.parse_args(j['command'][3:])
        assert parsed.split=='test'
        assert str(ROOT/('main' if method=='flowadmm' else 'baselines')) in j['output']
        if method=='flowadmm': assert parsed.flow_admm_k_avg==j['m']


def synthetic_checkpoint(tmp_path,cond='no_ckm',robust=False):
    torch.set_num_threads(1)
    data=tmp_path/'data';data.mkdir(exist_ok=True)
    for scene in [1,2,3]:
        _make_scene(data,scene,nt=4,npilot=2,banks=2)
        _make_shard(data,scene,'val',nr=4,nt=4,nsc=4,snapshots=2,trajectories=2)
        _make_shard(data,scene,'train',nr=4,nt=4,nsc=4,snapshots=2,trajectories=2)
    cfg=load_config(ROOT/f'main/configs/{"robustness" if robust else "standard"}/{cond}.yaml',
        ROOT/'configs/hardware/cpu_debug.yaml',environ={},cli_overrides=[
            'model.dim=8','model.dim_mults=[1,2]','model.condition.tx_beam_dim=4',
            'model.condition.rx_beam_dim=4','model.condition.adp_shape=[2,2,3]',
            'data.nr=4','data.nt=4','data.n_subcarriers=4','data.n_snapshots=2',
            'data.pilot_length=2'])
    ckpt=tmp_path/'synthetic.pt'
    model=build_model(cfg)
    torch.save({'model':model.state_dict(),'config':to_serializable(cfg),
                'normalization':{'h_rms':4000.,'split':'train'}},ckpt)
    return ckpt,data


def test_load_evaluate_and_idempotent_restart_with_real_synthetic_data(tmp_path):
    ckpt,data=synthetic_checkpoint(tmp_path)
    a=args('--conditions','no_ckm','--seeds','10009102','--checkpoint',str(ckpt),
           '--data-root',str(data),'--split','val','--iterations','2','--m','3',
           '--max-samples','1','--snr-db','10','--batch-size','1','--output-root',str(tmp_path/'out'))
    jobs=make_jobs(a);check_jobs(jobs,a);evaluate(jobs,a)
    result=Path(jobs[0]['result_dir'])/'results.json'
    before=result.read_bytes()
    a.resume=True;evaluate(jobs,a)
    assert result.read_bytes()==before
    wrong=copy.deepcopy(a);wrong.conditions=['ckm']
    with pytest.raises(ValueError,match='CKM condition'):
        check_jobs(make_jobs(wrong),wrong)


def test_robust_unconditional_is_truly_condition_invariant(tmp_path):
    ckpt,_=synthetic_checkpoint(tmp_path,robust=True)
    payload=torch.load(ckpt,weights_only=False)
    model=build_model(payload['config']).eval();model.load_state_dict(payload['model'])
    x=torch.randn(2,2,4,4);t=torch.ones(2)*.5
    cond={'scalar':torch.randn(2,6),'state':torch.randn(2,8),
          'tx_beam':torch.randn(2,4),'rx_beam':torch.randn(2,4),'adp':torch.randn(2,2,2,3)}
    with torch.no_grad():
        first=model(x,t,cond);second=model(x,t,None)
    torch.testing.assert_close(first,second,rtol=0,atol=0)


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_classical_results_and_paired_summary(tmp_path):
    from ckmsrf.fit_channel_stats import main as fit
    ckpt,data=synthetic_checkpoint(tmp_path)
    stats=tmp_path/'stats.pt'
    fit(['--protocol-checkpoint',str(ckpt),'--data-root',str(data),'--output',str(stats),
         '--device','cpu','--num-workers','0','--max-samples','2',
         '--set','hardware.use_accelerate=false','--set','data.backend=plain_h5'])
    for method in ['ls','lmmse']:
        a=args('--method',method,'--seeds','10009102','--checkpoint',str(ckpt),
               '--data-root',str(data),'--split','val','--max-samples','1','--snr-db','10',
               '--batch-size','1','--output-root',str(tmp_path/'outputs'))
        jobs=make_jobs(a)
        if method=='lmmse':
            cmd=jobs[0]['command'];cmd[cmd.index('--statistics')+1]=str(stats)
        check_jobs(jobs,a);evaluate(jobs,a)
        assert (Path(jobs[0]['result_dir'])/'metrics.csv').is_file()
    summary=load_script('summarize')
    summary.main(['--roots',str(tmp_path/'outputs'),'--output',str(tmp_path/'summary')])
    assert (tmp_path/'summary/full_test/per_snr.csv').is_file()
    # A changed observation fingerprint must not pass as a paired comparison.
    files=list((tmp_path/'outputs').rglob('subspace_per_sample.csv'))
    import csv
    with files[0].open(encoding='utf-8') as f: records=list(csv.DictReader(f))
    records[0]['problem_sha256']='deliberately_different'
    with files[0].open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    with pytest.raises(ValueError,match='Unpaired'):
        summary.main(['--roots',str(tmp_path/'outputs'),'--output',str(tmp_path/'summary')])


def test_migration_copies_and_never_overwrites_different_weights(tmp_path,monkeypatch):
    tool=load_script('migrate_artifacts');monkeypatch.setattr(tool,'ROOT',tmp_path)
    source=tmp_path/'original.pt';source.write_bytes(b'original weight data')
    cfg=tmp_path/'configs';cfg.mkdir()
    (cfg/'checkpoints.json').write_text(json.dumps([{'id':'sample','source':str(source),
        'target':'main/checkpoints/sample.pt','sha256':tool.digest(source)}]))
    tool.main([]);target=tmp_path/'main/checkpoints/sample.pt';assert not target.exists()
    tool.main(['--apply']);assert target.read_bytes()==source.read_bytes()
    target.write_bytes(b'new independent run')
    with pytest.raises(SystemExit):tool.main(['--apply'])
    assert target.read_bytes()==b'new independent run'
