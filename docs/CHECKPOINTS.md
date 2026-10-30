# Checkpoint reference

This document is the technical reference for the released KAP-Flow checkpoints:
format, provenance, EMA semantics, verification, and how to reproduce each
paper result. For the download quick start see
[`checkpoints/README.md`](../checkpoints/README.md); for the machine-readable
record see [`checkpoints/MANIFEST.json`](../checkpoints/MANIFEST.json).

- Release tag: **`v1.0-checkpoints`**
- 17 checkpoint files from 14 Release assets: 13 network checkpoints (inference
  exports) + 4 LMMSE statistics inside one zip + `SHA256SUMS.txt`
- One shared flow prior per conditioning setting serves both KAP-Flow (FlowADMM)
  and PnP-Flow
- Paper status: under review (do not cite as accepted/published)

---

## 1. Format of the released files

### 1.1 Original training checkpoint (authors' runs)

Each `best_nmse.pt` written by `src/ckmsrf/engine/checkpoint.py` is a `torch.save`
mapping of 14 entries, 61.6–64.9 MB:

| Key | Content | Needed for evaluation? |
| --- | --- | --- |
| `format_version` | checkpoint format (`1`) | yes (forward compatibility) |
| `model` | 258 tensors, 4,010,818 parameters (fp32, ~15.3 MB) | yes — loaded strictly |
| `ema` | `{decay, update_after_step, num_updates, shadow}`; `shadow` has the same 258 tensors (~15.3 MB) | yes — **this is what evaluation uses** |
| `optimizer` | Adam `state` + `param_groups` (~30.6 MB) | no (training resume only) |
| `scheduler` | cosine LR scheduler state | no (training resume only) |
| `scaler` | AMP GradScaler state (`None` in these runs) | no |
| `training_state` | `{epoch, global_step, best_metrics}` | provenance only |
| `epoch`, `global_step`, `best_metrics` | duplicated at top level for convenience | provenance only |
| `rng_state` | python/numpy/torch RNG state | no (training resume only) |
| `config` | full resolved training config (experiment, paths, data, model, loss, sampling, optimizer, scheduler, training, hardware) | **yes** — restores architecture and scientific settings |
| `normalization` | `{h_rms, feature_mean, feature_std, split, num_trajectories, ...}` | **yes** — evaluation uses the statistics stored in the checkpoint |
| `extra` | e.g. `{selected_metric, selected_value}` | provenance only |

### 1.2 Released inference checkpoint

Produced by [`tools/export_inference_ckpt.py`](../tools/export_inference_ckpt.py),
30.8–31.0 MB each (49.6 % smaller):

- **kept:** `format_version`, `model`, `ema`, `config`, `normalization`,
  `training_state`, `epoch`, `global_step`, `best_metrics`, `extra`
- **dropped:** `optimizer`, `scheduler`, `scaler`, `rng_state`
  (`--keep-rng` retains the RNG state if needed)
- **added:** a deterministic provenance marker
  `extra.inference_export = {tool, tool_version, source_sha256, kept_keys,
  dropped_keys, note}`. It records the SHA256 of the original training file and
  makes it impossible to mistake the file for a resumable training checkpoint.
  The marker contains no timestamps, so re-running the tool on the same input
  reproduces the same bytes and therefore the same SHA256.

Verification performed for every one of the 13 released weights (recorded in
`kap_flow_release_work/exported/export_report.json` during the release build):

1. round-trip: the exported file is reloaded and all **258 `model` tensors and
   258 `ema.shadow` tensors are `torch.equal`** to the source, and the
   `format_version` / `config` / `normalization` / `training_state` /
   `epoch` / `global_step` / `best_metrics` slots are unchanged;
2. real loader: `ckmsrf.factory.build_model(config)` +
   `ckmsrf.evaluate._load_model_weights(prefer_ema=True)` load the file and the
   resulting parameters match `ema.shadow` tensor by tensor; repeating with
   `prefer_ema=False` matches `model`;
3. registry: `python run.py check --method {flowadmm,pnpflow,rcflow,srf,pnp-cm,ls,lmmse}
   --track {standard,threefold} --data-root <existing dir> --device cpu` accepts
   the downloaded files (weights, condition/fold metadata and SHA256 guards).

### 1.3 Which weights does evaluation actually use? → EMA

`src/ckmsrf/evaluate.py::_load_model_weights` is the evaluation loader:

```python
state = payload.get("model", payload.get("state_dict", payload))
model.load_state_dict(state, strict=True)
ema = payload.get("ema")
if prefer_ema and isinstance(ema, Mapping) and isinstance(ema.get("shadow"), Mapping):
    current = model.state_dict()
    for name, value in ema["shadow"].items():
        if name in current:
            current[name].copy_(value.to(dtype=current[name].dtype))
    return "ema"
```

`prefer_ema` is `not args.use_model_weights`, i.e. **EMA weights are the default
and are what the reported NMSE numbers use**. `--use-model-weights` switches to
the raw `model` weights for ablations; the released files contain both, so no
information is lost. (`CheckpointManager.load` in
`src/ckmsrf/engine/checkpoint.py` is a different path used for training resume;
it loads `model` first and then the EMA state if an `EMA` object is supplied.)

### 1.4 Can I resume training from the released files?

No, not *exactly*: the Adam moments, LR-scheduler state, AMP scaler state and
RNG state were dropped to halve the size. `CheckpointManager.load` tolerates
their absence (it skips missing slots), so resuming would silently restart the
optimizer with fresh moments. These files are for evaluation/analysis only. The
retained `training_state`, `epoch`, `global_step`, `best_metrics` and
`extra.selected_value` document the training progress at selection time.

---

## 2. Provenance

`source_original_path` in `MANIFEST.json` is relative to the authors' training
run roots (`runs/<experiment>/<timestamp>/...`); machine-local absolute paths
are intentionally not published. `source_relative_path` is the original path
inside the authors' project checkout. Every file was copied from that path and
its SHA256 (`sha256_original`) is recorded before export.

| Checkpoint ID | Source run (relative) | Training / protocol config |
| --- | --- | --- |
| `straight_ckm` | `runs/ckm_straight_flow_prior/20260901-200938_seed3407/checkpoints/best_nmse.pt` | `main/configs/standard/ckm.yaml` |
| `straight_no_ckm` | `runs/straight_flow_no-ckm_in_domain/paper_straight_factor_20260904_in_domain_no-ckm/checkpoints/best_nmse.pt` | `main/configs/standard/no_ckm.yaml` |
| `straight_ckm_fold1` | `runs/straight_flow_ckm_holdout_scene_01/paper_straight_cross_scene_20260904_holdout_scene_01_ckm/checkpoints/best_nmse.pt` | `main/configs/threefold/holdout_01_ckm.yaml` |
| `straight_no_ckm_fold1` | `runs/straight_flow_no-ckm_holdout_scene_01/paper_straight_cross_scene_20260904_holdout_scene_01_no-ckm/checkpoints/best_nmse.pt` | `main/configs/threefold/holdout_01_no_ckm.yaml` |
| `straight_ckm_fold2` | `runs/straight_flow_ckm_holdout_scene_02/paper_straight_cross_scene_20260904_holdout_scene_02_ckm/checkpoints/best_nmse.pt` | `main/configs/threefold/holdout_02_ckm.yaml` |
| `straight_no_ckm_fold2` | `runs/straight_flow_no-ckm_holdout_scene_02/paper_straight_cross_scene_20260904_holdout_scene_02_no-ckm/checkpoints/best_nmse.pt` | `main/configs/threefold/holdout_02_no_ckm.yaml` |
| `straight_ckm_fold3` | `runs/straight_flow_ckm_holdout_scene_03/paper_straight_cross_scene_20260904_holdout_scene_03_ckm/checkpoints/best_nmse.pt` | `main/configs/threefold/holdout_03_ckm.yaml` |
| `straight_no_ckm_fold3` | `runs/straight_flow_no-ckm_holdout_scene_03/paper_straight_cross_scene_20260904_holdout_scene_03_no-ckm/checkpoints/best_nmse.pt` | `main/configs/threefold/holdout_03_no_ckm.yaml` |
| `rcflow_no_ckm` | `runs/baseline_no_ckm/20260716-212304_seed3407/checkpoints/best_nmse.pt` | `baselines/configs/rcflow_no_ckm.yaml` |
| `rcflow_ckm` | `runs/ckmsrf_prior/20260716-113417_seed3407/checkpoints/best_nmse.pt` | `baselines/configs/rcflow_ckm.yaml` |
| `srf_ckm` | `runs/ckmsrf_full/20260716-211753_seed3407/checkpoints/best_nmse.pt` | `baselines/configs/srf_ckm.yaml` |
| `pnp_cm_ckm` | `runs/ckmcm_prior/20260716-112220_seed3407/checkpoints/best_nmse.pt` | `baselines/configs/pnp_cm_ckm.yaml` |
| `pnp_cm_no_ckm` | `runs/ckmcm_no_ckm/20260717-174041_seed3407/checkpoints/best_nmse.pt` | `baselines/configs/pnp_cm_no_ckm.yaml` |
| `lmmse` | `runs/studies/solver_subspace/20260913_fixed/statistics/lmmse_train_only.pt` | protocol: `main/configs/standard/ckm.yaml` |
| `lmmse_fold1` | `runs/studies/straight_flow/paper_straight_cross_scene_20260904/statistics/holdout_scene_01/lmmse_train_only.pt` | protocol: `main/configs/threefold/holdout_01_ckm.yaml` |
| `lmmse_fold2` | `runs/studies/straight_flow/paper_straight_cross_scene_20260904/statistics/holdout_scene_02/lmmse_train_only.pt` | protocol: `main/configs/threefold/holdout_02_ckm.yaml` |
| `lmmse_fold3` | `runs/studies/straight_flow/paper_straight_cross_scene_20260904/statistics/holdout_scene_03/lmmse_train_only.pt` | protocol: `main/configs/threefold/holdout_03_ckm.yaml` |

`best_nmse.pt` means the checkpoint that minimized validation reconstruction
NMSE during training (`training_state.best_metrics.val_reconstruction_nmse`,
also recorded as `extra.selected_metric` / `extra.selected_value`). Seed 3407 is
recorded in every run directory name. The LMMSE files are second-order statistics
computed from the **training split only** (`lmmse_train_only.pt`), so they are
valid under both the in-domain and site-transfer protocols.

---

## 3. Download and verification

```bash
# from the repository root
python tools/download_checkpoints.py --tag v1.0-checkpoints --dest checkpoints/

# options: --only <id...>  --dry-run  --force  --from-dir <asset dir>
```

The script uses only the standard library, downloads into
`checkpoints/.download/`, checks each asset SHA256 against
`checkpoints/MANIFEST.json`, installs files at their `target_relative_path`
(the `configs/checkpoints.json` layout), unpacks `lmmse_statistics.zip`, and
re-verifies every installed file. Any mismatch aborts with exit code 1 and does
not install the file.

Manual verification of an installed tree:

```bash
cd checkpoints && sha256sum -c SHA256SUMS.txt
```

A useful sanity check that a released file is a valid checkpoint:

```bash
python - <<'PY'
import torch
ck = torch.load("checkpoints/main/straight/ckm/best_nmse.pt", map_location="cpu", weights_only=False)
print(sorted(ck))                                   # format_version, model, ema, config, ...
print(len(ck["model"]), len(ck["ema"]["shadow"]))   # 258 258
print(bool(ck["model"].keys() == ck["ema"]["shadow"].keys()))
PY
```

Expected tree after download (17 files, 406.9 MiB):

```text
checkpoints/
├── MANIFEST.json
├── README.md
├── SHA256SUMS.txt
├── main/
│   ├── straight/{ckm,no_ckm}/best_nmse.pt
│   └── threefold/holdout_0{1,2,3}/{ckm,no_ckm}/best_nmse.pt
├── baselines/
│   ├── rcflow_ckm/best_nmse.pt
│   ├── rcflow_no_ckm/best_nmse.pt
│   ├── srf_ckm/best_nmse.pt
│   ├── pnp_cm_ckm/best_nmse.pt
│   └── pnp_cm_no_ckm/best_nmse.pt
└── baselines_statistics/
    ├── in_domain/lmmse_train_only.pt
    └── holdout_0{1,2,3}/lmmse_train_only.pt
```

---

## 4. Reproducing the paper results

A dataset is required for NMSE evaluation and is **not** distributed with the
repository; see [`docs/DATA.md`](DATA.md), `data_generation/`, and pass
`--data-root` (or set `CKMSRF_DATA_ROOT`). All commands below are run from the
repository root. `run.py` resolves checkpoint paths through
`configs/checkpoints.json`, so the files must be downloaded to
`checkpoints/` first. Solver defaults follow the paper: `K = 40`,
`M = 1`, nine SNR points `-10 … 30 dB`, validation seed `9102`.

### 4.1 Table I — in-domain NMSE and timing

| Paper row (Table I) | Checkpoint file | Command |
| --- | --- | --- |
| KAP-C | `checkpoints/main/straight/ckm/best_nmse.pt` | `python run.py evaluate --method flowadmm --track standard --conditions ckm --data-root <root>` |
| KAP-U | `checkpoints/main/straight/no_ckm/best_nmse.pt` | `python run.py evaluate --method flowadmm --track standard --conditions no_ckm --data-root <root>` |
| PF-C | `checkpoints/main/straight/ckm/best_nmse.pt` | `python run.py evaluate --method pnpflow --track standard --conditions ckm --data-root <root>` |
| PF-U | `checkpoints/main/straight/no_ckm/best_nmse.pt` | `python run.py evaluate --method pnpflow --track standard --conditions no_ckm --data-root <root>` |
| RC-C | `checkpoints/baselines/rcflow_ckm/best_nmse.pt` | `python run.py evaluate --method rcflow --track standard --conditions ckm --data-root <root>` |
| RC-U | `checkpoints/baselines/rcflow_no_ckm/best_nmse.pt` | `python run.py evaluate --method rcflow --track standard --conditions no_ckm --data-root <root>` |
| RC-S | `checkpoints/baselines/srf_ckm/best_nmse.pt` | `python run.py evaluate --method srf --conditions ckm --data-root <root>` |
| CM-C | `checkpoints/baselines/pnp_cm_ckm/best_nmse.pt` | `python run.py evaluate --method pnp-cm --track standard --conditions ckm --data-root <root>` |
| CM-U | `checkpoints/baselines/pnp_cm_no_ckm/best_nmse.pt` | `python run.py evaluate --method pnp-cm --track standard --conditions no_ckm --data-root <root>` |
| LMMSE | `checkpoints/baselines_statistics/in_domain/lmmse_train_only.pt` | `python run.py evaluate --method lmmse --track standard --data-root <root>` |
| LS | *(no checkpoint)* | `python run.py evaluate --method ls --track standard --data-root <root>` |

`run.py evaluate` writes results under `main/results/` or `baselines/results/`
(use `--output-root` to change it) and runs the planned jobs for the default
inference seeds (`10009102–10009104` for KAP-Flow/PnP-Flow; one seed for the
deterministic baselines). Pass `--device cpu` if no GPU is available and
`--max-samples` for a quick smoke test.

Timing (Table I right-most column, 40 network evaluations, `M = 1`) is measured
from a separate timing manifest after the checkpoints are downloaded:

```bash
python run_paper.py timing --device cuda:0
```

(`run_paper.py timing` forwards to `timing/benchmark_all.py --manifest
timing/methods.json`; the manifest uses the same `checkpoints/...` paths.)

### 4.2 Fig. 3 — where the gains live (pilot-observable vs pilot-null)

Uses the same four in-domain files as the KAP/PF rows above. The evaluation
driver requests subspace metrics by default (`--subspace-metrics`, see
`main/results/**/subspace_metrics.json`); to run a reduced experiment manually:

```bash
python -m ckmsrf.evaluate \
  --checkpoint checkpoints/main/straight/ckm/best_nmse.pt \
  --data-root <root> --split test --subspace-metrics --device cuda:0 \
  --sampler flow-admm --snr-db -10 -5 0 5 10 15 20 25 30
```

### 4.3 Fig. 4(a) — site transfer (threefold holdouts)

Each fold trains on two sites and evaluates on the held-out site with its
historical CKM; the LMMSE statistics for that fold are source-only.

```bash
python run.py evaluate --method flowadmm --track threefold --folds 1 2 3 \
  --data-root <root>                       # KAP-C / KAP-U on holdout scenes
python run.py evaluate --method pnpflow   --track threefold --folds 1 2 3 \
  --data-root <root>                       # PF-C / PF-U on holdout scenes
python run.py evaluate --method lmmse     --track threefold --folds 1 2 3 \
  --data-root <root>                       # LMMSE reference per fold
```

This uses `checkpoints/main/threefold/holdout_0X/{ckm,no_ckm}/best_nmse.pt`
and `checkpoints/baselines_statistics/holdout_0X/lmmse_train_only.pt`.

### 4.4 Fig. 4(b) — endpoint averaging (sampling budget)

Uses the in-domain CKM prior with `M` endpoint samples (paper: `M = 1` default,
`M = 16` for the accuracy/cost point; run time follows the number of network
evaluations):

```bash
python run.py evaluate --method flowadmm --track standard --conditions ckm \
  --m 1 16 --data-root <root>
```

### 4.5 Quick validation without data

`run.py check` validates the downloaded files, their SHA256 against
`configs/checkpoints.json`, the checkpoint `config` semantics (CKM/unconditional
architecture, fold split) and the runtime; it only requires the data root path to
exist, so an empty directory is enough for this metadata-level check:

```bash
python run.py check --method flowadmm --track standard \
  --conditions ckm no_ckm --data-root <existing empty dir> --device cpu
```

---

## 5. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `HTTP 404` on download | Wrong `--tag`/asset name, or the Release has not been uploaded yet. Check the tag in `MANIFEST.json.release_tag`; use `--from-dir` for a local mirror. |
| `SHA256 mismatch` | Truncated/corrupted download or a different file: nothing is installed, rerun with `--force`. The expected hashes are in `MANIFEST.json` / `SHA256SUMS.txt`. |
| `FileNotFoundError: <data root>` | Evaluation needs the dataset; pass `--data-root` or set `CKMSRF_DATA_ROOT`. See `docs/DATA.md`. |
| `differs from recorded paper checkpoint SHA256` (`run.py check`) | The files under `checkpoints/` do not match `configs/checkpoints.json`; re-run the download script (or `--force`). Do not hand-edit those files. |
| Wrong/broken results after editing checkpoints | Never modify the `.pt` files; re-export with `tools/export_inference_ckpt.py` from an original training checkpoint, or download again. |
| Windows `sha256sum` not found | Use Git Bash/WSL, or rely on the download script, which verifies SHA256 on Windows too. |
| `--use-model-weights` changes results | Expected: the paper numbers use `ema.shadow`; that flag is for ablations only. |
