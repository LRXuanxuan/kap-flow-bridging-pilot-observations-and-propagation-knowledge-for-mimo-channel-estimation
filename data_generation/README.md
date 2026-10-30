# CKM-SRF unified dataset generator

MATLAB data generator for the three-site mobile mmWave MIMO dataset used in
KAP-Flow. The generator is self-contained (base MATLAB only, **no 5G Toolbox**)
and synthesises, per site:

- a unified wideband MIMO channel tensor `H(rx, tx, subcarrier, snapshot, sample)`;
- a location-indexed CKM built **only** from independent historical channel
  samples, plus bilinear CKM queries along each trajectory;
- finite-pilot observations `Y = H P + N` with semi-unitary DFT pilot matrices;
- path-level ground truth and UE state, and a spatially disjoint
  train/validation/test split.

KAP-Flow reads the exported **plain HDF5** shards directly
(`data.backend: plain_h5`); no `.mat` conversion is needed. The raw MATLAB
`.mat` shards remain readable through the optional `mat_shard_stream` backend.
The data contract and the paper data setting are documented in
[`../docs/DATA.md`](../docs/DATA.md).

> The dataset itself (hundreds of GiB) is **not** distributed with the code.
> Generate it locally with this generator, or point `CKMSRF_DATA_ROOT` at a
> matching HDF5 root. A Chinese version of this document (with local paths from
> the upstream generator README removed) is provided as
> [`README.zh.md`](README.zh.md).

## What the generator does

- **Unified channel dimension** `H(rx, tx, subcarrier, snapshot, sample)`, with
  `(N_t, N_r, N_f, N_s) = (64, 16, 48, 8)` for the paper configuration.
- **Geometry-defined sites**: fixed scatterers and blockers, LoS/NLoS paths,
  half-wavelength `8x8` transmit and `4x4` receive UPAs, `f_c = 28 GHz`,
  `Delta f = 15 kHz`.
- **High-mobility sequences**: constant-velocity UE trajectories
  (`20--60 m/s`), fixed physical path IDs over one trajectory, time-varying
  delays and carrier Doppler.
- **Independent historical CKM**: each CKM node aggregates 32 historical
  channel samples into a `16x16x12` angular-delay power profile plus 6 scalar
  features and 64/16 transmit/receive beam probabilities. Online channels are
  never used to build the map.
- **Spatially disjoint splits**: `40 m x 40 m` blocks split `15/5/5` for
  train/validation/test; every snapshot of a trajectory stays in one split.
- **Finite-pilot observations** `Y = H P + N`, with `P^H P = I_{N_p}`,
  `N_p = 19` for the paper configuration; each record stores its pilot index,
  SNR, and noise variance.
- **Shard-based output**: bounded memory, a `manifest.mat`, and explicit HDF5
  export for Python.
- **Optional CFO/ICI operator** and its exact adjoint for high-mobility
  generalisations (disabled for the paper dataset: `cfg.cfo.enable = false`).

Field-by-field generator schema: [`docs/SCHEMA.md`](docs/SCHEMA.md).
Historical SRF-era integration notes (superseded for KAP-Flow by
[`../docs/DATA.md`](../docs/DATA.md)): [`docs/SRF_INTEGRATION.md`](docs/SRF_INTEGRATION.md).

## Layout

```
data_generation/
├── run_demo.m                      # small end-to-end demo
├── generate_ckm_srf_dataset.m      # top-level generator (writes .mat shards)
├── ckmsrf_default_config.m         # smoke-test defaults
├── configs/research_config.m       # paper-scale configuration
├── export_scene_to_hdf5.m          # scene .mat  -> plain HDF5
├── export_shard_to_hdf5.m          # shard .mat  -> plain HDF5
├── export_all_to_hdf5.m            # batch wrapper for a whole output directory
├── src/*.m                         # geometry, CKM, trajectory, pilot, I/O helpers
├── tests/run_all_tests.m           # physics and schema tests (MATLAB)
├── python/
│   ├── ckmsrf_h5_dataset.py        # minimal NumPy HDF5 reader
│   ├── example_read.py             # reader example
│   ├── test_h5_loader.py           # reader smoke test
│   ├── requirements.txt            # numpy + h5py
│   └── prepare_kapflow_data.py     # validate/export-prep for the KAP-Flow loader
├── docs/SCHEMA.md                  # generator field schema
└── docs/SRF_INTEGRATION.md         # historical integration note
```

## Requirements

- **MATLAB** with HDF5 support (`save(..., '-v7.3')`, `h5create`, `h5write`).
  No toolbox is required.
- **Python >= 3.10** for the HDF5 readers and the KAP-Flow data loader:
  `python -m pip install -r python/requirements.txt`.

## Quick start (MATLAB demo)

```matlab
cd('<repo>/data_generation')
run_demo                                % 1 scene, 12/6/6 train/val/test records
run('tests/run_all_tests.m')            % physics + schema checks
```

The demo writes `output_demo/` (scene package, shards, `manifest.mat`) inside
the `data_generation/` directory. Demo outputs are excluded from git.

Export the demo files to plain HDF5 and smoke-test the Python reader (run the
shell commands from the repository root):

```matlab
export_all_to_hdf5('<repo>/data_generation/output_demo');   % writes output_demo_h5/
```

```bash
python -m pip install -r data_generation/python/requirements.txt
python data_generation/python/test_h5_loader.py \
    data_generation/output_demo_h5/scene_01_train_shard_001.h5 \
    data_generation/output_demo_h5/scene_01_ckm.h5
```

## Generating the paper-scale dataset

`configs/research_config.m` reproduces the settings reported in Sec. IV of the
manuscript (three sites, `200 m x 200 m`, 10 m CKM grid, `15/5/5` block split,
`10000/1000/2000` records per site, 16 DFT pilot matrices, SNR grid
`-10:5:30` dB); see [`../docs/DATA.md`](../docs/DATA.md) for the complete table.

```matlab
addpath('<repo>/data_generation');
addpath('<repo>/data_generation/configs');
addpath('<repo>/data_generation/src');

% 1) smoke test the I/O path with tiny counts first (separate directory)
smoke = research_config('<datasets>/ckm_srf_smoke');
smoke.dataset.samplesPerScene.train = 10;
smoke.dataset.samplesPerScene.val   = 2;
smoke.dataset.samplesPerScene.test  = 2;
smoke.overwrite = true;
manifest = generate_ckm_srf_dataset(smoke);

% 2) then run the full paper-scale generation (hours to days, see disk estimate)
cfg = research_config('<datasets>/ckm_srf_64x16');
manifest = generate_ckm_srf_dataset(cfg);
```

Keep `cfg.output.storeTrueChannel = true`. `cfg.output.storeObservation` may
stay `false`: KAP-Flow synthesises `Y` online from the clean channel and the
pilot bank, while the per-record pilot/SNR/noise metadata is always stored.
Storage grows with the clean channel tensor; one record is exactly
`3 MiB` for `48 x 8` subcarriers/snapshots and `16 x 64` antennas complex
single precision (about `29.3 GiB` for one site's 10,000 training records,
before HDF5 overhead).

Export the whole output directory to plain HDF5:

```matlab
export_all_to_hdf5('<datasets>/ckm_srf_64x16')   % -> <datasets>/ckm_srf_64x16_h5
```

## Feeding the data to KAP-Flow

KAP-Flow locates the data root through `CKMSRF_DATA_ROOT` (or `--data-root`) and
expects the HDF5 files at the top level of that directory:

```
<data root>/
├── scene_01_ckm.h5                 # pilot bank + offline CKM (one per scene)
├── scene_01_train_shard_001.h5     # ... train shards
├── scene_01_val_shard_001.h5       # ... validation shards
├── scene_01_test_shard_001.h5      # ... test shards
├── scene_02_ckm.h5  scene_02_*_shard_*.h5
├── scene_03_ckm.h5  scene_03_*_shard_*.h5
├── indices/{train,val,test}.json   # optional, built by the preparation step
└── normalization.json              # optional, train-only statistics
```

The loader can build `indices/` and missing statistics automatically. To make
the preparation explicit and reproducible, run the bundled preparation script
(the KAP-Flow replacement for the SRF-era `prepare_srf_mat.py`; no format
conversion is involved):

```bash
# from the repository root; install the package first: pip install -e .
export CKMSRF_DATA_ROOT="<datasets>/ckm_srf_64x16_h5"
python data_generation/python/prepare_kapflow_data.py --data-root "$CKMSRF_DATA_ROOT"
```

```powershell
# PowerShell equivalent
$env:CKMSRF_DATA_ROOT = "<datasets>\ckm_srf_64x16_h5"
python data_generation/python/prepare_kapflow_data.py --data-root $env:CKMSRF_DATA_ROOT
```

The script validates the exported HDF5 files, fixes the MATLAB root attributes
(see *Compatibility notes*), writes `indices/{split}.json`, computes
`normalization.json` from the train split only, and prints the next command.

Train the conditional KAP-Flow prior:

```bash
python -m ckmsrf.train \
    --config configs/experiments/ckm_straight_flow_prior.yaml \
    --data-root "$CKMSRF_DATA_ROOT"
```

Evaluate a checkpoint at the paper SNR grid (download the weights first, see
the top-level README):

```bash
python -m ckmsrf.evaluate \
    --config configs/experiments/ckm_straight_flow_prior.yaml \
    --checkpoint <path/to/checkpoint.pt> \
    --data-root "$CKMSRF_DATA_ROOT" --split test \
    --snr-db -10 -5 0 5 10 15 20 25 30 \
    --sampler flow-admm --pnp-iterations 40
```

The high-level launcher wraps the same commands and writes into `results/`:

```bash
python run_paper.py kap evaluate --data-root "$CKMSRF_DATA_ROOT"
```

Cross-scene (leave-one-site-out) statistics must be computed from the
corresponding source scenes and stored under the file name used by the fold
configuration, e.g. for `holdout_01` (train on scenes 02/03):

```bash
python data_generation/python/prepare_kapflow_data.py \
    --data-root "$CKMSRF_DATA_ROOT" --scenes 2 3 \
    --normalization-file normalization_cross_scene_01.json
```

Notes:

- `data.backend: plain_h5` is used by the paper protocol launcher; `backend:
  auto` prefers HDF5 when both HDF5 and raw `.mat` shards are present.
- The dataset and the checkpoints are **not** part of the git repository; only
  the generator and the preparation tooling are.
- The loading details (tensor layout, observation model, split index JSON,
  normalization statistics, and the exact paper settings) are specified in
  [`../docs/DATA.md`](../docs/DATA.md).

## HDF5 versus MATLAB `.mat` shards

| Item | Plain HDF5 (`plain_h5`) | Raw `.mat` (`mat_shard_stream`) |
|---|---|---|
| Produced by | `export_*_to_hdf5.m` | `generate_ckm_srf_dataset.m` |
| Python dependency | `h5py` | `hdf5storage` (optional extra) |
| Access pattern | lazy slices per sample | whole shard loaded into memory |
| Recommended for | training and evaluation | quick inspection of raw output |

MATLAB's high-level HDF5 writers store arrays with reversed axis order; the
KAP-Flow loader and `python/ckmsrf_h5_dataset.py` restore the logical
`(N_r, N_t, N_f, N_s, N)` order. Exact raw dataset shapes are listed in
[`../docs/DATA.md`](../docs/DATA.md).

## Compatibility notes (read before exporting)

`export_*_to_hdf5.m` is intentionally kept verbatim from the generator. Two
small attribute conventions differ from what the KAP-Flow loader expects:

1. `h5writeatt(..., 'scene_id', uint16(...))` writes a **one-element array**,
   while the loader reads the attribute with `int(...)`. Unpatched files raise
   `TypeError` in `build_shard_index`.
2. The exporters do not write a `storage_order` attribute. The loader falls
   back to a shape heuristic that is correct for the exported layouts but
   ambiguous if the number of records equals `N_r` (`16`).

Both are fixed automatically by
`python/prepare_kapflow_data.py` (idempotent; use `--check-only` to inspect a
directory without modifying it). Alternatively, the loader can be made robust
to the array-valued attribute upstream. If you build HDF5 files yourself, keep
the exporter's convention and run the preparation script once on the data root.

## Data-boundary rules

- `truth/*` fields are supervision labels and analysis variables only.
- Online model conditions must come from `state/*` and `context/*`.
- `context/*` is interpolated from an **independent historical** CKM; it does
  not contain instantaneous path gains of the current sample.
- `aux/*` (BIM/beamforming labels) is auxiliary and must not be used as an
  oracle model input.
- All snapshots of one trajectory belong to the same spatial split.

## 中文快速开始

1. **生成（MATLAB，无需 5G Toolbox）**

   ```matlab
   cd('<repo>/data_generation')
   run_demo                                  % 小规模端到端演示
   run('tests/run_all_tests.m')              % 物理与 schema 测试

   addpath('configs');
   cfg = research_config('<datasets>/ckm_srf_64x16');
   manifest = generate_ckm_srf_dataset(cfg); % 论文规模配置，先用小样本数验证磁盘/内存
   ```

2. **导出为 plain HDF5**

   ```matlab
   export_all_to_hdf5('<datasets>/ckm_srf_64x16');   % 输出到 <datasets>/ckm_srf_64x16_h5
   ```

3. **喂给 KAP-Flow**

   ```bash
   export CKMSRF_DATA_ROOT="<datasets>/ckm_srf_64x16_h5"
   python data_generation/python/prepare_kapflow_data.py --data-root "$CKMSRF_DATA_ROOT"
   python -m ckmsrf.train --config configs/experiments/ckm_straight_flow_prior.yaml \
       --data-root "$CKMSRF_DATA_ROOT"
   ```

   约定：数据根目录下直接放 `scene_XX_ckm.h5` 与
   `scene_XX_{train,val,test}_shard_XXX.h5`；`indices/{split}.json` 与
   `normalization.json` 可预先生成，也可由训练/评估 loader 自动构建。
   数据与权重均不入 git；公开仓库不包含数据集本身。字段、观测模型、
   划分与归一化细节见 [`../docs/DATA.md`](../docs/DATA.md)。

## See also

- Data contract and paper data setting: [`../docs/DATA.md`](../docs/DATA.md)
- Generator field schema: [`docs/SCHEMA.md`](docs/SCHEMA.md)
- Repository overview, installation, training and evaluation: top-level
  [`README.md`](../README.md)
