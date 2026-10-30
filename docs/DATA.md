# Data contract and paper data setting

This document specifies the dataset used by KAP-Flow (*KAP-Flow: Bridging Pilot
Observations and Propagation Knowledge for MIMO Channel Estimation*,
manuscript under review): the tensor layout, the HDF5 file/split/normalization
contract expected by the training and evaluation code in `src/ckmsrf/data/`,
and the data-generation settings reported in Sec. IV of the manuscript.

**Nothing in this section is distributed with the repository.** Datasets and
checkpoints are excluded from git; generate the data with
[`data_generation/`](../data_generation/README.md) or point
`CKMSRF_DATA_ROOT` at a compatible HDF5 root. The KAP-Flow manuscript is under
review.

Notation: `N_r` receive antennas, `N_t` transmit antennas, `N_f` subcarriers,
`N_s` measurement times (snapshots) per observation interval, `N_p` pilot
symbols per subcarrier, and `N` independent records ("samples") in a shard.

## 1. At a glance

| Item | Value for the paper data | Source |
|---|---|---|
| Channel tensor | `H(rx, tx, subcarrier, snapshot, sample)` = `(16, 64, 48, 8, N)` | `data_generation/docs/SCHEMA.md`; Sec. IV |
| Observation tensor | `Y(rx, pilot, subcarrier, snapshot, sample)` = `(16, 19, 48, 8, N)` | Sec. IV; generator schema |
| Sites / scenes | 3 independent sites, `200 x 200 m` each | manuscript.tex Sec. IV |
| CKM grid | `21 x 21` nodes, 10 m spacing | manuscript.tex Sec. IV |
| Spatial split | `40 x 40 m` blocks, 15/5/5 train/val/test | manuscript.tex Sec. IV |
| Records per site | 10000 / 1000 / 2000 train/val/test | manuscript.tex Sec. IV |
| Antennas | `8x8` Tx and `4x4` Rx half-wavelength UPAs -> `(N_t, N_r) = (64, 16)` | manuscript.tex Sec. IV |
| Pilots | `N_p = 19`; 16 DFT pilot matrices of size `64 x 19` | manuscript.tex Sec. IV |
| Waveform | `f_c = 28 GHz`, `Delta f = 15 kHz`, `N_f = 48`, `N_s = 8`, `Delta t = 1 ms` | manuscript.tex Sec. IV |
| Mobility | UE speed 20--60 m/s | manuscript.tex Sec. IV |
| Test slices | zero-based `(k, n) = (23, 3)` | manuscript.tex Sec. IV |
| SNR grid | `xi in {-10, -5, ..., 30} dB`, 9 points | manuscript.tex Sec. IV |

## 2. Unified channel tensor

The generator writes one unified tensor per shard **file**:

```
H(rx, tx, subcarrier, snapshot, sample)
  = H[:, :, k, t, n]  with shapes (N_r, N_t, N_f, N_s, N),
```

| Axis | Range | Meaning |
|---|---|---|
| `rx` | `0 .. N_r-1` | receive antenna (UPA element) |
| `tx` | `0 .. N_t-1` | transmit antenna (UPA element) |
| `subcarrier` | `0 .. N_f-1` | subcarrier index `k`; baseband offset `f_k = (k - (N_f-1)/2) Delta f` |
| `snapshot` | `0 .. N_s-1` | measurement time within one observation interval, `t_n = n Delta t` |
| `sample` | `0 .. N-1` | independent record: one UE trajectory realization |

- For the paper configuration the logical shape is `(16, 64, 48, 8, N)`.
- `H` is **complex single precision** in MATLAB and `complex64` after loading.
- `H` contains the physical channel including path loss; the generator does not
  normalize it. KAP-Flow normalizes by the training-set `h_rms` (Sec. 6).
- `channel_scale_lin(k, t, n) = ||H[:, :, k, t, n]||_F` is stored per record as
  an analysis/label field (not used as a model input).
- The CKM context is queried at each snapshot, so every record carries
  `N_s` condition vectors in addition to the `N_s` channel slices.
- The observation tensor is `Y[:, :, k, t, n]` with shape `(N_r, N_p, N_f, N_s, N)`,
  i.e. `(16, 19, 48, 8, N)` for the paper configuration.

### Observation model and SNR definition

At subcarrier `k` and snapshot `n` the receiver observes the finite-pilot
measurement

```
Y_{k,n} = H_{k,n} P + N_{k,n},      N_{k,n} ~ CN(0, nu_{k,n}),
```

with pilot matrix `P in C^{N_t x N_p}`, `P^H P = I_{N_p}`. The noise variance
used throughout the paper is

```
nu = || H P ||_F^2 / (N_r N_p 10^{xi/10}),
```

i.e. the received pilot energy per complex coefficient divided by the linear
SNR. The generator implements exactly this definition in
`src/ckmsrf_add_awgn.m`: `signalPower = mean(|H P|^2)` and
`noiseVar = signalPower / 10^(xi/10)`; since
`mean(|H P|^2) = ||H P||_F^2 / (N_r N_p)`, the two formulas agree.

Pilot bank (`src/ckmsrf_generate_pilot_bank.m`):

- 16 semi-unitary DFT matrices per scene (`cfg.pilot.numPilotMatrices = 16`,
  Sec. IV: "sixteen `64 x 19` DFT pilot matrices");
- each `P` is `N_p = 19` columns of the normalized 64-point DFT,
  `P = F[:, idx]` with `idx = randperm(64, 19)` and `F^H F = I`;
- the selected 1-based column indices are stored as
  `pilot/selected_dft_column`;
- for training/evaluation the paper configuration uses density
  `0.30 -> N_p = round(0.30 * 64) = 19`.

Per-record observation metadata stored in each shard: `observation/pilot_id`
(1-based index into the 16 pilot matrices), `observation/snr_db`
(from `-10:5:30` dB), `observation/noise_var`, `observation/cfo_hz`
(0 because `cfg.cfo.enable = false`), and `sample_seed`.

> The stored `snr_db`/`noise_var` are the generator's per-record draws. The
> evaluation protocol in Sec. IV sweeps the SNR grid independently; at each
> sweep point KAP-Flow synthesises `Y` from the clean `H`, the stored pilot
> matrix, and the target `xi` (see `src/ckmsrf/physics/observation.py`), and
> uses the resulting noise variance. The stored values are therefore metadata,
> not the evaluation noise levels.

## 3. Sites, grids, splits and records

All values below follow `configs/research_config.m` merged over
`ckmsrf_default_config.m`, and match manuscript.tex Sec. IV.

| Quantity | Value | Configuration |
|---|---|---|
| Sites/scenes | 3 (`scene_01`, `scene_02`, `scene_03`) | `cfg.numScenes = 3` |
| Service area | `200 x 200 m` | defaults `cfg.areaX = [20 220]`, `cfg.areaY = [-100 100]` |
| CKM grid | 10 m -> `21 x 21 = 441` nodes | `cfg.ckm.gridResolutionM = 10` |
| Spatial blocks | `40 x 40 m` -> `5 x 5 = 25` blocks | `cfg.split.blockSizeM = 40` |
| Split fractions | 0.60 / 0.20 / 0.20 -> `15/5/5` blocks | `cfg.split.{train,val,test}Fraction` |
| Records per scene | 10000 / 1000 / 2000 | `cfg.dataset.samplesPerScene.{train,val,test}` |
| Shard size | 250 records | `cfg.dataset.shardSize = 250` |
| Snapshots per record | 8 | `cfg.numSnapshots` |
| Snapshot period | 1 ms | `cfg.snapshotPeriodSec` |

Consequences:

- One site has 40 train shards, 4 validation shards and 8 test shards.
- Three sites have 30000 train / 3000 validation / 6000 test records. The
  6000-record test set is the one used for every SNR in the in-domain analyses
  (Sec. IV).
- Trajectories are sampled so that **all snapshots of a trajectory stay inside
  one split**; test/validation geometry is disjoint from training geometry.
- File naming: `scene_<SS>_<split>_shard_<KKK>.mat` (generator) and
  `scene_<SS>_<split>_shard_<KKK>.h5` (plain HDF5 export), with
  `split in {train, val, test}`, e.g. `scene_02_val_shard_003.h5`.
  The per-scene CKM/pilot package is `scene_<SS>_ckm.h5`.

## 4. HDF5 contract read by KAP-Flow

KAP-Flow's loader (`src/ckmsrf/data/shard_index.py`,
`src/ckmsrf/data/lazy_dataset.py`) supports two backends:

- `plain_h5` (recommended, used for the paper): the explicit HDF5 exports
  produced by `export_scene_to_hdf5.m` / `export_shard_to_hdf5.m`;
- `mat_shard_stream`: the raw MATLAB `-v7.3` shards, read eagerly with
  `hdf5storage` (optional dependency).

`backend: auto` prefers plain HDF5 when both are present.

### 4.1 MATLAB axis order (important)

MATLAB's `h5create`/`h5write` store a MATLAB array of size `[d1 d2 ...]` so
that raw `h5py` sees the **reversed** axis order. For a paper-scale shard the
raw and logical shapes are:

| Data | Raw `h5py` shape | Logical (MATLAB/schema) shape | dtype |
|---|---|---|---|
| `H_real` / `H_imag` | `(N, 8, 48, 64, 16)` | `(16, 64, 48, 8, N)` | float32 |
| `Y_real` / `Y_imag` (optional) | `(N, 8, 48, 19, 16)` | `(16, 19, 48, 8, N)` | float32 |
| `channel_scale_lin` | `(N, 8, 48)` | `(48, 8, N)` | float32 |

The loader reads this convention from the `storage_order` root attribute
(`matlab_reversed`) when present, or infers it from the `H`/`pilot_id` shapes,
then slices `/H_real[n, t, k, :, :]` and transposes to obtain
`H[:, :, k, t, n]` of shape `(N_r, N_t)`. The standalone reader
`data_generation/python/ckmsrf_h5_dataset.py` reverses all axes the same way.

### 4.2 Shard datasets

Paper-scale shapes are logical (MATLAB/schema) shapes; the raw HDF5 order is
reversed as in Sec. 4.1. "Loader" marks what `CKMSRFDataset` reads.

| HDF5 path | Logical shape | Loader | Meaning |
|---|---|---|---|
| `/H_real`, `/H_imag` | `(16, 64, 48, 8, N)` | yes | clean channel |
| `/Y_real`, `/Y_imag` | `(16, 19, 48, 8, N)` | no | stored observation (`storeObservation = true` only) |
| `/channel_scale_lin` | `(48, 8, N)` | yes | `||H[:,:,k,t]||_F` |
| `/state/position_xyz` | `(3, 8, N)` | yes | true UE position [m] |
| `/state/velocity_xyz` | `(3, 8, N)` | yes | UE velocity [m/s] |
| `/state/ue_orientation_deg` | `(1, 8, N)` | yes | UE yaw [deg] |
| `/state/time_sec` | `(8, 1)` | yes | snapshot times `t_n` [s] |
| `/state/block_id` | `(8, N)` | yes | spatial split block id |
| `/context/observed_position_xyz` | `(3, 8, N)` | yes | position estimate available online [m] |
| `/context/grid_indices` | `(4, 8, N)` | yes | 4 bilinearly interpolated CKM nodes (1-based linear indices) |
| `/context/grid_weights` | `(4, 8, N)` | yes | bilinear interpolation weights |
| `/context/scalar_feature` | `(6, 8, N)` | yes | CKM scalar features (Sec. 4.4) |
| `/context/angular_delay_power` | `(3072, 8, N)` | yes | CKM angular-delay power, flattened `16x16x12` |
| `/context/tx_beam_probability` | `(64, 8, N)` | yes | CKM transmit beam probabilities |
| `/context/rx_beam_probability` | `(16, 8, N)` | yes | CKM receive beam probabilities |
| `/observation/pilot_id` | `(1, N)` | yes | 1-based pilot-matrix index |
| `/observation/snr_db` | `(1, N)` | yes | generator SNR draw [dB] |
| `/observation/cfo_hz` | `(1, N)` | yes | CFO (0 for the paper data) |
| `/observation/noise_var` | `(1, N)` | yes | generator noise variance |
| `/sample_seed` | `(1, N)` | yes | per-record seed |
| `/truth/path_id`, `path_mask`, `path_power_lin`, `delay_sec`, `aod_az_rad`, `aod_el_rad`, `aoa_az_rad`, `aoa_el_rad`, `doppler_hz` | `(L, 8, N)` | no | path-level labels |
| `/truth/path_gain_real`, `/truth/path_gain_imag` | `(L, 8, N)` | no | complex path gains |

`L = cfg.numPaths = 6`: the six strongest physical paths selected per
trajectory from the candidate set (LoS plus scatterer paths). The
`truth/*` and `aux/*` fields are supervision labels only; `aux/*` (BIM/beam
labels) is not exported to HDF5 and must not be used as an oracle input.

Root attributes written by the exporter:

| Attribute | Example | Notes |
|---|---|---|
| `schema_version` | `"1.0"` | string |
| `dimension_order_H` | `rx,tx,subcarrier,snapshot,sample` | documentation only |
| `dimension_order_Y` | `rx,pilot,subcarrier,snapshot,sample` | documentation only |
| `split_name` | `train` | documentation only |
| `scene_id` | `1` | **must be scalar** for the loader (see Sec. 11) |
| `storage_order` | `matlab_reversed` | added by the preparation script if absent |

### 4.3 Scene (CKM + pilot) files

`scene_<SS>_ckm.h5` holds the pilot bank, the DFT codebooks, the offline CKM and
the spatial split plan. Paper-scale logical shapes (raw order reversed):

| HDF5 path | Logical shape | Meaning |
|---|---|---|
| `/pilot/P_real`, `/pilot/P_imag` | `(64, 19, 16)` | 16 pilot matrices `P` |
| `/pilot/selected_dft_column` | `(19, 16)` | selected 1-based DFT columns |
| `/codebook/Ftx_real`, `/codebook/Ftx_imag` | `(64, 64)` | transmit DFT codebook |
| `/codebook/Wrx_real`, `/codebook/Wrx_imag` | `(16, 16)` | receive DFT codebook |
| `/ckm/grid_x` | `(1, 21)` | CKM x coordinates [m] |
| `/ckm/grid_y` | `(1, 21)` | CKM y coordinates [m] |
| `/ckm/grid_xy` | `(441, 2)` | CKM node coordinates [m] |
| `/ckm/scalar_feature` | `(6, 441)` | per-node scalar features |
| `/ckm/angular_delay_power` | `(3072, 441)` | per-node angular-delay power (normalized) |
| `/ckm/tx_beam_probability` | `(64, 441)` | per-node tx beam probabilities |
| `/ckm/rx_beam_probability` | `(16, 441)` | per-node rx beam probabilities |
| `/ckm/adp_shape` | `(1, 3)` | `[16, 16, 12]` = `[n_AoD, n_AoA, n_delay]` |
| `/split/assignment` | `(1, 25)` | block id -> split code `1/2/3` |

The KAP-Flow loader reads only `/pilot/P` from the scene file: at sample time it
maps the record's 1-based `pilot_id` to a 0-based index and returns
`P in C^{64 x 19}`. With `include_observation: true` the dataset then synthesises
`Y = H P + N` online; expected `/pilot/P_real`, `/pilot/P_imag` (or a complex
`/pilot/P`) layout.

### 4.4 CKM feature semantics

- `scalar_feature` order (fixed, stored as `scalarFeatureNames` in the MATLAB
  scene package):
  `1` LoS probability, `2` path-loss mean [dB], `3` path-loss std [dB],
  `4` mean delay [s], `5` RMS delay spread [s], `6` historical sample count.
- `angular_delay_power` is normalized over the 16x16x12 bins and flattened with
  **AoD as the fastest axis**, then AoA, then delay:
  `linear = aod_bin + (aoa_bin-1)*16 + (delay_bin-1)*16*16` (1-based MATLAB
  indices). The dataset converts this vector to C order before reshaping to
  `[16, 16, 12]`, so `H.condition.adp` (and `ckm_adp`) is a 3072-vector whose
  C-order reshape is `[AoD, AoA, delay]`.
- `tx_beam_probability` / `rx_beam_probability` are probabilities over the DFT
  beam codebooks (`topKBeams = 8` labels are stored in the raw `.mat` shard but
  are not required by KAP-Flow).
- `grid_indices` / `grid_weights` identify the four CKM nodes and bilinear
  weights used for each snapshot. `observed_position_xyz` is the online
  position estimate; with `cfg.localization.errorStdM = 0` (paper data) it
  equals the true position.

## 5. Split index JSON (`indices/{split}.json`)

The training/evaluation loader uses a portable JSON index (`DatasetIndex`):

```json
{
  "schema_version": 1,
  "split": "train",
  "records": [
    {
      "path": "scene_01_train_shard_001.h5",
      "scene_id": 1,
      "split": "train",
      "shard_id": 1,
      "num_trajectories": 250,
      "num_subcarriers": 48,
      "num_snapshots": 8,
      "num_rx": 16,
      "num_tx": 64,
      "backend": "plain_h5",
      "storage_order": "matlab_reversed"
    }
  ],
  "scene_files": {"1": "scene_01_ckm.h5"}
}
```

- Record `path` and `scene_files` values are stored **relative to the data
  root**. The JSON itself does not contain the root, so a direct load must pass
  it explicitly: `DatasetIndex.load(".../indices/train.json", root=<data root>)`.
  The training/evaluation factory does this automatically.
- Records are sorted by `(scene_id, shard_id, path)`. The global dataset index
  is the concatenation of the records in that order; `trajectory_global` /
  `trajectory_id` enumerate records over the whole split, while
  `trajectory_in_shard` is the local record index.
- `scene_files` maps scene ids to CKM/pilot HDF5 files; it is required whenever
  observations are synthesised (`include_observation: true`).
- The data configuration uses `data.index_file: indices/{split}.json`. On first
  use `load_or_build_index` scans the HDF5 metadata (no channel data is read),
  builds the index, and caches it under a scene/backend-tagged name such as
  `indices/train_plain-h5_s01-02-03.json` (backend `plain_h5`, scenes 01--03).
  A pre-built `indices/{split}.json` documents the contract and is portable,
  but training does not require it.
- `data_generation/python/prepare_kapflow_data.py` writes the untagged
  `indices/{split}.json` files with exactly the schema above.

## 6. Normalization statistics (`normalization.json`)

KAP-Flow normalizes channels and conditions with training-set statistics; the
file is stored in the data root (config `data.normalization.stats_file`,
default `normalization.json`). Annotated schema (`[n]` denotes a length-`n`
vector; `h_rms` is shown as a placeholder value):

```json
{
  "schema_version": 1,
  "split": "train",
  "h_rms": 1.0,
  "feature_mean": {"scalar": [6], "adp": [3072], "position": [3],
                    "velocity": [3], "observed_position": [3]},
  "feature_std":  {"scalar": [6], "adp": [3072], "position": [3],
                    "velocity": [3], "observed_position": [3]},
  "num_trajectories": 30000,
  "num_channel_coefficients": 11796480000,
  "epsilon": 1e-06
}
```

Definitions (`src/ckmsrf/data/normalization.py`):

- `h_rms = sqrt(mean |H|^2)` over **all** complex coefficients of the selected
  training split (all shards, all `N_r x N_t x N_f x N_s x N` entries).
  Channels are normalized as `H / h_rms`; noise variances are divided by
  `h_rms^2`.
- `feature_mean/std` are per-dimension moments over all `(record, snapshot)`
  pairs of the training split for: `scalar` (6), `adp` (3072 = 16x16x12),
  `position` (3), `velocity` (3), `observed_position` (3). Standard deviations
  are floored at `epsilon`.
- `orientation` is intentionally **not** z-scored: the model receives
  `sin(psi), cos(psi)` from the physical yaw in degrees, and the loader keeps
  the raw angle for that computation.
- The statistics must be computed from the **train** split only;
  `compute_normalization` refuses `val`/`test` indices by default to prevent
  evaluation leakage.
- The per-dimension moments are streamed over trajectory chunks
  (`data.normalization.chunk_trajectories`, default 8) so that the full
  training set never has to fit in RAM.
- Cross-scene (leave-one-site-out) folds use **source-site** statistics stored
  under the file names referenced by the fold configs:
  `normalization_cross_scene_01.json` (train scenes 02/03, test scene 01),
  `normalization_cross_scene_02.json`, `normalization_cross_scene_03.json`.
- If the file is missing, training computes it automatically and saves it. The
  preparation script in `data_generation/python/` does the same explicitly.

### What one condition vector contains

After normalization the loader exposes (widths from
`configs/experiments/ckm_straight_flow_prior.yaml`):

| Field | Width | Construction |
|---|---|---|
| `scalar` | 6 | normalized CKM scalar features |
| `adp` | 3072 | normalized angular-delay power, `[16,16,12]` model reshape |
| `tx_beam` | 64 | normalized tx beam probabilities |
| `rx_beam` | 16 | normalized rx beam probabilities |
| `position` / `velocity` | 3 / 3 | normalized state components |
| `orientation` | 1 | UE yaw in **raw degrees** (not z-scored; used for `sin/cos`) |
| `observed_position` | 3 | normalized online position estimate |
| `state` | 8 | `observed_position(3)`, `velocity(3)`, `sin(psi)` and `cos(psi)` (2) |
| `tf` | 2 | `[2k/(N_f-1)-1, 2t/(N_s-1)-1]` |
| `time_sec` | 1 | physical measurement time [s] |
| `grid_indices` / `grid_weights` | 4 / 4 | CKM interpolation support |

## 7. How KAP-Flow samples the data

- **Training** (`sampling: random_tf`, the paper protocol): one deterministic
  pseudo-random `(subcarrier, snapshot)` pair per record per epoch, seeded from
  `(seed, epoch, scene_id, shard_id, record)`. This matches Sec. IV: "Training
  draws one subcarrier--time pair per record per epoch."
- **Validation / test** (`sampling: fixed`): the default fixed pair is the
  lower-centre slice
  `((N_f-1)//2, (N_s-1)//2) = (23, 3)` (zero-based) for `N_f = 48`,
  `N_s = 8` -- exactly the test slice in Sec. IV. Explicit `fixed_pairs` can
  override it.
- **Prior training** uses clean channels for the prior objective; the
  experiment YAMLs used for the paper priors set
  `include_observation: false`. Setting `include_observation: true` only adds
  the online observation `(P, Y)` to the returned batch (used by the
  solver-side evaluation code), it does not change the stored data.
- **Evaluation**: for each test record the loader returns the clean
  `H` (normalized), the record's pilot `P` (from the scene file, chosen by the
  stored 1-based `pilot_id`), the CKM condition and, when
  `include_observation: true`, an observation `Y`. The evaluation code then
  regenerates `Y` from `H`, `P` and the target SNR with the paper's variance
  formula, and runs the solver. The 16 pilot matrices and the per-record pilot
  assignment implement Sec. IV's "sixteen `64 x 19` DFT pilot matrices".
- The full paper protocol (9 SNRs, all 6000 test records, three seeds, linear
  NMSE averaging before dB conversion) is orchestrated by the top-level
  commands; the data side only has to provide `H`, `P` and the CKM condition.

## 8. Paper data setting

All numbers in this section are taken from **manuscript.tex Sec. IV**
("Parameters and Evaluation Protocol") and are reproduced here as the single
source of truth for the data generation. Do not tune them without updating the
manuscript.

| Setting | Value | Source |
|---|---|---|
| Sites | 3 independent sites | manuscript.tex Sec. IV |
| Service area | `200 x 200 m` | manuscript.tex Sec. IV |
| CKM grid | `21 x 21` nodes, 10 m spacing | manuscript.tex Sec. IV |
| Spatial split | `40 x 40 m` blocks, `15/5/5` train/validation/test | manuscript.tex Sec. IV |
| Records | `10000/1000/2000` train/validation/test per site | manuscript.tex Sec. IV |
| Tx/Rx arrays | half-wavelength `8 x 8` Tx, `4 x 4` Rx UPAs | manuscript.tex Sec. IV |
| Antennas / pilots | `(N_t, N_r, N_p) = (64, 16, 19)` | manuscript.tex Sec. IV |
| Carrier / spacing | `f_c = 28 GHz`, `Delta f = 15 kHz` | manuscript.tex Sec. IV |
| Subcarriers / snapshots | `N_f = 48`, `N_s = 8` | manuscript.tex Sec. IV |
| Snapshot period | `Delta t = 1 ms` | manuscript.tex Sec. IV |
| UE speed | 20--60 m/s | manuscript.tex Sec. IV |
| CKM node history | 32 historical channels per node | manuscript.tex Sec. IV |
| Angular-delay bins | `16 x 16 x 12` | manuscript.tex Sec. IV |
| Beam features | 64 tx / 16 rx | manuscript.tex Sec. IV |
| Test slice | zero-based `(k, n) = (23, 3)` | manuscript.tex Sec. IV |
| Test pilots | 16 DFT pilot matrices of size `64 x 19` | manuscript.tex Sec. IV |
| SNR grid | `xi in {-10, -5, ..., 30} dB` (9 values) | manuscript.tex Sec. IV |
| Noise variance | `nu = ||HP||_F^2 / (N_r N_p 10^{xi/10})` | manuscript.tex Sec. IV |
| Test set size | 6000 records per SNR (3 sites x 2000) | manuscript.tex Sec. IV |
| Seeds | three random seeds, shared by all experiments | manuscript.tex Sec. IV |

Related protocol facts from the same section (not data-generation knobs):
the 4.01-M-parameter network, AdamW with learning rate `2e-4`, EMA `0.999`,
`(K, tau, s_min, s_max, gamma) = (40, 0.125, 0, 0.95, 0.25)` for KAP-Flow and
`(K, eta, alpha) = (40, 1.5, 0.25)` for PnP-Flow, `M = 1` unless varied
(manuscript.tex Sec. IV).

## 9. Generator configuration mapping

`data_generation/configs/research_config.m` encodes Sec. IV. It is merged over
`ckmsrf_default_config.m` by `generate_ckm_srf_dataset.m`.

| Paper setting | MATLAB configuration field | Value |
|---|---|---|
| 3 sites | `cfg.numScenes` | `3` |
| 200 x 200 m | `cfg.areaX`, `cfg.areaY` (defaults) | `[20 220]`, `[-100 100]` |
| 10 m CKM grid | `cfg.ckm.gridResolutionM` | `10` |
| 32 history samples/node | `cfg.ckm.historySamplesPerCell` | `32` |
| 16x16x12 bins | `cfg.ckm.numAodBins/numAoaBins/numDelayBins` | `16/16/12` |
| 64/16 beam features | `cfg.numTxAntennas`, `cfg.numRxAntennas` | `64`, `16` |
| 8x8 / 4x4 UPA | `cfg.txArraySize`, `cfg.rxArraySize` | `[8 8]`, `[4 4]` |
| 28 GHz, 15 kHz | `cfg.carrierFrequencyHz`, `cfg.subcarrierSpacingHz` | `28e9`, `15e3` |
| 48 / 8 / 1 ms | `cfg.numSubcarriers`, `cfg.numSnapshots`, `cfg.snapshotPeriodSec` | `48`, `8`, `1e-3` |
| 20--60 m/s | `cfg.speedRangeMps` (default) | `[20 60]` |
| 40 m blocks, 15/5/5 | `cfg.split.blockSizeM`, `cfg.split.*Fraction` | `40`, `0.60/0.20/0.20` |
| 10000/1000/2000 records | `cfg.dataset.samplesPerScene.*` | `10000/1000/2000` |
| 250 records/shard | `cfg.dataset.shardSize` | `250` |
| 16 pilot matrices | `cfg.pilot.numPilotMatrices` | `16` |
| `N_p = 19` | `cfg.pilot.density` (rounds to 19) | `0.30` |
| SNR grid `-10:5:30` | `cfg.pilot.snrDbChoices` | `-10:5:30` |
| large-scale storage off | `cfg.output.storeObservation` | `false` |
| clean channel stored | `cfg.output.storeTrueChannel` | `true` |
| no CFO/ICI | `cfg.cfo.enable` | `false` |
| exact online position | `cfg.localization.errorStdM` | `0` |

Notes:

- `research_config.m` is described in the generator as a *suggested starting
  point for formal experiments*; its values match Sec. IV. The exact production
  run configuration and RNG state used for the paper corpus are not part of the
  release, so bit-exact regeneration of the published tables is **not**
  claimed.
- The paper-scale generation stores only the clean channel plus observation
  metadata; `Y` is synthesized online by KAP-Flow. This keeps the dataset at
  roughly `3 MiB` per record for `H` (`29.3 GiB` per site's training split
  before HDF5 overhead).

## 10. End-to-end pipeline

Commands below assume the repository Python package is importable
(`python -m pip install -e .` from the repository root, or `PYTHONPATH=src`).

```bash
# 1) MATLAB: generate paper-scale .mat shards and export plain HDF5
#    run_demo / tests for a smoke test first (see data_generation/README.md)
#    export_all_to_hdf5('<datasets>/ckm_srf_64x16')

# 2) Python: validate the HDF5 contract, patch MATLAB root attributes,
#    build indices/{split}.json and train-only normalization.json
export CKMSRF_DATA_ROOT="<datasets>/ckm_srf_64x16_h5"
python data_generation/python/prepare_kapflow_data.py --data-root "$CKMSRF_DATA_ROOT"

# 3) train the conditional prior
python -m ckmsrf.train \
    --config configs/experiments/ckm_straight_flow_prior.yaml \
    --data-root "$CKMSRF_DATA_ROOT"

# 4) evaluate a checkpoint on the paper SNR grid
python -m ckmsrf.evaluate \
    --config configs/experiments/ckm_straight_flow_prior.yaml \
    --checkpoint <checkpoint.pt> --data-root "$CKMSRF_DATA_ROOT" \
    --split test --snr-db -10 -5 0 5 10 15 20 25 30 \
    --sampler flow-admm --pnp-iterations 40
```

`CKMSRF_DATA_ROOT` is also read when `--data-root` is omitted. Note the
different defaults: the high-level launcher (`run.py` / `run_paper.py`) defaults
to `<repo>/data/ckm_srf_64x16_h5`, while the bare `python -m ckmsrf.train` /
`python -m ckmsrf.evaluate` entry points use the config default
`paths.data_root: data` (i.e. `<repo>/data`). For a generated dataset always
set `CKMSRF_DATA_ROOT` or pass `--data-root`. On Windows PowerShell:

```powershell
$env:CKMSRF_DATA_ROOT = "<datasets>\ckm_srf_64x16_h5"
```

## 11. Gaps and compatibility notes (read this before reporting bugs)

The generator, the HDF5 exporter and the KAP-Flow loader are three
independently versioned pieces. What is true in this release:

1. **The HDF5 export exists and matches the loader semantically.**
   `export_scene_to_hdf5.m` and `export_shard_to_hdf5.m` produce exactly the
   datasets listed in Sec. 4. They were exercised end-to-end for this release
   on the generator's demo export: `Y` equals `H P` plus exactly the
   generator's AWGN level (relative residual `~0.10` at 20 dB, `~0.56` at
   5 dB), and the KAP-Flow `CKMSRFDataset` loads `H (16,64)`, `P (64,19)`,
   condition elements (scalar 6, ADP, tx beam 64, rx beam 16, state 8, tf 2)
   and the 4 grid indices/weights from the exported files. The demo uses
   `(N_f,N_s)=(4,3)` and an `8x8x6` CKM (ADP width 384); the paper-scale
   `48/8` and `16x16x12` (ADP width 3072) follow the same code path and layout.
2. **There is no auto-detectable "one-shot" HDF5 export script in the
   generator proper** -- only the per-file functions. This release adds
   `data_generation/export_all_to_hdf5.m` as a thin batch wrapper; it does not
   change the exporter or its output.
3. **Root-attribute mismatch (known, fixed by the preparation script).**
   `h5writeatt(..., 'scene_id', uint16(id))` writes a one-element array, while
   `src/ckmsrf/data/shard_index.py` calls `int(attribute)`, which raises
   `TypeError` on current NumPy. The exporter also omits `storage_order`, so
   the loader falls back to a shape heuristic that is ambiguous when
   `N == N_r = 16`.
   `data_generation/python/prepare_kapflow_data.py` rewrites `scene_id` as a
   scalar and writes `storage_order = matlab_reversed`; the `.m` exporters are
   kept verbatim from the generator repository.
4. **Index and normalization preparation are not part of the MATLAB
   generator.** They are KAP-Flow-side operations. The source checkout provides
   `tools/build_index.py` and `tools/compute_normalization.py` as thin wrappers
   around `ckmsrf.data`; those wrappers are not copied into this release, so
   the bundled `data_generation/python/prepare_kapflow_data.py` performs the
   same two steps and keeps the data-generation directory self-sufficient.
5. **The raw `.mat` path is still supported** (`backend: mat_shard_stream`,
   optional `hdf5storage` dependency), but it loads a whole shard per access.
   Use it for inspection only at paper scale.
6. **The dataset itself is not distributed, and the exact paper corpus is not
   part of the release.** The released generator + configuration reproduce the
   data *model* and the Sec. IV settings; bit-exact reproduction of the paper
   tables additionally requires the original dataset or the exact production
   configuration/RNG state, which are not published.
7. **The export directory must keep `storeTrueChannel = true`.** With
   `storeTrueChannel = false` the exported shards have no `/H`, and no backend
   can train on them.
8. **Do not confuse demo dimensions with the paper configuration.**
   `run_demo.m` deliberately uses 4 subcarriers, 3 snapshots and small shards
   (e.g. `H` raw shape `(6, 3, 4, 64, 16)`); the paper data uses
   `(N, 8, 48, 64, 16)`.

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `TypeError` in `build_shard_index` | Unpatched exporter attributes; run `prepare_kapflow_data.py` once on the data root |
| `FileNotFoundError: No plain_h5 files for split 'val'` | Validation/test shards not exported; run `export_all_to_hdf5.m` and check the file names `scene_<SS>_<split>_shard_<KKK>.h5` |
| `FileNotFoundError: Missing scene CKM files for scenes [...]` | `scene_<SS>_ckm.h5` was not exported; the loader needs the pilot bank |
| ADP width warning, or `model.condition.adp_shape` mismatch | Data generated with non-paper CKM bins; paper data has `16x16x12 = 3072` |
| `ModuleNotFoundError: hdf5storage` | Only needed for `backend: mat_shard_stream`; install the `mat` extra or use `plain_h5` |
| `DatasetIndex` file not found / shards resolved against `indices/` | Pass `root=<data root>` when loading an index JSON directly; training/evaluation resolve it automatically |
| Very slow first epoch | The loader is building the tagged index cache or computing `normalization.json`; pre-run `prepare_kapflow_data.py` |
| Out of memory while loading | Keep `mat_shard_stream` off, lower `data.max_open_files` (default 2), and use the HDF5 backend |

## See also

- Generator quick start and HDF5 export: [`../data_generation/README.md`](../data_generation/README.md)
- Generator field schema: [`../data_generation/docs/SCHEMA.md`](../data_generation/docs/SCHEMA.md)
- Repository overview, installation, training and evaluation: top-level [`README.md`](../README.md)
