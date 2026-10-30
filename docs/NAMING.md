# Naming and migration map

The paper terminology, the code names, the archived experiment names and the
released artifact names are not identical. This document maps them.

The paper is **KAP-Flow: Bridging Pilot Observations and Propagation Knowledge
for MIMO Channel Estimation** (manuscript under review; no venue assigned at
release time). The code package is `src/ckmsrf` with the CLI layer
`src/journal_flow`. Every name below was checked against
`src/journal_flow/cli.py`, `configs/checkpoints.json`,
`main/configs/`, `baselines/configs/` and the submitted Table I
(`tables/main_nmse.tex`) in the paper source package.

## 1. One-line summary

| Paper | Code method key (`journal_flow.cli`) | `run_paper.py` key | `--sampler` string | Legacy / archived label | Table I row |
| --- | --- | --- | --- | --- | --- |
| KAP-Flow, conditioned | `flowadmm` | `kap` | `flow-admm` | FlowADMM, FA, **FA-C** | **KAP-C** |
| KAP-Flow, unconditioned | `flowadmm` | `kap` | `flow-admm` | FlowADMM, FA, **FA-U** | **KAP-U** |
| PnP-Flow, conditioned | `pnpflow` | `pnpflow` | `pnp-flow` | PnP-Flow, PF, PF-C | PF-C |
| PnP-Flow, unconditioned | `pnpflow` | `pnpflow` | `pnp-flow` | PnP-Flow, PF, PF-U | PF-U |
| Recursive Flow, conditioned | `rcflow` | `rcflow` | `srf-rc` | RC / RC-Flow / RC-C | RC-C |
| Recursive Flow, unconditioned | `rcflow` | `rcflow` | `srf-rc` | RC / RC-Flow / RC-U | RC-U |
| RC-C with SRF | `srf` | `srf` | `srf-rc` | SRF, ckmsrf_full, RC-S | RC-S |
| PnP-CM, conditioned | `pnp-cm` | `pnp-cm` | `pnp-cm` | CM, PnP-CM, CM-C | CM-C |
| PnP-CM, unconditioned | `pnp-cm` | `pnp-cm` | `pnp-cm` | CM, PnP-CM, CM-U | CM-U |
| LS | `ls` | `ls` | -- | LS | LS |
| LMMSE | `lmmse` | `lmmse` | -- | LMMSE, "Global LMMSE" | LMMSE |

Two naming facts are easy to get wrong:

1. **The paper method KAP-Flow is called `flowadmm` in the code.** The code
   pre-dates the paper name; the CLI's `METHODS` tuple and `configs/checkpoints.json`
   still say `flowadmm`. The paper table calls the same rows **KAP-C / KAP-U**,
   while the archived source CSVs and the timing manifest call them
   **FA-C / FA-U** (FA = FlowADMM).
2. **The paper's `RC` rows are implemented by the same GMFlow sampler as
   `RC-S`; they differ by checkpoint.** `rcflow` and `srf` both dispatch to
   the sampler string `srf-rc` (`src/journal_flow/cli.py::make_jobs`), which
   calls `ckmsrf.sampling.sample_ckmsrf`. `rcflow` loads a plain GMFlow prior
   (`rcflow_ckm` / `rcflow_no_ckm`), `srf` loads the SRF-trained prior
   (`srf_ckm`, experiment `ckmsrf_full`). The paper's "RC-C with SRF (RC-S)"
   row is therefore `--method srf --conditions ckm`.

## 2. Condition suffix: U/C vs `no_ckm`/`ckm`

| Paper | Code config wording | Checkpoint folder | Meaning |
| --- | --- | --- | --- |
| `-C` (conditioned) | `ckm`, `condition: {enabled: true, force_unconditional: false}` | `.../ckm/` | CKM descriptor + UE state + time--frequency coordinates are used |
| `-U` (unconditioned) | `no_ckm`, `condition: {enabled: true, force_unconditional: true}` | `.../no_ckm/` | the **same** backbone with every condition contribution suppressed; a separately trained checkpoint, not a zeroed input |

The code spelling is `no_ckm` (underscore, CLI choice and checkpoint folder
name); the paper text and Table I use `-U` / `-C`. There is no `uncond` /
`nocond` spelling anywhere in the release.

## 3. Paper method names and their components

| Paper acronym | Expansion (from the manuscript) | Code location |
| --- | --- | --- |
| KSC | knowledge-and-state conditioning | `src/ckmsrf/models/condition.py`, `src/ckmsrf/models/unet.py` |
| PPC | pilot--prior consensus | `src/ckmsrf/sampling_flow.py::sample_flow_admm` |
| FlowADMM | ADMM variant with a flow prior (the reference work cited by the paper) | legacy name kept for the code method `flowadmm` |
| SRF | pilot-aware surrogate refinement (our previous work) | `src/ckmsrf/sampling.py::sample_ckmsrf`; `srf` method |
| RC / RC-Flow | recursive flow | `src/ckmsrf/sampling.py::sample_ckmsrf` with a GMFlow checkpoint |
| PBG / FIG+ | posterior bridge guidance (SRF-specific; legacy names `FIG`, `fig_plus`; not used by KAP-Flow) | `src/ckmsrf/physics/guidance.py::pilot_guided_update`, `::fig_plus_projection` |
| PnP-CM | plug-and-play consistency model | `src/ckmsrf/sampling_cm.py::sample_ckmcm_pnp` |
| PnP-Flow | plug-and-play flow matching | `src/ckmsrf/sampling_flow.py::sample_pnp_flow` |
| PnP-DM | plug-and-play diffusion model (not used in the paper table) | `src/ckmsrf/sampling_pnpdm.py::sample_ckm_diffusion_pnpdm` |

## 4. Experiment/run names vs release configs

`configs/experiments/*.yaml` are the general research entry configs (the
`ckmsrf.train` / `ckmsrf.evaluate` flags can use them); `main/configs/` and
`baselines/configs/` are the **paper protocol templates** that the released
checkpoints were trained/validated with.

| Paper row / role | Experiment name (`experiment.name`) | Release config template | Registry id of released weights |
| --- | --- | --- | --- |
| KAP-C, PF-C prior | `ckm_straight_flow_prior` | `main/configs/standard/ckm.yaml` (also `configs/experiments/ckm_straight_flow_prior.yaml`) | `straight_ckm` |
| KAP-U, PF-U prior | `ckm_straight_flow_no_ckm` | `main/configs/standard/no_ckm.yaml` (also `configs/experiments/ckm_straight_flow_no_ckm.yaml`) | `straight_no_ckm` |
| transfer fold 1 (test scene 1) | `straight_flow_ckm_holdout_scene_01` / `straight_flow_no_ckm_holdout_scene_01` | `main/configs/threefold/holdout_01_{ckm,no_ckm}.yaml` | `straight_ckm_fold1`, `straight_no_ckm_fold1` |
| transfer fold 2 (test scene 2) | `straight_flow_ckm_holdout_scene_02` / `..._no_ckm_holdout_scene_02` | `main/configs/threefold/holdout_02_{ckm,no_ckm}.yaml` | `straight_ckm_fold2`, `straight_no_ckm_fold2` |
| transfer fold 3 (test scene 3) | `straight_flow_ckm_holdout_scene_03` / `..._no_ckm_holdout_scene_03` | `main/configs/threefold/holdout_03_{ckm,no_ckm}.yaml` | `straight_ckm_fold3`, `straight_no_ckm_fold3` |
| RC-U | `baseline_no_ckm` | `baselines/configs/rcflow_no_ckm.yaml` (also `configs/experiments/baseline_no_ckm.yaml`) | `rcflow_no_ckm` |
| RC-C | `ckmsrf_prior` | `baselines/configs/rcflow_ckm.yaml` (also `configs/experiments/ckmsrf_prior.yaml`) | `rcflow_ckm` |
| RC-S | `ckmsrf_full` | `baselines/configs/srf_ckm.yaml` (also `configs/experiments/ckmsrf_full.yaml`) | `srf_ckm` |
| CM-C | `ckmcm_prior` | `baselines/configs/pnp_cm_ckm.yaml` (also `configs/experiments/ckmcm_prior.yaml`) | `pnp_cm_ckm` |
| CM-U | `ckmcm_no_ckm` | `baselines/configs/pnp_cm_no_ckm.yaml` | `pnp_cm_no_ckm` |
| LMMSE (in-domain) | statistics fit on the source training split | produced by `python -m ckmsrf.fit_channel_stats`; `configs/experiments/ckmcm_pnp.yaml` is the PnP-CM research entry, **not** LMMSE | `lmmse` |
| LMMSE (threefold) | per-fold source-only statistics | -- | `lmmse_fold1/2/3` |

`configs/experiments/cross_scene_03.yaml` is a generalisation study entry
(train on scenes 01/02, test scene 03); it is not the paper's three-fold
protocol.

## 5. Released checkpoint registry (`configs/checkpoints.json`)

The registry has 17 entries: 13 network weights and 4 LMMSE statistics. Target
paths are relative to the repository root:

| id | kind | methods served | target |
| --- | --- | --- | --- |
| `straight_ckm` | weights | `flowadmm`, `pnpflow` | `checkpoints/main/straight/ckm/best_nmse.pt` |
| `straight_no_ckm` | weights | `flowadmm`, `pnpflow` | `checkpoints/main/straight/no_ckm/best_nmse.pt` |
| `straight_ckm_fold1..3` | weights | `flowadmm`, `pnpflow` | `checkpoints/main/threefold/holdout_0{1,2,3}/ckm/best_nmse.pt` |
| `straight_no_ckm_fold1..3` | weights | `flowadmm`, `pnpflow` | `checkpoints/main/threefold/holdout_0{1,2,3}/no_ckm/best_nmse.pt` |
| `rcflow_ckm` | weights | `rcflow_ckm` | `checkpoints/baselines/rcflow_ckm/best_nmse.pt` |
| `rcflow_no_ckm` | weights | `rcflow_no_ckm` | `checkpoints/baselines/rcflow_no_ckm/best_nmse.pt` |
| `srf_ckm` | weights | `srf_ckm` | `checkpoints/baselines/srf_ckm/best_nmse.pt` |
| `pnp_cm_ckm` | weights | `pnp_cm_ckm` | `checkpoints/baselines/pnp_cm_ckm/best_nmse.pt` |
| `pnp_cm_no_ckm` | weights | `pnp_cm_no_ckm` | `checkpoints/baselines/pnp_cm_no_ckm/best_nmse.pt` |
| `lmmse` | statistics | `lmmse` | `checkpoints/baselines_statistics/in_domain/lmmse_train_only.pt` |
| `lmmse_fold1..3` | statistics | `lmmse` | `checkpoints/baselines_statistics/holdout_0{1,2,3}/lmmse_train_only.pt` |

`flowadmm` and `pnpflow` **share** the straight-flow weights; there is no
separate FlowADMM network. KAP-Flow vs PnP-Flow is a solver choice on the same
prior, exactly as described in the paper. See `docs/CHECKPOINTS.md` for the
download/placement instructions and file hashes.

## 6. Paper Table I rows vs source data labels

`results/table1.md` is the Markdown transcription of
`tables/main_nmse.tex`. The same rows appear with two different label sets in
the source package:

| Table I row | archived CSV label (`source_data/main_table_nmse.csv`) | timing manifest label (`timing/methods.json`) | code method |
| --- | --- | --- | --- |
| KAP-C | `FA-C` | `FA-C` | `flowadmm` |
| KAP-U | `FA-U` | `FA-U` | `flowadmm` |
| PF-C / PF-U | `PF-C` / `PF-U` | `PF-C` / `PF-U` | `pnpflow` |
| RC-U / RC-C / RC-S | `RC-U` / `RC-C` / `RC-S` | `RC-U` / `RC-C` / `RC-S` | `rcflow` / `rcflow` / `srf` |
| CM-U / CM-C | `CM-U` / `CM-C` | `CM-U` / `CM-C` | `pnp-cm` |
| LS / LMMSE | `LS` / `LMMSE` | `LS` / `LMMSE` | `ls` / `lmmse` |

The figure data and the M-budget tables use the same convention:
`FA-C`=KAP-C, `FA-U`=KAP-U.

## 7. Entry points and their naming

| Command | Meaning |
| --- | --- |
| `python run.py ...` (repo root) | general CLI (`journal_flow.cli`): `list/plan/check/evaluate/train/fit-stats`, `--method` selects the paper method key |
| `python main/run.py ...` | convenience wrapper for the main method only; forces `--method flowadmm` and rejects an explicit `--method` override |
| `python baselines/run.py ...` | convenience wrapper for the baselines; defaults to `--method pnpflow`, rejects `flowadmm` |
| `python run_paper.py kap ...` | paper-method launcher mapping `kap -> flowadmm`; methods `kap/pnpflow/rcflow/srf/pnp-cm/ls/lmmse`; output defaults to `<repo>/results` (`KAPFLOW_OUTPUT_ROOT` override) |
| `python run_paper.py timing ...` | runs `timing/benchmark_all.py --manifest timing/methods.json` |
| `main/run_standard.sh`, `main/run_threefold.sh`, `main/run_m_sweep.sh` | standard/tri-fold/M-sweep evaluation tracks |
| `timing/run_paper.sh` | timing benchmark wrapper |

`--track` values: `standard` (in-domain, three seeds for flow methods),
`threefold` (leave-one-site-out, one archived inference seed) and
`robustness` (CKM-mismatch study — the code path exists, but it is not part of
the paper and its configs/weights are not included in this release).

## 8. Notation differences between paper and code

| Paper symbol/term | Code term | Note |
| --- | --- | --- |
| flow time $s\in[0,1]$ | `t`, `time`, `time_batch` | code `t=0` is latent, `t=1` is data |
| endpoint samples $M$ | `latent_samples` (solvers) / `k_avg` (FlowADMM plan) / `--m` (CLI) | $M$ is **not** the pilot length $N_p$; `--m` is validated as a positive integer |
| outer iterations $K$ | `iterations` / `--pnp-iterations` | FlowADMM `K` = `pnp_iterations` |
| inverse penalty $\tau$ | `tau` (FlowADMM) and `rho=1/tau` in the data step | the paper's penalty is $1/(2\tau)$; the PnP data step uses the precision form |
| noise variance $\nu$ | `noise_var`, `sigma2` | complex variance |
| CKM descriptor $\mat m_n$ | `condition["scalar"]`, `["adp"]`, `["tx_beam"]`, `["rx_beam"]` | plus `grid_indices`/`grid_weights` metadata |
| UE state | `condition["state"]` (8 dims) | position(3)+velocity(3)+sin/cos heading(2) |
| $\widetilde f_k,\widetilde t_n$ | `condition["frequency"]`, `condition["time"]` | already normalized in the dataset |
| H (data copy) / Z (prior copy) / U (dual) | `x` / `z` / `scaled_dual` | see `sample_flow_admm` |
| conditional prior refinement | "renoise--denoise" (`_average_renoise_denoise`) | not fine-tuning; it is a forward evaluation of the same weights |
| $\mathcal R_{\rm CKM}$ | not materialized | the flow operator replaces the prior proximal step |
| $\mat Q,\mat Q_\perp$ | `projection`, `apply_range_projection`, `apply_null_projection` | `Q = P @ pinv(P)` in `physics/rc.py` |

## 9. Historical names kept for traceability

These appear in the paper's archived logs/CSVs (which are not redistributed
here) and in the source notes of `docs/RESULTS.md`. They are **not** separate
methods:

* `flowadmm`, `FA`, `FA-U`, `FA-C` -> KAP-Flow/KAP-U/KAP-C.
* `CM`, `CM-U`, `CM-C` -> PnP-CM.
* `PF`, `PF-U`, `PF-C` -> PnP-Flow.
* `RC`, `RC-Flow`, `RC-U`, `RC-C` -> Recursive Flow rows.
* `SRF`, `ckmsrf_full`, `RC-S` -> the RC-C checkpoint trained with the SRF
  objective; `RC-S` is a Table I row, not a separate `journal_flow` method.
* `FIG`, `FIG+`, `fig_plus`, `gm1tr05` -> SRF-era names for PBG components
  (shared ancestral code; see `docs/METHOD.md` Section 8). They are not part of
  the KAP-Flow paper.
* `Diff-U`, `Diff-C`, `PnP-DM` -> diffusion/PnP-DM baselines present in the
  research code and timing manifests but **not** in the paper's Table I.
* `baseline_no_ckm`, `ckmsrf_prior`, `ckmsrf_full`, `ckmcm_prior`,
  `ckmcm_no_ckm`, `ckm_straight_flow_prior`, `ckm_straight_flow_no_ckm` ->
  experiment/run names; see Section 4.
* `straight_flow_no-ckm_in_domain`, `straight_flow_ckm_holdout_scene_0X` ->
  archived training-run directory names for the released weights.

## 10. Intentionally not carried over

* Server-absolute paths and `/home/test/...` defaults: replaced by
  `CKMSRF_DATA_ROOT` / `--data-root` and repo-relative paths.
* The paper PDF and the submission source package are **not** redistributed in
  this repository (copyright / under-review status). Only code, documentation,
  Markdown result tables and (optionally) figure files are shipped.
* The internal grid-search, one-off plotting and archive-audit scripts of the
  paper project are not part of the release; `results/table1.md` and
  `docs/RESULTS.md` record the source files instead.

If a name in a document does not appear here, prefer the code name from
`src/journal_flow/cli.py::METHODS` and treat the document as incomplete rather
than guessing.
