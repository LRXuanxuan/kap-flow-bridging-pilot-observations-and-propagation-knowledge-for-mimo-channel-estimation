# Table I -- NMSE (dB) and inference time (ms/channel)

Transcription of the paper's Table I in
`KAP-Flow: Bridging Pilot Observations and Propagation Knowledge for MIMO
Channel Estimation` (manuscript under review; no venue assigned at release
time).

**Provenance.** The NMSE values are transcribed from the paper's
`tables/main_nmse.tex`; the machine-readable copy
`source_data/main_table_nmse.csv` (same 11 rows x 9 SNR points) **is** shipped
in this repository at [`../source_data/main_table_nmse.csv`](../source_data/main_table_nmse.csv).
The paper source bundle and PDF are not redistributed.

* NMSE values: `tables/main_nmse.tex` / `source_data/main_table_nmse.csv`.
* Time column: the paper's unified timing scan (two RTX 5090 runs averaged;
  see the protocol note below). The release reproduces this measurement with
  [`timing/benchmark_all.py`](../timing/benchmark_all.py) and
  [`timing/methods.json`](../timing/methods.json).

Bold marks the lowest reported NMSE at that SNR across all rows (as in the
paper): **CM-C at -10 dB, KAP-C from -5 to 30 dB**.

| Method | -10 | -5 | 0 | 5 | 10 | 15 | 20 | 25 | 30 | t (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LS | 6.18 | 2.45 | 0.00 | -1.19 | -1.64 | -1.80 | -1.85 | -1.86 | -1.87 | 0.236 |
| LMMSE | 1.07 | -0.19 | -1.35 | -2.23 | -2.81 | -3.14 | -3.30 | -3.37 | -3.39 | 0.033 |
| RC-U | 0.35 | 0.27 | 0.06 | -0.51 | -1.25 | -1.71 | -1.98 | -2.17 | -2.24 | 13.241 |
| RC-C | 0.18 | -0.14 | -0.70 | -1.53 | -2.26 | -2.71 | -2.97 | -3.06 | -3.06 | 15.297 |
| RC-S | 0.14 | -0.19 | -0.78 | -1.64 | -2.38 | -2.83 | -3.09 | -3.18 | -3.18 | 14.912 |
| CM-U | -0.94 | -1.76 | -2.06 | -2.35 | -2.56 | -2.65 | -2.69 | -2.70 | -2.63 | 17.246 |
| CM-C | **-2.55** | -3.04 | -3.44 | -3.87 | -4.19 | -4.36 | -4.41 | -4.43 | -4.36 | 17.106 |
| PF-U | 5.24 | 1.05 | -1.93 | -3.41 | -3.91 | -4.07 | -4.11 | -4.11 | -4.08 | 15.344 |
| PF-C | 4.82 | 0.52 | -2.65 | -4.17 | -4.80 | -4.88 | -4.89 | -4.81 | -5.00 | 15.349 |
| KAP-U | 0.02 | -1.20 | -2.64 | -3.72 | -4.33 | -4.93 | -5.37 | -5.59 | -5.69 | 16.467 |
| KAP-C | -1.90 | **-3.52** | **-5.18** | **-6.21** | **-6.89** | **-7.37** | **-7.58** | **-7.76** | **-7.87** | 16.422 |

NMSE is evaluated as `10 log10(||H_hat - H||_F^2 / ||H||_F^2)` in dB, averaged
in linear units over records, seeds and the nine SNR points (the overline
notation in the manuscript), then converted to dB for the table.

## Row names: FA vs KAP

The paper table uses **KAP** for the proposed method. The archived experiment
files that produced the same numbers are labelled **FA** (FlowADMM):

| Paper row | Archived label | Code method (`journal_flow.cli`) | Notes |
| --- | --- | --- | --- |
| KAP-C | `FA-C` | `flowadmm` + CKM prior | conditioned KAP-Flow |
| KAP-U | `FA-U` | `flowadmm` + no-CKM prior | unconditioned control |
| PF-C / PF-U | `PF-C` / `PF-U` | `pnpflow` | shares the same straight-flow prior as KAP per condition |
| RC-U / RC-C / RC-S | `RC-U` / `RC-C` / `RC-S` | `rcflow`, `rcflow`, `srf` | Recursive Flow; RC-S = RC-C + SRF |
| CM-U / CM-C | `CM-U` / `CM-C` | `pnp-cm` | PnP-CM |
| LS / LMMSE | `LS` / `LMMSE` | `ls` / `lmmse` | classical baselines |

See [`docs/NAMING.md`](../docs/NAMING.md) for the complete naming map.

## Table composition caveat (important)

The rows of Table I were assembled from two different evaluation cohorts
(recorded in the submission-side protocol notes and in the archived result
metadata):

* **LS, RC-U/C/S, CM-U/C:** archived full-test evaluations;
* **LMMSE, PF-U/C, KAP-U/C:** paired post-hoc subset with 5997 of the 6000
  test records retained per SNR (three trajectories excluded:
  scene 1 / id 41, scene 2 / id 2094 and record 2844; the cause was not
  established, so this is a sensitivity subset, not data cleaning).

Consequently the table is **not** a strict same-cohort ranking between all
rows, and the paper does not claim it is. The paired conditioning/solver
comparisons (U vs C, KAP vs PF) are computed within the same 5997-record
subset. See [`docs/RESULTS.md`](../docs/RESULTS.md) for the paired numbers and
their source files.

## Timing protocol (paper Sec. IV-D)

* Receiver-only wall time on resident GPU tensors, synchronized at both ends;
  includes sampler diagnostics and initialization; excludes data I/O,
  observation generation, metrics, weight loading and CPU-to-GPU transfer.
* FP32, batch size 16, one warm-up plus one timed batch for each of the 27
  site--SNR cells; two independent RTX 5090 runs, averaged (the table values
  are that average); `K=40`, `M=1` for KAP.
* The 16.42 ms (KAP-C) vs 15.35 ms (PF-C) vs 17.11 ms (CM-C) comparison in the
  manuscript is taken from this column; KAP-C adds 7.0% over PF-C.
* The released `timing/benchmark_all.py` implements the same timing boundary.
  Its published defaults are `--samples-per-scene 32 --warmup 2 --repeats 1`;
  for a paper-style number use a 16-sample cell and one warm-up plus one timed
  batch (`--samples-per-scene 16 --warmup 1 --repeats 1`), and average two
  runs. Different sample budgets must not be mixed in one ranking.

## Notes on number formatting

* The paper (and this transcription) prints two decimals; e.g. LS at 0 dB is
  `-0.0020 dB` in `source_data/main_table_nmse.csv` and prints as `0.00`.
* Times are printed to three decimals exactly as in `tables/main_nmse.tex`
  (not recomputed or rounded here).
* The time column is amortized per channel at batch size 16; it is **not**
  single-request latency.

## Reproducing with the released code

```bash
# 1) prepare data (see docs/DATA.md) and place weights (see docs/CHECKPOINTS.md)
export CKMSRF_DATA_ROOT=/path/to/ckm_srf_64x16_h5

# 2) inspect the exact evaluation jobs the paper protocol defines
python run.py plan --method flowadmm --track standard

# 3) validate checkpoints/statistics/conditions before running
python run.py check --method flowadmm --track standard

# 4) evaluate (KAP-C/KAP-U; outputs under <repo>/results by default)
python run_paper.py kap evaluate --track standard --resume

# 5) PnP-Flow, RC/SRF, PnP-CM, LS/LMMSE with the same protocol
python run_paper.py pnpflow evaluate --track standard --resume
python run_paper.py rcflow  evaluate --conditions ckm --resume
python run_paper.py srf     evaluate --conditions ckm --resume
python run_paper.py pnp-cm  evaluate --track standard --resume
python run_paper.py ls      evaluate --resume
python run_paper.py lmmse   evaluate --resume

# 6) timing (paper-style: 16 samples/scene, 1 warm-up + 1 timed batch)
python run_paper.py timing --samples-per-scene 16 --warmup 1 --repeats 1
```

The paper results were produced from the archived test sets; exact row-by-row
regeneration also requires the same data release and the checkpoints listed in
`docs/CHECKPOINTS.md`. The paired post-hoc subset used for LMMSE/PF/KAP is a
property of the archived evaluation metadata, not of the code defaults (the
code evaluates the full test split unless the archived subset is reproduced
separately).
