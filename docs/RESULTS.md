# Results: reported numbers and where they come from

All values in this document are taken from the submitted manuscript or from
the paper's archived result files. Nothing is retrained or re-estimated for
this release. Each section lists the exact source files.

**Provenance convention.** Paths of the form `source_data/...` are relative to
the **repository root** and the cited files *are* shipped here (see
[`../source_data/`](../source_data/)); only the subset of the paper's archived
result files needed to re-derive the numbers in this document is included.
Paths of the form `tables/...` refer to the paper *submission source package*,
which is not redistributed (neither the PDF nor the LaTeX bundle). Paths of the
form `timing/...` and `results/...` are in this repository. Where a number is an
aggregate that the paper quotes in the text but does not tabulate, the
derivation rule is stated explicitly.

**Aggregate NMSE rule used everywhere below.** Average the per-SNR linear NMSE
over the nine equally weighted SNRs, then convert to dB:
`10 log10( (1/9) * sum_{xi in {-10,...,30}} NMSE_lin(xi) )`. This is the paper's
"aggregate" convention (manuscript Sec. IV-A). For gains expressed in dB,
`gain = aggregate_dB(baseline) - aggregate_dB(method)`; the same value is
obtained by comparing the linear aggregates in dB.

## 1. Aggregate NMSE of the main table (derived from Table I)

Derived from `tables/main_nmse.tex` / `source_data/main_table_nmse.csv` with
the rule above (the paper does not print this column; it follows from the
printed rows):

| Method | aggregate NMSE (dB) | cohort note |
| --- | ---: | --- |
| LS | 0.859 | archived full test |
| LMMSE | -1.788 | paired post-hoc subset |
| RC-U | -0.901 | archived full test |
| RC-C | -1.626 | archived full test |
| RC-S | -1.717 | archived full test |
| CM-U | -2.222 | archived full test |
| CM-C | -3.800 | archived full test |
| PF-U | -0.690 | paired post-hoc subset |
| PF-C | -1.272 | paired post-hoc subset |
| KAP-U | -3.250 | paired post-hoc subset |
| KAP-C | -5.518 | paired post-hoc subset |

Because the cohorts differ, only comparisons inside the same cohort are used
for the paper's claims (the paired post-hoc subset for LMMSE/PF/KAP). The
aggregate values for `flowadmm`/`pnp-flow`/`lmmse` also appear directly in
`source_data/current_20260918/solver_seed_summary.csv`
(`nmse_linear_mean` averaged over 9 SNR rows, `scene_id == "overall"`).

## 2. Conditioning and solver gains (paper Sec. IV-B)

| Claim (paper text) | Value | Derivation / source |
| --- | ---: | --- |
| Joint conditioning improves aggregate NMSE, KAP-Flow | **2.27 dB** | KAP-C vs KAP-U aggregates: -5.518 - (-3.250) = 2.268 dB (`source_data/main_table_nmse.csv`; same pair in `source_data/current_20260918/solver_seed_summary.csv`) |
| Joint conditioning improves aggregate NMSE, PnP-Flow | **0.58 dB** | PF-C vs PF-U: -1.272 - (-0.690) = 0.582 dB (same files) |
| Same conditional prior, KAP-Flow vs PnP-Flow | **4.25 dB** | KAP-C (-5.518) vs PF-C (-1.272) |
| Same unconditional prior, KAP-Flow vs PnP-Flow | **2.56 dB** | KAP-U (-3.250) vs PF-U (-0.690) |
| All four gains persist across the evaluated SNRs | qualitative | paper Sec. IV-B |
| (derived, not stated in the paper) PnP-CM conditioning gain | 1.58 dB | CM-C (-3.800) vs CM-U (-2.222) |
| (derived, not stated in the paper) RC-C vs RC-U | 0.72 dB | RC-C (-1.626) vs RC-U (-0.901) |
| (derived, not stated in the paper) RC-S vs RC-C | 0.09 dB | RC-S (-1.717) vs RC-C (-1.626) |

The identical numbers are recovered from the archived per-record summaries in
`source_data/current_20260918/all_method_summary.csv`
(`nmse_linear_mean`, `e_obs_mean`, `e_null_mean`; rows with
`scene_id == "overall"`, 5997 records/SNR, 3 inference seeds).

## 3. Subspace analysis (paper Sec. IV-B, Fig. 3)

Definitions (paper, Eq. after Table I): for `A in {Q, Q_perp}`,
`E_A = ||(H_hat - H) A||_F^2 / ||H||_F^2`, so `E_obs + E_null = NMSE` in
linear units. The implementation is
`src/ckmsrf/subspace_metrics.py::pilot_subspace_errors`, with `Q` formed from
the numerically truncated range of `P` (`rtol = 1e-6`).

Source: `source_data/current_20260918/solver_seed_summary.csv`
(linear means over records and 3 inference seeds, 5997 records/SNR);
figure contract `source_data/fig3_contract.json`. Aggregates over the 9 SNRs:

| Quantity (aggregate linear) | KAP-C | KAP-U | PF-C | PF-U | LMMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `e_obs` | 0.05118 | 0.07882 | 0.39988 | 0.40943 | 0.10673 |
| `e_null` | 0.22950 | 0.39435 | 0.34627 | 0.44376 | 0.55572 |
| `e_obs + e_null` | 0.28068 | 0.47317 | 0.74615 | 0.85318 | 0.66245 |

Derived shares reported in the paper:

| Claim (paper text) | Value | Derivation |
| --- | ---: | --- |
| Share of the **conditioning** gain in pilot-null directions, KAP-Flow | **85.6 %** | (e_null(KAP-U) - e_null(KAP-C)) / total reduction = 0.164846 / 0.192486 = 85.64 % |
| Share of the **conditioning** gain in pilot-null directions, PnP-Flow | **91.1 %** | 0.097482 / 0.107030 = 91.08 % |
| Share of the **solver** gain (KAP-C over PF-C) in observable directions | **74.9 %** | 0.348700 / 0.465475 = 74.91 % |

The paper also notes that full-channel prior refinement changes null errors,
and that at high SNR KAP-C has higher observable error than PF-C but lower null
error and lower total NMSE. The derived observable share for KAP vs PF under
the unconditioned prior is 87.0 % (not stated in the paper).

## 4. Site transfer (paper Sec. IV-C, Fig. 4(a))

Protocol: leave-one-site-out; train/validate on two source sites (source-only
normalization and checkpoint selection); evaluate 2000 full-test records per
site and SNR at the target site using the target historical CKM; globally
fixed solver settings (`K = 40`); LMMSE uses source-only statistics; one
inference seed. Source: `source_data/cross_scene_updated_20260918.json`
(per-fold aggregates and in-domain references; `aggregation` field: "Full
test: 2000 samples per site/SNR ... one inference seed"), cross-checked against
`source_data/current_20260918/threefold_threefold_summary.csv`.

| Held-out site | KAP-C (dB) | in-domain KAP-C, same site (dB) | gap (dB) | PF-C (dB) | PF-U (dB) | KAP-U (dB) | LMMSE (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -4.253 | -4.554 | **+0.30** | -1.646 | -0.582 | -2.494 | -1.843 |
| 2 | -5.098 | -5.050 | **-0.05** | -1.657 | -0.356 | -2.829 | -1.959 |
| 3 | -4.708 | -7.234 | **+2.53** | -0.635 | -0.540 | -3.455 | -1.999 |
| macro (mean over sites) | **-4.673** | -- | -- | **-1.285** | -0.491 | -2.908 | **-1.933** |

Paper claims and their verification:

* Transferred KAP-C achieves **-4.67 dB** versus **-1.29 dB** for PF-C and
  **-1.93 dB** for LMMSE (macro values above; paper rounds to two decimals).
* Conditioning improves KAP-Flow in every fold by **1.25--2.27 dB**
  (KAP-U - KAP-C per fold: 1.759 / 2.269 / 1.253 dB).
* KAP-Flow outperforms PnP-Flow with either prior in every fold (all six
  comparisons above).
* The gaps to in-domain KAP-C are **+0.30, -0.05, +2.53 dB** (transferred
  minus in-domain; positive = worse transfer), showing site-dependent transfer
  difficulty.

Caveat: one inference seed for the transfer evaluations; the "in-domain
KAP-C, same site" reference is the per-site in-domain aggregate stored in the
same JSON. Do not compare these numbers directly with the 5997-record
three-seed in-domain aggregates of Section 1.

## 5. Sampling budget: endpoint averaging (paper Sec. IV-D, Fig. 4(b))

Protocol: fixed `K = 40`; `M` endpoint samples per PPC iteration; `NFE = K*M`;
all other settings as the standard protocol. Accuracy is reported on the paired
post-hoc subset; the timing scan is the separate full-test archived timing run
(the two cohorts differ - this is stated in the source metadata and must not
be merged into a new accuracy claim).

Sources:
`source_data/current_20260918/m_seed_summary.csv` (accuracy; aggregation as in
Section 1), `source_data/current_20260918/mc_full_m_seed_summary.csv`
(`sampling_ms_per_sample_mean`, averaged over the 9 SNR rows), and the release
copy `timing/accuracy_reference_m.csv` (same accuracy values: `subset_nmse_db`,
`full_nmse_db`).

| M | NFE | KAP-C NMSE, paired subset (dB) | KAP-C time, full-test scan (ms/ch) | time / M=1 | gain over M=1 (dB) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40 | -5.518 | 21.341 | 1.000 | 0.000 |
| 2 | 80 | -6.091 | 31.385 | 1.471 | 0.573 |
| 4 | 160 | -6.440 | 55.202 | 2.587 | 0.922 |
| 8 | 320 | -6.654 | 115.443 | 5.409 | 1.136 |
| 16 | 640 | -6.771 | 216.680 | 10.153 | 1.253 |

Paper claims and their verification:

* Increasing `M` from 1 to 16 improves KAP-C from **-5.52 to -6.77 dB**, while
  network evaluations rise from 40 to 640. (Exact: -5.5179 -> -6.7708 dB;
  1.2529 dB gain.)
* The separate full-test timing scan measures **21.34 to 216.68 ms/channel**
  at batch size 16: a **1.25-dB gain at 10.15 times the runtime**
  (216.680 / 21.341 = 10.153).
* Doubling `M` from 8 to 16 adds only **0.12 dB** while nearly doubling runtime
  (0.1167 dB; 216.680 / 115.443 = 1.877x).
* `M = 1` provides the default accuracy--cost balance.

Note: the timing column here is the archived full-test timing scan
(`sampling_ms_per_sample_mean`), not the two-GPU unified timing of Table I; the
two numbers are therefore not directly comparable (21.34 vs 16.42 ms for KAP-C
at `M = 1`).

## 6. Inference cost (paper Sec. IV-D, Table I time column)

Protocol: receiver-only wall time on resident GPU tensors, synchronized at both
ends, including sampler diagnostics and initialization; excluding data I/O,
observation generation, metrics, weight loading and host-to-device transfer.
FP32, batch size 16, one warm-up plus one timed batch per site--SNR cell (27
cells per method), two independent RTX 5090 runs averaged. Source:
`tables/main_nmse.tex` time column (paper), reproduced by
`timing/benchmark_all.py` with `timing/methods.json`.

| Method | ms/channel | vs KAP-C |
| --- | ---: | --- |
| LS | 0.236 | -- |
| LMMSE | 0.033 | -- |
| RC-U | 13.241 | -- |
| RC-C | 15.297 | -- |
| RC-S | 14.912 | -- |
| CM-U | 17.246 | -- |
| CM-C | 17.106 | KAP-C is 0.68 ms faster |
| PF-U | 15.344 | -- |
| PF-C | 15.349 | KAP-C is 7.0 % slower (16.422 vs 15.349) |
| KAP-U | 16.467 | -- |
| KAP-C | 16.422 | -- |

Related facts from the paper: the network has **4.01 M parameters** (Sec.
IV-A; `docs/CHECKPOINTS.md` records 4,010,818 parameters for the released
checkpoints), and a recovery costs `K*M` network evaluations (40 for the
default `K=40, M=1`).

## 7. Protocol summary (paper Sec. IV-A)

Three independent 200 x 200 m sites, 21 x 21 CKM grids with 10 m spacing,
40 x 40 m blocks split 15/5/5; `(N_t, N_r, N_p) = (64, 16, 19)`;
`f_c = 28 GHz`, `Delta f = 15 kHz`, `(N_f, N_s) = (48, 8)`, `Delta t = 1 ms`,
UE speed 20--60 m/s; each CKM node aggregates 32 historical channels into
16 x 16 x 12 angular-delay bins and 64/16 beam features; 10000/1000/2000
train/validation/test records per site; test subcarrier/time pair
`(k, n) = (23, 3)` (zero-based); 16 `64 x 19` DFT pilot matrices; SNR grid
`{-10, -5, ..., 30}` dB with
`nu = ||H P||_F^2 / (N_r N_p 10^{xi/10})`; aggregate NMSE averaged over
records, seeds and the nine SNRs. See `docs/DATA.md` for the data contract and
`docs/METHOD.md` for the solver settings.

## 8. Caveats and open items

1. **Table I cohorts differ.** LS/RC/CM are archived full-test results; LMMSE,
   PF and KAP are the paired post-hoc subset with 5997 of the 6000 test records
   per SNR retained (three trajectories excluded post hoc; cause unresolved).
   The paper does not present the table as a strict same-cohort ranking; the
   conditioning/solver claims are within the paired subset.
2. **Transfer uses one inference seed** (Fig. 4(a)), unlike the three-seed
   in-domain main table.
3. **Subspace panel** (Fig. 3) uses the paired post-hoc subset and evaluates
   LMMSE once.
4. **M budget**: accuracy cohort (paired subset, 5997 records/SNR) and timing
   cohort (full-test archived timing scan) differ; the paper keeps them
   separate.
5. **No new experiments were run for this release.** Numbers are the archived
   ones; no result in this document should be read as a new evaluation.
6. **Citation metadata is provisional** - the manuscript is under review and
   the target venue is not assigned; see `CITATION.cff` and `citation.bib`.
   Do not cite a venue or DOI until the authors update these files.

## 9. Source file index

| Claim | File |
| --- | --- |
| Table I NMSE rows | `tables/main_nmse.tex`, `source_data/main_table_nmse.csv` |
| Table I time column | `tables/main_nmse.tex` (paper; two-GPU timing scan) |
| Conditioning/solver gains, subspace errors | `source_data/current_20260918/solver_seed_summary.csv`; `source_data/current_20260918/all_method_summary.csv`; `source_data/fig3_contract.json` |
| Site transfer (per-fold aggregates, in-domain references, gaps) | `source_data/cross_scene_updated_20260918.json`; `source_data/current_20260918/threefold_threefold_summary.csv`; `source_data/fig4_contract.json` |
| M budget accuracy | `source_data/current_20260918/m_seed_summary.csv`; release copy `timing/accuracy_reference_m.csv` |
| M budget timing | `source_data/current_20260918/mc_full_m_seed_summary.csv` |
| Timing protocol / reproduction | `timing/benchmark_all.py`, `timing/methods.json`, `timing/run_paper.sh` |
| Method/hyperparameters | `manuscript Secs. II--IV`; `main/configs/standard/*.yaml`; `configs/experiments/*.yaml` |
| Model size | manuscript Sec. IV-A; checkpoint manifest (see `docs/CHECKPOINTS.md`) |
