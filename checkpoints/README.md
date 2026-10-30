# KAP-Flow checkpoints

Trained network weights and LMMSE second-order statistics used by the KAP-Flow
paper (under review). The `.pt` files are **not stored in git**: they are
published as GitHub Release assets and installed into this directory by
[`tools/download_checkpoints.py`](../tools/download_checkpoints.py).

[`MANIFEST.json`](MANIFEST.json) is the single source of truth. It lists the 17
files (13 network checkpoints + 4 statistics), what each one is used for in the
paper, where it must be placed, its size, and the SHA256 of both the original
training checkpoint and the released inference file. [`SHA256SUMS.txt`](SHA256SUMS.txt)
contains the same checksums in `sha256sum -c` format.

- **Release tag:** `v1.0-checkpoints`
- **Release assets:** 14 files — 13 inference checkpoints (`*.pt`) plus
  `lmmse_statistics.zip` (contains the 4 LMMSE statistics); `SHA256SUMS.txt` is
  also uploaded for convenience.
- **Files on disk after download:** 17
- **Download size:** 406.9 MiB / 426.7 MB (49.6 % of the 807.1 MiB / 846.4 MB
  original training checkpoints; see [Why Release assets](#why-github-release-assets-and-not-git--git-lfs)).

## Download

From the repository root:

```bash
python tools/download_checkpoints.py --tag v1.0-checkpoints --dest checkpoints/
```

The script downloads every asset, verifies its SHA256 against
`checkpoints/MANIFEST.json` *before* installing it, re-verifies the installed
files, and aborts with a non-zero exit code if anything does not match. It only
needs the Python standard library.

Useful options:

```bash
# show what would be downloaded/placed, write nothing
python tools/download_checkpoints.py --dry-run

# only the two in-domain priors needed for the main table (KAP-C/KAP-U/PF-*)
python tools/download_checkpoints.py --only straight_ckm straight_no_ckm

# offline / local mirror: assets already downloaded into ./assets
python tools/download_checkpoints.py --from-dir ./assets

# overwrite existing files (e.g. after a corrupt download)
python tools/download_checkpoints.py --force
```

Direct URL pattern (if you prefer `curl`/`wget`):

```text
https://github.com/<owner>/<repo>/releases/download/v1.0-checkpoints/<asset-name>
```

## Contents

### What each file is used for

| Checkpoint ID | Published asset | Purpose in the paper |
| --- | --- | --- |
| `straight_ckm` | `main_straight_ckm.pt` | In-domain CKM-conditioned straight-flow prior shared by **KAP-C** and **PF-C** (Table I rows KAP-C/PF-C; Fig. 3 subspace gains; Fig. 4(b) endpoint averaging) |
| `straight_no_ckm` | `main_straight_no_ckm.pt` | In-domain unconditional prior shared by **KAP-U** and **PF-U** (Table I rows KAP-U/PF-U; Fig. 3) |
| `straight_ckm_fold1` | `main_threefold_holdout_01_ckm.pt` | Site-transfer fold 1 (holdout scene 1) CKM prior for KAP-C/PF-C (Fig. 4(a)) |
| `straight_no_ckm_fold1` | `main_threefold_holdout_01_no_ckm.pt` | Site-transfer fold 1 unconditional prior for KAP-U/PF-U (Fig. 4(a)) |
| `straight_ckm_fold2` | `main_threefold_holdout_02_ckm.pt` | Site-transfer fold 2 (holdout scene 2) CKM prior for KAP-C/PF-C (Fig. 4(a)) |
| `straight_no_ckm_fold2` | `main_threefold_holdout_02_no_ckm.pt` | Site-transfer fold 2 unconditional prior for KAP-U/PF-U (Fig. 4(a)) |
| `straight_ckm_fold3` | `main_threefold_holdout_03_ckm.pt` | Site-transfer fold 3 (holdout scene 3) CKM prior for KAP-C/PF-C (Fig. 4(a)) |
| `straight_no_ckm_fold3` | `main_threefold_holdout_03_no_ckm.pt` | Site-transfer fold 3 unconditional prior for KAP-U/PF-U (Fig. 4(a)) |
| `rcflow_ckm` | `baselines_rcflow_ckm.pt` | **RC-C** baseline prior (Table I row RC-C) |
| `rcflow_no_ckm` | `baselines_rcflow_no_ckm.pt` | **RC-U** baseline prior (Table I row RC-U) |
| `srf_ckm` | `baselines_srf_ckm.pt` | **RC-S** baseline prior (Table I row RC-S; SRF) |
| `pnp_cm_ckm` | `baselines_pnp_cm_ckm.pt` | **CM-C** baseline denoiser (Table I row CM-C) |
| `pnp_cm_no_ckm` | `baselines_pnp_cm_no_ckm.pt` | **CM-U** baseline denoiser (Table I row CM-U) |
| `lmmse` | `lmmse_statistics.zip` | In-domain LMMSE second-order statistics (Table I row LMMSE) |
| `lmmse_fold1` | `lmmse_statistics.zip` | Source-only LMMSE statistics for site-transfer fold 1 (Fig. 4(a)) |
| `lmmse_fold2` | `lmmse_statistics.zip` | Source-only LMMSE statistics for site-transfer fold 2 (Fig. 4(a)) |
| `lmmse_fold3` | `lmmse_statistics.zip` | Source-only LMMSE statistics for site-transfer fold 3 (Fig. 4(a)) |

KAP-Flow/FlowADMM and PnP-Flow share the same trained flow prior, so one prior
file serves both of their rows. LS needs no checkpoint. The four LMMSE files are
small (1.7–2.2 MB each) and are shipped inside one zip asset.

### Where the files are placed

Paths are relative to the repository root (`checkpoints/` = this directory) and
are exactly the `target` paths in [`configs/checkpoints.json`](../configs/checkpoints.json),
so the code finds them without any extra configuration.

| Checkpoint ID | Installed path | Size | SHA256 (first 16 hex) |
| --- | --- | --- | --- |
| `straight_ckm` | `checkpoints/main/straight/ckm/best_nmse.pt` | 32.32 MB | `f2b3520b16e3ba35` |
| `straight_no_ckm` | `checkpoints/main/straight/no_ckm/best_nmse.pt` | 32.32 MB | `79877c3d67001c84` |
| `straight_ckm_fold1` | `checkpoints/main/threefold/holdout_01/ckm/best_nmse.pt` | 32.32 MB | `ef715566ca0359a5` |
| `straight_no_ckm_fold1` | `checkpoints/main/threefold/holdout_01/no_ckm/best_nmse.pt` | 32.32 MB | `5e35e8a4adddcc6a` |
| `straight_ckm_fold2` | `checkpoints/main/threefold/holdout_02/ckm/best_nmse.pt` | 32.32 MB | `c0c60040a0fc0b26` |
| `straight_no_ckm_fold2` | `checkpoints/main/threefold/holdout_02/no_ckm/best_nmse.pt` | 32.32 MB | `604362bc5137fb1d` |
| `straight_ckm_fold3` | `checkpoints/main/threefold/holdout_03/ckm/best_nmse.pt` | 32.32 MB | `68e8e6f08bbc93f2` |
| `straight_no_ckm_fold3` | `checkpoints/main/threefold/holdout_03/no_ckm/best_nmse.pt` | 32.32 MB | `bc5ab44367ba762d` |
| `rcflow_ckm` | `checkpoints/baselines/rcflow_ckm/best_nmse.pt` | 32.46 MB | `88cf42bf7231c189` |
| `rcflow_no_ckm` | `checkpoints/baselines/rcflow_no_ckm/best_nmse.pt` | 31.46 MB | `a04b9f447d2f5558` |
| `srf_ckm` | `checkpoints/baselines/srf_ckm/best_nmse.pt` | 32.46 MB | `0796fc56a6d78ad5` |
| `pnp_cm_ckm` | `checkpoints/baselines/pnp_cm_ckm/best_nmse.pt` | 32.32 MB | `4bca4f3cbe3fb8dd` |
| `pnp_cm_no_ckm` | `checkpoints/baselines/pnp_cm_no_ckm/best_nmse.pt` | 32.32 MB | `aa1ed3b3f8bedbf5` |
| `lmmse` | `checkpoints/baselines_statistics/in_domain/lmmse_train_only.pt` | 2.19 MB | `f7bee570d3a58d79` |
| `lmmse_fold1` | `checkpoints/baselines_statistics/holdout_01/lmmse_train_only.pt` | 1.66 MB | `baa1a19c1ad17240` |
| `lmmse_fold2` | `checkpoints/baselines_statistics/holdout_02/lmmse_train_only.pt` | 1.66 MB | `f3614b077511e92b` |
| `lmmse_fold3` | `checkpoints/baselines_statistics/holdout_03/lmmse_train_only.pt` | 1.66 MB | `d0e2d40ae17ef4da` |

The full SHA256 of every file is in [`MANIFEST.json`](MANIFEST.json) and
[`SHA256SUMS.txt`](SHA256SUMS.txt).

## Manual verification

The download script verifies everything itself, but you can also check the
installed files independently (Git Bash / WSL / macOS / Linux):

```bash
cd checkpoints
sha256sum -c SHA256SUMS.txt
```

On Windows PowerShell, `Get-FileHash <file> -Algorithm SHA256` works per file;
the download script is the easiest path because it also verifies the archives and
places the files correctly.

## Using the checkpoints

- Evaluation restores the network architecture, normalization statistics and
  model/EMA weights from the checkpoint itself. `src/ckmsrf/evaluate.py`
  loads `model` first and then overwrites it with `ema.shadow`
  (`prefer_ema=True`), so **all reported numbers use the EMA weights**;
  `--use-model-weights` disables that for ablations.
- `configs/checkpoints.json` is the id → path registry used by `run.py`:

  ```bash
  python run.py list                              # print the registry
  python run.py check --method flowadmm --track standard \
      --data-root /path/to/data --device cpu      # verify files, SHA256 and metadata
  python run.py evaluate --method flowadmm --track standard \
      --data-root /path/to/data --device cuda:0   # run the planned evaluations
  ```

- A dataset is required to compute NMSE; it is not distributed with the
  repository. See [`docs/DATA.md`](../docs/DATA.md) and `data_generation/`, and
  pass the data root via `--data-root` or `CKMSRF_DATA_ROOT`.
- These inference exports **cannot resume training exactly** (the Adam
  optimizer state, LR scheduler, AMP scaler and RNG state were removed to halve
  the file size). They are numerically identical to the training checkpoints for
  evaluation.
- The published assets were produced with
  [`tools/export_inference_ckpt.py`](../tools/export_inference_ckpt.py); for
  details on the retained keys, the EMA semantics and the verification steps see
  [`docs/CHECKPOINTS.md`](../docs/CHECKPOINTS.md).

## Why GitHub Release assets (and not git / Git LFS)

Original training checkpoints are 61.6–64.9 MB each (846.4 MB for all 17 files),
because each one stores the parameter set twice (`model` + `ema.shadow`) plus the
Adam optimizer state. Only inference needs `model`/`ema`/`config`/
`normalization`, so the released files are 30.8–31.0 MB each (426.7 MB total).
Even so, binaries of this size do not belong in a git history:

| Option | Repository / clone cost | Constraints | Verdict |
| --- | --- | --- | --- |
| Full training checkpoints in git | +846.4 MB (807.1 MiB) | Below GitHub's 100 MiB per-file hard limit, but close to the 1 GB repository soft limit; every clone pays it; binary history cannot be diffed | Rejected |
| Inference exports in git | +426.7 MB (406.9 MiB) | Smaller, but still a slow clone for every reviewer and no versioning benefit for weights | Rejected |
| Git LFS | Storage counts against the free 1 GB LFS quota; each download against the 1 GB/month bandwidth quota | ~2–3 full downloads per month exhaust the free bandwidth for a public paper repo; download failures are likely exactly during review | Rejected |
| **GitHub Release assets (chosen)** | **0 MB in the repository** | 2 GiB per-asset limit, CDN download, partial downloads, assets can be replaced without rewriting git history | **Chosen** |

This is why `*.pt` is in `.gitignore` and the repository only tracks this README,
`MANIFEST.json` and `SHA256SUMS.txt`. Downloading only the files you need is
cheap (`--only`), and the SHA256 checks make the provenance of every file
verifiable.

## Citation and license

The code and weights are released under the MIT License (see [`LICENSE`](../LICENSE)).
If you use these checkpoints, please cite the KAP-Flow paper (see
[`CITATION.cff`](../CITATION.cff) / [`citation.bib`](../citation.bib)). The
manuscript is currently under review; please link to the official version once
available.
