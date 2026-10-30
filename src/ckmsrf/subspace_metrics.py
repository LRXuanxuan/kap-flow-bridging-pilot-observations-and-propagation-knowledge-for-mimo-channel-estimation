"""Pilot-subspace diagnostics, evaluated in complex128 outside sampler timing."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch

ID_FIELDS = ("scene_id", "trajectory_id", "snapshot_idx", "subcarrier_idx", "sample_seed")


def pilot_subspace_errors(estimate, truth, pilot, *, rtol=1e-6):
    """Per-sample linear contributions with the SAME total-channel denominator.

    Inputs: complex [B,Nr,Nt], [B,Nr,Nt], [B,Nt,Np] (or shared [Nt,Np]).
    Numerical range(P) uses singular values > rtol*s_max. This is a rank
    decision, not an assumption that P has independent/orthonormal columns.
    """
    if not 0 < rtol < 1:
        raise ValueError("rtol must be in (0,1)")
    if not all(x.is_complex() for x in (estimate, truth, pilot)):
        raise ValueError("Use complex channel/pilot tensors, not stacked real/imag channels")
    estimate, truth, pilot = [x.detach().to(device="cpu", dtype=torch.complex128)
                              for x in (estimate, truth, pilot)]
    if estimate.ndim != 3 or estimate.shape != truth.shape:
        raise ValueError("estimate and truth must have the same [B,Nr,Nt] shape")
    if pilot.ndim == 2:
        pilot = pilot.unsqueeze(0).expand(truth.shape[0], -1, -1)
    if pilot.ndim != 3 or pilot.shape[:2] != (truth.shape[0], truth.shape[2]):
        raise ValueError("pilot must have shape [B,Nt,Np] or [Nt,Np]")
    if not all(torch.isfinite(x).all() for x in (estimate, truth, pilot)):
        raise ValueError("Non-finite subspace inputs")
    energy = truth.abs().square().sum((1, 2))
    if (energy <= 1e-12).any():
        raise ValueError("Channel energy <= 1e-12: normalized NMSE is undefined/unstable")
    u, s, _ = torch.linalg.svd(pilot, full_matrices=False)
    mask = s > rtol * s[:, :1]
    basis = u * mask.unsqueeze(1)
    q = basis @ basis.mH
    error = estimate - truth
    obs = error @ q
    null = error - obs
    e_obs = obs.abs().square().sum((1, 2)) / energy
    e_null = null.abs().square().sum((1, 2)) / energy
    nmse = error.abs().square().sum((1, 2)) / energy
    closure = (e_obs + e_null - nmse).abs()
    if not torch.allclose(e_obs + e_null, nmse, rtol=1e-10, atol=1e-12):
        raise AssertionError("Subspace energy decomposition failed")
    return {
        "nmse_linear": nmse, "e_obs": e_obs, "e_null": e_null,
        "h_obs_fraction": (truth @ q).abs().square().sum((1, 2)) / energy,
        "pilot_rank": mask.sum(1), "null_dim": truth.shape[2] - mask.sum(1),
        "decomposition_abs_error": closure,
        "qperp_p_relative": (pilot - q @ pilot).abs().square().sum((1, 2)).sqrt()
                              / pilot.abs().square().sum((1, 2)).sqrt().clamp_min(1e-300),
    }


def problem_fingerprints(truth, pilot, observation, noise_var):
    """Exact per-sample fingerprints verify pairing of the actual inverse problem."""
    tensors = [x.detach().cpu().contiguous() for x in (truth, pilot, observation, noise_var)]
    hashes = []
    for i in range(truth.shape[0]):
        digest = hashlib.sha256()
        for tensor in tensors:
            value = tensor[i].contiguous()
            digest.update(str((tuple(value.shape), str(value.dtype))).encode())
            digest.update(value.numpy().tobytes())
        hashes.append(digest.hexdigest())
    return hashes


class SubspaceRecorder:
    """Stream per-sample output; publish summaries only after evaluation succeeds."""
    def __init__(self, output_dir: Path, *, rtol=1e-6):
        self.output_dir, self.rtol = Path(output_dir), rtol
        self.partial = self.output_dir / "subspace_per_sample.partial.csv"
        self.handle = self.partial.open("w", newline="", encoding="utf-8")
        self.writer = None
        self.groups = {}

    def add(self, estimate, truth, pilot, batch, *, snr_db, iterations, fingerprints):
        metrics = {k: v.tolist() for k, v in
                   pilot_subspace_errors(estimate, truth, pilot, rtol=self.rtol).items()}
        ids = {}
        for name in ID_FIELDS:
            if name not in batch:
                raise ValueError(f"Subspace pairing requires sample identifier {name}")
            ids[name] = torch.as_tensor(batch[name]).reshape(-1).tolist()
        for i in range(truth.shape[0]):
            row = {"snr_db": float(snr_db), "iterations": iterations,
                   **{k: int(v[i]) for k, v in ids.items()},
                   "problem_sha256": fingerprints[i],
                   **{k: v[i] for k, v in metrics.items()}}
            if self.writer is None:
                self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            for scene in (row["scene_id"], "overall"):
                group = self.groups.setdefault((float(snr_db), iterations, scene), {
                    "count": 0, "sums": {k: 0.0 for k in metrics},
                    "max_error": 0.0, "rank_min": row["pilot_rank"], "rank_max": row["pilot_rank"]})
                group["count"] += 1
                for k in metrics:
                    group["sums"][k] += row[k]
                group["max_error"] = max(group["max_error"], row["decomposition_abs_error"])
                group["rank_min"] = min(group["rank_min"], row["pilot_rank"])
                group["rank_max"] = max(group["rank_max"], row["pilot_rank"])
        self.handle.flush()

    def finish(self):
        self.handle.close()
        if not self.groups:
            raise ValueError("No subspace samples evaluated")
        rows = []
        for (snr, iterations, scene), group in self.groups.items():
            row = {"snr_db": snr, "iterations": iterations, "scene_id": scene,
                   "num_samples": group["count"]}
            row.update({k: value / group["count"] for k, value in group["sums"].items()})
            row["max_decomposition_abs_error"] = group["max_error"]
            row["pilot_rank_min"] = group["rank_min"]
            row["pilot_rank_max"] = group["rank_max"]
            rows.append(row)
        with (self.output_dir / "subspace_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.partial.replace(self.output_dir / "subspace_per_sample.csv")
