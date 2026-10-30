#!/usr/bin/env python3
"""Download the KAP-Flow checkpoints from a GitHub Release into ``checkpoints/``.

The repository intentionally does not store the ``.pt`` files in git: they are
GitHub Release assets (see ``checkpoints/README.md``).  This script downloads the
assets listed in ``checkpoints/MANIFEST.json``, verifies every SHA256 and places
each file at its ``target_relative_path`` (e.g.
``checkpoints/main/straight/ckm/best_nmse.pt``), which is the layout expected by
``configs/checkpoints.json`` and the evaluation entry points.

Only the Python standard library is required (no ``requests``/``gh``).

Typical use, from the repository root::

    python tools/download_checkpoints.py --tag v1.0-checkpoints --dest checkpoints/

Useful options::

    --force                 overwrite existing files (e.g. after a corrupt download)
    --only straight_ckm ... download only selected checkpoint ids
    --dry-run               print the download/placement plan without writing
    --from-dir DIR          offline mode: copy assets from a local directory that
                            contains files named like the Release assets (this is
                            how the maintainers verify the mapping before upload)

Every file is verified against the SHA256 recorded in ``checkpoints/MANIFEST.json``
before it is moved into place; any mismatch aborts with a non-zero exit code and
leaves the destination untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Release identity -- filled in by the release maintainers.
# The repository and tag below are the canonical KAP-Flow checkpoint release.
# If a different tag is cut later, pass --tag on the command line.
# ---------------------------------------------------------------------------
REPO = "LRXuanxuan/kap-flow-bridging-pilot-observations-and-propagation-knowledge-for-mimo-channel-estimation"
TAG = "v1.0-checkpoints"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "checkpoints" / "MANIFEST.json"
DEFAULT_DEST = REPO_ROOT / "checkpoints"

#: ``target_relative_path`` values are repository-root relative and start with
#: this prefix; ``--dest`` designates the directory the prefix points at.
TARGET_PREFIX = "checkpoints/"
CHUNK = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "checkpoints" not in payload:
        raise ValueError(f"{path} is not a KAP-Flow checkpoint manifest")
    return payload


def resolve_target(target_relative_path: str, dest: Path) -> Path:
    """Map ``checkpoints/main/...`` onto ``<dest>/main/...``."""

    if not target_relative_path.startswith(TARGET_PREFIX):
        raise ValueError(
            f"unexpected target_relative_path {target_relative_path!r}; "
            f"expected a {TARGET_PREFIX!r} prefix"
        )
    return dest / target_relative_path[len(TARGET_PREFIX) :]


def plan(
    manifest: dict, *, only: set[str] | None
) -> tuple[list[dict], dict[str, dict]]:
    """Return the ordered list of entries to install and the required assets."""

    entries = manifest["checkpoints"]
    if only:
        known = {entry["id"] for entry in entries}
        unknown = sorted(only - known)
        if unknown:
            raise SystemExit(
                f"unknown checkpoint id(s): {', '.join(unknown)}\n"
                f"known ids: {', '.join(sorted(known))}"
            )
        entries = [entry for entry in entries if entry["id"] in only]
    assets = {
        asset["name"]: asset
        for asset in manifest.get("release_assets", [])
        if asset.get("kind") in {"checkpoint", "archive"}
    }
    needed: dict[str, dict] = {}
    for entry in entries:
        asset = assets.get(entry["release_asset_name"])
        if asset is None:
            raise ValueError(
                f"manifest lists asset {entry['release_asset_name']!r} for "
                f"{entry['id']!r} but no release asset metadata was found"
            )
        holder = needed.setdefault(asset["name"], {**asset, "entries": []})
        holder["entries"].append(entry)
    # Stable order: weights first, archives after.
    ordered = sorted(needed.values(), key=lambda item: (item["kind"] != "checkpoint", item["name"]))
    return entries, ordered


def resolve_asset_source(
    name: str, *, repo: str, tag: str, from_dir: Path | None
) -> tuple[str, object]:
    if from_dir is not None:
        candidate = from_dir / name
        if not candidate.is_file():
            raise FileNotFoundError(f"{candidate} not found (--from-dir mode)")
        return f"file://{candidate.as_uri()}", candidate
    return f"https://github.com/{repo}/releases/download/{tag}/{name}", None


def fetch_asset(
    name: str,
    sha256: str,
    *,
    repo: str,
    tag: str,
    from_dir: Path | None,
    timeout: float,
    destination_dir: Path,
) -> Path:
    """Download/copy one asset into ``destination_dir`` and verify its SHA256."""

    url, local = resolve_asset_source(name, repo=repo, tag=tag, from_dir=from_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=destination_dir)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        if local is not None:
            print(f"  copy {local}")
            shutil.copyfile(local, temporary_path)
        else:
            print(f"  GET  {url}")
            request = urllib.request.Request(url, headers={"User-Agent": "kap-flow-checkpoints"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                expected = response.headers.get("Content-Length")
                expected_text = f" ({int(expected) / 1e6:.1f} MB)" if expected else ""
                print(f"       {response.status} {response.reason}{expected_text}")
                with temporary_path.open("wb") as output:
                    shutil.copyfileobj(response, output, CHUNK)
        actual = sha256_file(temporary_path)
        if actual != sha256:
            raise ValueError(
                f"SHA256 mismatch for {name}:\n  expected {sha256}\n  actual   {actual}\n"
                "The download was aborted; nothing was installed."
            )
        final = destination_dir / name
        os.replace(temporary_path, final)
        return final
    except urllib.error.HTTPError as error:
        raise SystemExit(
            f"HTTP {error.code} for {url}\n"
            "Check --repo/--tag and that the Release assets were uploaded. "
            "If you are offline, use --from-dir with a local asset directory."
        ) from error
    except urllib.error.URLError as error:
        raise SystemExit(
            f"could not reach {url}: {error.reason}\n"
            "If you are offline, use --from-dir with a local asset directory."
        ) from error
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def install_from_archive(
    archive_path: Path,
    asset: dict,
    *,
    dest: Path,
    force: bool,
    dry_run: bool,
) -> list[Path]:
    installed: list[Path] = []
    with zipfile.ZipFile(archive_path) as bundle:
        for entry in asset["entries"]:
            member = entry.get("release_asset_member")
            if not member:
                raise ValueError(f"{entry['id']}: archive entry has no release_asset_member")
            expected = entry["sha256_inference"]
            target = resolve_target(entry["target_relative_path"], dest)
            try:
                data = bundle.read(member)
            except KeyError as error:
                raise ValueError(
                    f"{archive_path.name} does not contain {member!r} "
                    f"(needed for {entry['id']})"
                ) from error
            actual = sha256_bytes(data)
            if actual != expected:
                raise ValueError(
                    f"SHA256 mismatch inside {archive_path.name} for {entry['id']}:\n"
                    f"  expected {expected}\n  actual   {actual}"
                )
            if dry_run:
                print(f"  [dry-run] unpack {archive_path.name}:{member} -> {target}")
                continue
            if target.exists() and not force and sha256_file(target) == expected:
                print(f"  keep {target} (already verified)")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            try:
                temporary.write_bytes(data)
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            installed.append(target)
            print(f"  install {target}")
    return installed


def install_checkpoint(
    asset_path: Path,
    asset: dict,
    *,
    dest: Path,
    force: bool,
    dry_run: bool,
) -> list[Path]:
    installed: list[Path] = []
    for entry in asset["entries"]:
        expected = entry["sha256_inference"]
        target = resolve_target(entry["target_relative_path"], dest)
        if dry_run:
            print(f"  [dry-run] {asset_path.name} -> {target}")
            continue
        if target.exists() and not force:
            if sha256_file(target) == expected:
                print(f"  keep {target} (already verified)")
                continue
            raise SystemExit(
                f"{target} exists but its SHA256 does not match the manifest.\n"
                "Pass --force to replace it."
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(asset_path, target)
        installed.append(target)
        print(f"  install {target}")
    return installed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the KAP-Flow checkpoints from the GitHub Release and place "
            "them under checkpoints/ (SHA256-verified)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo", default=REPO, help="GitHub owner/name")
    parser.add_argument("--tag", default=TAG, help="Release tag")
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=(
            "local directory that corresponds to the repository's checkpoints/ "
            "directory"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="path to checkpoints/MANIFEST.json",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="ID",
        default=None,
        help="install only the given checkpoint ids (see MANIFEST.json)",
    )
    parser.add_argument(
        "--from-dir",
        type=Path,
        default=None,
        help=(
            "offline mode: copy assets from this local directory instead of "
            "downloading; file names must equal the Release asset names"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="show the plan without writing files"
    )
    parser.add_argument(
        "--timeout", type=float, default=300.0, help="per-request timeout in seconds"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest.expanduser())
    dest = args.dest.expanduser()
    only = set(args.only) if args.only else None
    entries, assets = plan(manifest, only=only)
    acyclic_names = ", ".join(asset["name"] for asset in assets)
    print(f"KAP-Flow checkpoints: {len(entries)} file(s) from {len(assets)} asset(s)")
    print(f"  repository : {args.repo}")
    print(f"  tag        : {args.tag}")
    print(f"  destination: {dest}")
    print(f"  assets     : {acyclic_names}")
    if args.dry_run:
        print("Dry run: nothing will be written.")
    else:
        dest.mkdir(parents=True, exist_ok=True)

    staging = dest / ".download"
    installed: list[Path] = []
    for asset in assets:
        print(f"\n[{asset['kind']}] {asset['name']} ({asset['size_bytes'] / 1e6:.2f} MB)")
        if args.dry_run:
            for entry in asset["entries"]:
                target = resolve_target(entry["target_relative_path"], dest)
                print(f"  [dry-run] {asset['name']} -> {target}")
            continue
        try:
            asset_path = fetch_asset(
                asset["name"],
                asset["sha256"],
                repo=args.repo,
                tag=args.tag,
                from_dir=args.from_dir,
                timeout=args.timeout,
                destination_dir=staging,
            )
            if asset["kind"] == "archive":
                installed += install_from_archive(
                    asset_path, asset, dest=dest, force=args.force, dry_run=False
                )
                asset_path.unlink(missing_ok=True)
            else:
                installed += install_checkpoint(
                    asset_path, asset, dest=dest, force=args.force, dry_run=False
                )
        except (ValueError, FileNotFoundError, zipfile.BadZipFile) as error:
            print(f"\nERROR: {error}", file=sys.stderr)
            raise SystemExit(1) from error
    if staging.exists() and not any(staging.iterdir()):
        staging.rmdir()

    if args.dry_run:
        print("\nDry run complete; no files were written.")
        return 0

    # Final verification pass over everything the manifest declares.
    print("\nVerifying installed files against MANIFEST.json")
    failures: list[str] = []
    for entry in entries:
        target = resolve_target(entry["target_relative_path"], dest)
        if not target.is_file():
            failures.append(f"{target}: missing")
            continue
        actual = sha256_file(target)
        if actual != entry["sha256_inference"]:
            failures.append(
                f"{target}: sha256 {actual} != {entry['sha256_inference']}"
            )
        else:
            print(f"  OK {target}")
    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"\nDone: {len(entries)} file(s) verified under {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
