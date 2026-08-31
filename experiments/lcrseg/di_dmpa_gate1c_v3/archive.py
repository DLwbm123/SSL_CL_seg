"""Promote a fully verified private incoming bundle on the same filesystem."""
import argparse
import json
import os
from pathlib import Path

from .durable import now, sha256, verify, write_new


def promote(incoming, archive_parent, expected_manifest_sha256):
    incoming, archive_parent = Path(incoming), Path(archive_parent)
    if incoming.is_symlink() or archive_parent.is_symlink():
        raise RuntimeError("archive roots must not be symlinks")
    manifest_path = incoming / "PRIVATE_BUNDLE_MANIFEST.json"
    if sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("remote/local manifest hash mismatch")
    manifest = verify(incoming)
    archive_parent.mkdir(parents=True, exist_ok=True)
    destination = archive_parent / manifest["content_sha256"]
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if incoming.stat().st_dev != archive_parent.stat().st_dev:
        raise RuntimeError("atomic archive promotion requires the same filesystem")
    if sha256(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("manifest changed during verification")
    # One owner controls this private archive; no concurrent archive writers.
    os.rename(incoming, destination)
    directory = os.open(archive_parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    receipt = dict(status="PASS_PRIVATE_ARCHIVE", verified_at=now(), archive=str(destination),
                   manifest_sha256=expected_manifest_sha256, content_sha256=manifest["content_sha256"],
                   files=manifest["files"], bytes=manifest["bytes"], every_byte_and_sha_verified=True,
                   atomic_promotion=True, remote_copy_deleted=False)
    write_new(archive_parent / (manifest["content_sha256"] + ".audit.json"), receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", required=True, type=Path)
    parser.add_argument("--archive-parent", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(promote(args.incoming, args.archive_parent, args.expected_manifest_sha256), sort_keys=True))
