#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
    printf 'Usage: bash with_nas_storage.sh COMMAND [ARG ...]\n' >&2
    exit 64
fi

export SSLCL_STORAGE_ROOT="${SSLCL_STORAGE_ROOT:-/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg}"
# pro5000 uses the explicitly authorized /data volume; other hosts require NFS.
[[ "$SSLCL_STORAGE_ROOT" = /* ]] || exit 78
case "$(findmnt -rn -T "$SSLCL_STORAGE_ROOT" -o FSTYPE)" in
    nfs|nfs4) ;;
    ext4) [[ "$SSLCL_STORAGE_ROOT" == /data/* && "$(findmnt -rn -T "$SSLCL_STORAGE_ROOT" -o TARGET)" == /data ]] || exit 78 ;;
    *) printf 'The experiment storage root is not on NFS; stopping.\n' >&2; exit 78 ;;
esac

export LCRSEG_RUN_ROOT="$SSLCL_STORAGE_ROOT/runs"
export TMPDIR="$SSLCL_STORAGE_ROOT/tmp"
export XDG_CACHE_HOME="$SSLCL_STORAGE_ROOT/cache/xdg"
export TORCH_HOME="$SSLCL_STORAGE_ROOT/cache/torch"
export HF_HOME="$SSLCL_STORAGE_ROOT/cache/huggingface"
export TRITON_CACHE_DIR="$SSLCL_STORAGE_ROOT/cache/triton"
export CUDA_CACHE_PATH="$SSLCL_STORAGE_ROOT/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$LCRSEG_RUN_ROOT" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" \
    "$HF_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

# A real write probe is necessary: NFS ACCESS is unreliable on this mount.
probe="$(mktemp "$TMPDIR/.sslcl_write_probe.XXXXXX")"
trap 'rm -f -- "$probe"' EXIT
printf 'NAS_WRITE_OK\n' > "$probe"
[[ "$(cat -- "$probe")" == NAS_WRITE_OK ]]
rm -- "$probe"
trap - EXIT
exec "$@"
