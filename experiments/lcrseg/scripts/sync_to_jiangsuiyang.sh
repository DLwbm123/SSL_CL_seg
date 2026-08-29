#!/usr/bin/env bash
# Data-only LCR-Seg synchronizer. A dry run is the default.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
local_root="/Volumes/DataP/LCRSeg"
remote_host="jiangsuiyang"
remote_user=""
ssh_port="22"
bind_address=""
remote_root=""
execute=0
python_bin="${LCRSEG_PYTHON:-/opt/miniconda3/bin/python}"
remote_python=""

usage() {
  cat <<'EOF'
Usage: sync_to_jiangsuiyang.sh --remote-root /absolute/existing/path [options]

Options:
  --local-root PATH       Derived-data root (default: /Volumes/DataP/LCRSeg)
  --remote-host HOST      SSH host alias or IP (default: jiangsuiyang)
  --ssh-user USER         Optional remote SSH user
  --ssh-port PORT         SSH port (default: 22)
  --bind-address IPV4     Optional local source address, for example 10.75.81.150
  --remote-root PATH      Required existing absolute writable remote directory
  --remote-python PATH    Required with --execute; remote Python with h5py/numpy
  --execute               Perform rsync after a successful dry-run style preflight
  --python PATH           Local Python used for bundle validation
  -h, --help              Show this help

Only h5/, manifests/, splits/, checksums/, and reports/preprocessing/ are sent.
Raw source data and deletion semantics are never used.
EOF
}

while (($#)); do
  case "$1" in
    --local-root) local_root="$2"; shift 2 ;;
    --remote-host) remote_host="$2"; shift 2 ;;
    --ssh-user) remote_user="$2"; shift 2 ;;
    --ssh-port) ssh_port="$2"; shift 2 ;;
    --bind-address) bind_address="$2"; shift 2 ;;
    --remote-root) remote_root="$2"; shift 2 ;;
    --remote-python) remote_python="$2"; shift 2 ;;
    --python) python_bin="$2"; shift 2 ;;
    --execute) execute=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$remote_root" || "$remote_root" != /* || "$remote_root" == "/" || "$remote_root" == *$'\n'* || "$remote_root" == *".."* ]]; then
  printf '%s\n' 'ERROR: --remote-root must be a non-root, absolute, normalized path.' >&2
  exit 2
fi
if ! [[ "$remote_host" =~ ^[A-Za-z0-9._-]+$ ]] || ! [[ "$remote_user" =~ ^[A-Za-z0-9._-]*$ ]] || ! [[ "$ssh_port" =~ ^[1-9][0-9]{0,4}$ ]] || ! [[ "$bind_address" =~ ^[0-9.]*$ ]]; then
  printf '%s\n' 'ERROR: invalid SSH host, user, port, or bind address.' >&2
  exit 2
fi
if [[ ! -d "$local_root" ]]; then
  printf 'ERROR: local root does not exist: %s\n' "$local_root" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  printf 'ERROR: local Python is not executable: %s\n' "$python_bin" >&2
  exit 2
fi
if ((execute == 1)) && [[ -z "$remote_python" || "$remote_python" != /* ]]; then
  printf '%s\n' 'ERROR: --execute requires an explicit absolute --remote-python with h5py and numpy.' >&2
  exit 2
fi

"$python_bin" "$script_dir/verify_local_bundle.py" --root "$local_root"

remote_root_quoted="$(printf '%q' "$remote_root")"
remote_target="${remote_user:+${remote_user}@}${remote_host}"
ssh_args=(-p "$ssh_port")
if [[ -n "$bind_address" ]]; then
  ssh_args+=(-4 -b "$bind_address")
fi
ssh "${ssh_args[@]}" "$remote_target" "test -d $remote_root_quoted && test -w $remote_root_quoted"

sources=()
for relative in h5 manifests splits checksums; do
  if [[ ! -d "$local_root/$relative" ]]; then
    printf 'ERROR: required bundle directory is missing: %s\n' "$local_root/$relative" >&2
    exit 2
  fi
  sources+=("$local_root/$relative")
done
reports_source=""
if [[ -d "$local_root/reports/preprocessing" ]]; then
  reports_source="$local_root/./reports/preprocessing"
fi

rsync_args=(
  -a
  --checksum
  --itemize-changes
  --exclude='*.tmp'
  --exclude='._*'
  --exclude='.DS_Store'
)
if ((execute == 0)); then
  rsync_args+=(--dry-run)
fi
rsync_ssh=(ssh -p "$ssh_port")
if [[ -n "$bind_address" ]]; then
  rsync_ssh+=(-4 -b "$bind_address")
fi
rsync_remote_shell="$(printf '%q ' "${rsync_ssh[@]}")"
rsync "${rsync_args[@]}" -e "$rsync_remote_shell" "${sources[@]}" "$remote_target:$remote_root/"
if [[ -n "$reports_source" ]]; then
  # Preserve reports/preprocessing rather than flattening it to preprocessing/.
  rsync "${rsync_args[@]}" --relative -e "$rsync_remote_shell" "$reports_source" "$remote_target:$remote_root/"
fi

if ((execute == 0)); then
  printf '%s\n' 'Dry-run completed. Re-run with --execute only after reviewing the itemized transfer.'
  exit 0
fi
remote_verify="set -e; cd $remote_root_quoted; if command -v sha256sum >/dev/null 2>&1; then sha256sum -c checksums/checksums.sha256; else shasum -a 256 -c checksums/checksums.sha256; fi"
ssh "${ssh_args[@]}" "$remote_target" "$remote_verify"
"$python_bin" "$script_dir/verify_remote_h5.py" --remote-host "$remote_host" --ssh-user "$remote_user" --ssh-port "$ssh_port" --bind-address "$bind_address" --remote-root "$remote_root" --remote-python "$remote_python"
printf '%s\n' 'Transfer and remote SHA-256 verification completed.'
