# SSL_CL_seg storage policy

- The user requires this project's experiment outputs and future large files to live on NAS (2026-08-31).
- Canonical server storage: `/data_nas/jiangsuiyang/LCR-Seg/SSL_CL_seg`. Use a new, create-only directory for each future protocol/run.
- Launch future experiment commands through `bash experiments/lcrseg/scripts/with_nas_storage.sh COMMAND ...`. Explicit output/cache/scratch arguments must also resolve to NAS. Do not silently fall back to `/home` or the local checkout when NAS is unavailable.
- Store checkpoints, banks, validation caches, raw arrays, generated datasets, run logs, temporary files, and model downloads on NAS. Keep only source code and small operational metadata on home. Do not put private artifacts in Git.
- Existing `/home/jiangsuiyang/SSL_CL/runs` and the Gate1C v3 result root are compatibility symlinks after migration. Preserve these links; do not replace them with home directories.
- Verify the actual NAS mount and a create/write/read probe before large writes. On this server, `os.access(..., os.W_OK)` can report false even when real NFS writes succeed; do not use it alone as an admission gate.
- Relocation must preserve hardlinks, file bytes, historical manifests, and old path access. Verify the full copy before removing any home duplicate. Never overwrite an existing NAS archive.
- Existing frozen HDF5/manifests/splits/checksums, environments, unrelated projects, and local verified archives are protected unless the user separately authorizes changing them.
- Storage maintenance does not authorize new experiments or lift any scientific hard stop. Continue to follow `experiments/lcrseg/AGENTS.md` and the applicable frozen protocol.
