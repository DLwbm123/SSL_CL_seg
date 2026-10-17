# Project relocation

The 2026-10-17 user request authorizes copying the project to pro5000's data volume
and running a source-stage development control on physical GPU2. This host-specific
authorization supersedes the earlier NAS-only placement for new work on pro5000.
The original NAS mount and compatibility symlinks remain unchanged.

Completed: current local checkout with Git history, active working tree, and the
entire current my-gpu project directory. All three streamed copies exited 0 on both
ends; destination directories and key manifest/source files were checked once.
Hardlinks and symlinks were carried by tar; no source removal, checksum scan or
bytewise comparison was performed. Destination tar ignored macOS provenance extended
attributes. The copied active run is a live snapshot, not a final experiment receipt;
its terminal outputs must be added after the original run finishes.

The older NAS project archive was copied into a separate create-only directory
by a background rsync process on pro5000, preserving hardlinks. The legacy server
checkout (about 16 GiB, including its original input files) completed its separate
streamed copy into another create-only directory; both ends exited 0 without errors.
The NAS copy exited 23 after about 61.5 minutes, with 177,386,666,979 bytes transferred
and one reported inaccessible historical directory. Full-project migration is not yet complete. Training uses the existing
verified data and environment while historical relocation proceeds.

During the NAS copy, one historical stage directory rejected access with permission
denied. Its mode is 000 when inspected from both hosts. The copy has ended;
source permissions remain unchanged. This outstanding access gap
must be resolved before claiming a complete historical relocation.

Large artifacts, environments, cache and logs stay on the data volume. Private
paths and operational receipts live in ignored/local records, not this report.
