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
and one reported inaccessible historical directory. That gap is now resolved as
described below. Training uses the existing
verified data and environment while historical relocation proceeds.

During the NAS copy, one historical stage directory rejected access with permission
denied. Its mode is 000 when inspected from both hosts. The copy has ended;
source permissions initially remained unchanged. The user subsequently authorized
temporarily granting owner read/execute access and restoring mode 000 afterward.
The first repair exited 3 because the copied destination also had mode 000. After
temporarily granting destination owner access, the focused repair exited 0 with no
errors and confirmed that the directory is empty. Source and destination modes
were both restored to 000; both repair receipts remain preserved.

All requested historical copies are now complete according to transfer exit
receipts and focused checks, including this repaired gap. The completed L-only run
was also copied separately after its evaluation recovery finished, with both stream
ends exiting 0 and its terminal status readable. Original sources remain intact.
No bytewise equivalence claim is made.

Large artifacts, environments, cache and logs stay on the data volume. Private
paths and operational receipts live in ignored/local records, not this report.
