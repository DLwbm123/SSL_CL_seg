# Destination admission: blocked

Status: **BLOCKED_DESTINATION_IDENTITY_UNRESOLVED**. No authenticated remote
shell, deployment, supervisor, training, or diagnostic was started.

The local OpenSSH configuration has no usable `jiangsuiyang` alias. The
handoff repository README supplies a candidate IP, user `jiangsuiyang`, and
port 22; the IP also has a trusted local known-host key. A noninteractive
read-only identity probe reached SSH authentication and returned
`Permission denied (publickey,password)`. The SSH agent lists no identities.
Neither the destination hostname/uid nor any writable root has been verified.

The SSH client's 255 is a connection/authentication observation. It is not an
experiment exit code and says nothing about the old v2.2 full result.

Every requested server observation remains unobserved: processes, GPUs,
driver/CUDA, packages, mounts, bytes/inodes/quota, project/run/data directories,
Git checkouts, JASCL, and frozen data integrity. Historical `/home/jiangsuiyang/SSL_CL`
paths and hostname are candidates only. No other account/host was substituted,
no password was read, and strict host-key checking remained enabled.

Exact host/user/port, fingerprints, probe argv/stdout/stderr and private receipt
hashes are retained in the local private preflight bundle. The public JSON
redacts the address only. The user's section-4 hard stop applies; an authenticated
connection is required before any destination audit or write can continue.

The [local data audit](DATA_AND_RUNTIME_AUDIT.json) is useful preparation, but
cannot satisfy destination admission. [Machine-readable record](DESTINATION_AUDIT.json).
