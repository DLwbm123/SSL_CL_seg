# Gate1C v3 destination admission

The newly user-authorized SSH endpoint resolves to `zmic44`, uid/euid 1006, port 22. Exact endpoint and other-task process inventory remain in the hashed private audit. This is a new admission record; the earlier public-key-only blocked snapshot is preserved.

A separate owned writable root was created at `/home/jiangsuiyang/SSL_CL/gate1c_v3_clean_regeneration_20260831`. Existing source-only code, data, processes, permissions and mounts were not changed. Only physical GPUs 5, 6, and 7 may be shared; no process was killed and no GPU was reserved. The host has eight 24-GiB RTX 3090 cards.

The existing `py38` interpreter is Python 3.10.6 / Torch 2.2.1+cu121. Official JASCL is clean and pinned to `3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53`. All 2962 frozen files (1,850,621,123 bytes) match the original checksums, and all three manifest/split pairs and leakage audits pass.

The owned ext4 filesystem had 119,633,264,640 bytes available and 231,515,060 free inodes at audit. It is 97% used; quota could not be queried because the quota command is absent. The NAS candidate is not writable. Free space must be rechecked before execution, with a separate later diagnostic storage estimate. No package installation or permission change is authorized.
