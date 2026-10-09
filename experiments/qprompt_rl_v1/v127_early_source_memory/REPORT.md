# V127 early source-memory supervision

RUNNING_NOT_RESULTS. Started2026-10-09 13:03:32UTC /21:03:32China time. Run `v127_early_20261009T130231Z`. Preregistration and execution commit `f8f8b258efb5a8e03248d1d594c5677591c267ff` were pushed and anonymously accessible before launch.

V126 found SOURCE→100 fit Dice+6.509pp, held−5.713pp. This run moves the existing mixed frozen-memory/EMA target to step0 and compares a matched selected-EMA prefix; both use identical mixed-target continuation after100. Cached BASE and frozen SOURCE/ENTRY controls retain the prior negative evidence. All16contexts×2streams remain.

At the startup check13:04:57UTC, qualification passed108 updates, including exact historical100-step full-state reproduction and exact replay of both new prefixes. Four main workers were active on admitted GPUs4–7, with455 main updates already logged successfully at the scheduler snapshot; no failures. Full coordinator/worker argv and nvidia-smi names were neutral. NAS NFS/write/read probe passed with more than13TiB free; each admitted GPU had24121MiB free at launch.

Fixed cap19308 native updates (108qualification+19200main),64new trajectories/192sealed snapshots,384performance calls1536images,0actor/solve/new annotation/Q_dev. Historical and frozen controls are cached, not recomputed. Seal all snapshots before readout. Absolute improvement must be relative to SOURCE, with matched controls and all fold/stream gates; less degradation is not positive learning. See PREREG.md for exact definitions.

Expected wall time roughly30minutes, workload estimate only. The existing hourly monitor follows this run; no new automation or continuous session wait. Completion requires terminal audit and publication of all positive/negative results. Current files are a startup delivery, not a completed experiment or demonstrated gain.

Private states/images/IDs/raw logs remain on NAS. No independent confirmation, later-domain CL or RL-effectiveness claim.
