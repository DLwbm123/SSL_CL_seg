# V111 —13-action matched sequential startup

RUNNING, not a result. Uses the sealed V110120-state13-action training table, with explicit equal state weights and unchanged first9-action advantage baseline. Adds true supervised-only OFF to the matched WARM/CE/RL/targetCE comparison. Allcontrols are retained, including separate GLOBAL, TIME and fixedOFF. See PREREG for exact matrix and criteria.

At startup, all8 actors completed 4,097 ledgered optimizer updates (4,096data +1synthetic); one training ridge solve passed its residual check. The13-action analytical-target grouping/KL/finite-decreasing-loss qualification passed. Models were sealed before deployment. Native qualification completed8updates: two exact paired continuations (originalaction9 andOFF12), two archived stable-state matches, and policy/probe snapshot/RNG isolation. OFF updates explicitly checked zeroUreads.

Four workers on GPU4–7 each advanced50native updates toMID150 at the startup check. Fit and qualification exited0; no failures were observed. Full process command lines and GPU process displays were neutral; run identities and NAS mount/write/read/free-memory admission passed. Shared engine defaults reproduce the old V108240-row decision exactly; archived NAS code was not modified.

Fixed complete run:22policies×4contexts×3streams=264actual trajectories,528snapshots allsealed before newQ_devreadout;52,808native updates,4,097actor updates,1ridge,2,128stableprobes and3,168Q_devimages. No newQ_trainqueries or annotations. Historical paired V111−V108 results will be reported for all240 common method/context/stream keys using saved results, without additional queries.

Only the whole fixed result can determine candidate status. Primary is sampledRL against both matchedactor controls and allsimplecontrols; argmax cannot rescue a failedprimary. Development-informed factors and shared patients do not support independentgeneralization. A passing candidate still requires separately frozen fresh-stream confirmation. Private features/weights/snapshots/rawlogs remain onNAS.
