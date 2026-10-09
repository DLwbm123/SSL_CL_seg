# V116 engineering recovery, authorized 2026-10-09

The first attempt stopped at the first group of both learning seeds. The trajectory list was passed to the existing dictionary checkpoint writer; tensor serialization completed but receipt generation raised AttributeError. The post-update actor/optimizer/private-RNG bundle and group rewards had not been committed. There is no exact saved state from which to continue group 1.

The user explicitly instructed: “你解决一下修复继续跑”. This authorizes one engineering rerun of the two learning jobs with unchanged seeds, recipe, data roles, thresholds and readout. The original failed directory, partial checkpoints, ledgers and exit receipts remain intact. A new create-only recovery directory reuses the completed qualification job and restarts both learning jobs from their original stored entries. No scientific variant or seed selection is introduced.

The fix wraps the trajectory list in a dictionary, matching the existing checkpoint API. A CPU-only regression check exercises the real writer, tensor reload and committed receipt, without model forwards, queries or optimizer calls. Historical checkpoint code is unchanged. No consumer previously loaded the trajectory list.

Original attempt actual costs: 408 native updates (8 qualification + 400 training), 6 actor updates (2 qualification + 2 RL + 2 CE), 112 Q_train images / 28 calls. All physical calls succeeded; this is a persistence failure, not an optimizer failure. No development trajectory or Q_dev readout ran.

The recovery uses the same logical 29608-native/130-actor protocol, with the 8 native/2 actor qualification calls explicitly reused, not executed again. New execution is 29600 native, 128 actor, 4592 query images / 1148 calls. Combined physical campaign totals if it completes are 30008 native, 134 actor, 4704 query images / 1176 calls. Extra failed-attempt cost is 400 native, 4 actor, 112 query images / 28 calls. These costs must remain separate from scientific protocol counts and historical costs before V116. There are still 84 development trajectories, 0 new annotations, 0 solves and 0 stable probes.

Recovery is not a result. Original candidate/absolute gates and development-only limitations remain unchanged. No further automatic engineering retry is authorized by this note.
