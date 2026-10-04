# V5_QUICK4H — user-authorized exploratory screen

## Scope and time

The user explicitly requested reducing 6 groups ×4 full trajectories and keeping this experiment within four hours to see initial results. This supersedes the prior unlimited-time V5 matrix. It does not silently redefine that original preregistration as completed.

Original session start is retained: 2026-10-04 18:12:57 CST. Hard cutoff is **22:12:57 CST**. Student/optimizer scheduling stops by **21:57:57 CST**, reserving 15 minutes for independent evaluation and reports. No clock or consumed-cost reset. At cutoff, retain checkpoints/receipts, mark incomplete cells uncompleted, and report only full-horizon endpoints actually obtained. No automatic continuation.

## Frozen reduced matrix

- Source seed168, controller seed401, both target domains, same REFUGE entry.
- GRPO_FS and PPO_MATCHED only, each domain **1 group ×2 independent full trajectories**. Native horizons3200/2100, block25, actions0/.125/.5 unchanged. Each group has4 epochs×4 minibatches=16 actor steps. PPO critic remains separate; FS has none.
- Same reward-free softened V4 OFFLINE_25 initialization, fixed historical normalization and pure-softmax sampling. Existing initialization and 690-call native qualification are reused without repeating them.
- Keep the already-running seed168 calibration:4 initial-policy full trajectories plus1 U0 per domain. Its G4 calibration group determines the same fixed-scale formula, now from one source seed instead of three. No selection by outcomes. Stop the source169 calibration; its partial work remains charged and its checkpoints remain private.
- Compute one fresh complete-state U0 reference/domain with exact entry/provenance validation. No unverified cache sharing across the source revision.
- Frozen argmax endpoint deployment for ORIGINAL, OFFLINE_25, GRPO_FS and PPO_MATCHED: **8 complete endpoints**, full current L/U as in V4. Baselines can train on free GPUs while calibration proceeds; no val result is exposed to learners.
- All completed endpoint models are frozen and all training is finished or stopped before any independent val evaluator runs. No shortened trajectory is counted as an endpoint.
- Remove GRPO_STD, Bandit, reward shuffle, phase shuffle, FINE_05, Clip-Higher and automatic extension/confirmation from this screen. No original pilot promotion rule is applied to this reduced matrix.

## Budget

| Stage | Student updates |
|---|---:|
| Retained source168 calibration |26500|
| Two U0 references |5300|
| Two learners, two domains, one G2 group |21200|
| Eight full endpoints |21200|
| Performance total |74200|
| Completed native qualification |690|

The source169 work stopped by the user's amendment is additional already-spent cost, recorded from its attempt/success ledger; it is not erased or hidden. Previous synthetic qualification65900 optimizer calls plus one new16-call G2 qualification are separate from student updates. Previous failed pre-student setup and its65884 synthetic calls are included in that accounting.

## Scheduling and storage

Only physical GPUs5/6/7, preferably free GPUs6/7, without preemption of others. Adopt the two existing source168 calibration processes without restarting them. Stop the old coordinator so it cannot launch the superseded full matrix. The new finite dependency scheduler uses ready baseline/reference/learning jobs to fill idle GPUs. Every new process uses the NAS wrapper and neutral argv; all outputs/checkpoints/logs/source snapshots remain on NAS. Retain the original archive and revised source SHA separately. Storage reserve20GiB for this screen, within the prior80GiB allocation.

The existing calibration was compiled from the original source; its scientific implementation is unchanged. New full-state references and both learners use the new revision. All relevant source hashes are recorded, and native engine files are unchanged. G4 controller behavior remains the default; G2 is enabled only by an explicit group_size argument and receives a16-call CPU-only check before launch. No new native qualification updates are required for the unchanged student engine.

## Interpretation limits

This yields one-seed initial paired directions for GRPO_FS versus supervision, offline policy and matched PPO. One G2 group is weak controller training; a small or negative difference does not establish that RL cannot work. No sample SD over seeds, stable advantage, independent-controller replication, Bandit/shuffle advantage, or multistep-credit evidence can be claimed. Val patients are reused historical development patients; test and hidden U labels remain sealed. Q/probability changes are not Dice gains. Original KI identity remains unverified. Only source and anonymous aggregates are public; private raw logs, features, roles, checkpoints and case scores stay on NAS.
