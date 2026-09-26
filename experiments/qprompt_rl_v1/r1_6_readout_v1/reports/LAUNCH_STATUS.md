# R1.6 launch status

Observed 2026-09-26 14:27:34 UTC. **Running, not completed.**

E0: `1a0b2468c80f32005dc4fec7f4888dc87e9d0aa8`; active training: `e94413b499bcf6b348c841d80fbde93263b4c4ef`. The separate closeout helper and launch report do not change running training source.

All four historical prefixes passed state/provenance checks. CPU mathematical, writer/checkpoint and lease/accounting regressions passed. Native qualification used 18 synthetic attempts, including two retained failed-qualification attempts; 16 real-L smoke updates and 64 zero-update forensic batches completed. DINO numerical update comparison maximum absolute discrepancy was below 4.9e-9; restoration was exact, bitwise training reproducibility is not claimed.

Four formal workers made real optimizer progress, with over 3,000 committed updates in this live snapshot. One endpoint had reached step3000; endpoint evaluation and full results remain pending. No formal replay or training exception had occurred. See STARTUP_STATUS.json for the sampled checkpoint/commit positions.

The registered matrix remains 56 tasks, 48 endpoints and 64,000 valid updates. Training deadline: 2026-09-27 02:19:58 UTC. Monitoring is every 30 minutes with a bounded maintenance deadline. No R2/R3. Results and final reports will be published after execution closes.
