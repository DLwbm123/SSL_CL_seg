# First-domain priors: complete, gate failed

Both seeds completed 64 groups and 25,600 candidate student updates. Actor 601 made 72 optimizer updates and skipped 46 groups; actor 602 made 64 updates and skipped 48 groups. Both final checkpoints are retained on NAS. All group rewards, states, probabilities and diagnostics are published in B_PRIOR_601_GROUPS.jsonl and B_PRIOR_602_GROUPS.jsonl.

All 20 development endpoints were sealed before readout. Both priors have lower mean utility than the matched uniform controller. See B_FINAL_REPORT.md and B_DEVELOPMENT_RESULTS.json. No target training or low-label comparison ran.
