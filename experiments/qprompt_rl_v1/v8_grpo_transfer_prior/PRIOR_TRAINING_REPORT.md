# First-domain GRPO prior — running

Stage A passed exactly 4/8 contexts. Both controller seeds 601 and 602 have started after stage-B qualification passed (8 additional real student updates and 4 actor qualification updates). Stage A/repair costs are carried forward, not reset.

Each prior has 64 groups, four 100-step candidates per group, paired full-state entry restoration and branch-zero continuation. Low-signal groups are skipped without replacement. The frozen sigma floor is 0.002027520245732116. Actor updates are at most four full-group epochs per group. Both final policies are retained.

The registered development stage runs 20 endpoints (four fixed contexts × five methods, 200 updates each), plus 400 shared entry updates. All 20 finish training before Q_dev is read. Development gates and utility are frozen in B_RUN_CONFIG.json. No target-domain endpoint has run.

No GRPO effectiveness conclusion is available yet. Progress is a time-stamped snapshot in B_RUN_STATUS.json, not a live counter.
