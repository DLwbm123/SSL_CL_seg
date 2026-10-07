# V8.2 final: aligned episodes did not improve the prior

All 128 policy groups and 20 development endpoints completed and sealed before evaluation. Outcome: STOP_EPISODE_ALIGNMENT_NO_PRACTICAL_GAIN. C/D did not run. This was a reused first-domain diagnostic, not independent generalization evidence.

| Method | New Dice (%) | Old Dice (%) | Utility ×100 |
|---|---:|---:|---:|
| NATIVE | 73.4741 | 81.2819 | -7.3256 |
| FIXED_BEST | 73.7624 | 81.0925 | -7.2268 |
| UNIFORM_ACTION | 73.7753 | 81.2891 | -7.0331 |
| PRE_FROZEN_601 | 73.5989 | 81.2731 | -7.2396 |
| PRE_FROZEN_602 | 73.6995 | 81.2820 | -7.1554 |

## Evidence and conclusion

Fresh native horizon-300 entries and decision progress 1/3 and 2/3 passed runtime checks. Both priors completed 64 groups; actor updates were 152 and 136. The intended episode alignment was active. Nevertheless, prior 601 changed new/old Dice versus uniform by -0.1763/-0.0159 pp, and prior 602 by -0.0758/-0.0070 pp. Both mean utilities were worse than uniform and matching V8.1 priors. No floor or checkpoint was retuned. Retire the episode-alignment intervention as a demonstrated improvement.

Native/fixed/uniform scores remain identical across the three studies, consistent with preserved matched control paths. All methods still lose about five absolute percentage points on the frozen old query reference; small relative differences do not prevent forgetting. Shared source, tiny query sets, simulated photometric tasks and previously observed development endpoints limit all conclusions.

## Next bounded question

Two targeted GRPO adjustments failed. A distinct dense reward-supervision diagnostic will use the complete nine-action reward vector at both native decision states (100/200) with two paired streams, and train the same small policy directly on fixed soft preferences. This tests whether complete action information can train useful selection when sampled GRPO does not. It is not another GRPO success claim, cannot establish independent generalization, and must beat random/fixed/native controls under separately frozen gates. No target training follows automatically.

## Cost

New V8.2 student invocations: 56,408; actor invocations: 292 (four qualification, 288 prior). Across V8/V8.1/V8.2: 190,845 student = 45 qualification + 8,000 auxiliary + 2,800 entries + 14,400 audit + 153,600 prior + 12,000 development, and 704 actor = 16 qualification + 688 prior. Historical common source adds 8,000 student separately. All optimizer attempts succeeded; all consumed earlier qualification-repair costs remain included.

All anonymous group trajectories, 20 endpoint/channel outcomes and cumulative ledger are public; raw images, role mappings, labels, weights and private snapshots stay on NAS.
