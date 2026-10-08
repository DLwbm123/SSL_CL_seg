# V100 exact conditional expectation and argmax deployment

**NO_V100_ARGMAX_RL_INCREMENTAL_GAIN.** Completed 2026-10-08T10:55:57.815407+00:00, with zero optimizer updates and zero image queries. Deterministic argmax RL loses to matched CE in both seeds. Exact T1 policy expectation favors RL over CE and expanded initialization, but the magnitude is small and does not establish practical or independent improvement. The originally sampled V99 outcomes remain unchanged.

| Readout | Policy | New soft Dice (%) | Simulated old soft Dice (%) | Original utility |
|---|---|---:|---:|---:|
| EXACT_T1_EXPECTATION | EXPANDED_RL | 74.107764 | 81.337773 | -0.067081666 |
| EXACT_T1_EXPECTATION | EXPANDED_CE | 74.054055 | 81.274035 | -0.068197345 |
| EXACT_T1_EXPECTATION | EXPANDED_INIT | 74.059991 | 81.303561 | -0.067821829 |
| EXACT_T1_EXPECTATION | UNIFORM_12 | 73.634405 | 81.125680 | -0.073305598 |
| FIXED_ARGMAX | EXPANDED_RL | 74.159156 | 81.372283 | -0.066218143 |
| FIXED_ARGMAX | EXPANDED_CE | 74.189137 | 81.391539 | -0.065719856 |
| FIXED_ARGMAX | EXPANDED_INIT | 74.047213 | 81.322128 | -0.067721836 |
| FIXED_ARGMAX | UNIFORM_12 | 73.474111 | 81.281920 | -0.073255628 |
| ORIGINAL_V99_T1_SAMPLE | EXPANDED_RL | 74.190009 | 81.410084 | -0.065445736 |
| ORIGINAL_V99_T1_SAMPLE | EXPANDED_CE | 74.172202 | 81.383098 | -0.065820113 |
| ORIGINAL_V99_T1_SAMPLE | EXPANDED_INIT | 74.254622 | 81.440875 | -0.064624201 |
| ORIGINAL_V99_T1_SAMPLE | UNIFORM_12 | 73.646964 | 81.094609 | -0.073476563 |

## What the three readouts show

Argmax RL-minus-CE is -0.029981pp new, -0.019257pp old, utility -0.000498287. Paired utility differences are seed601 -0.000966536, seed602 -0.000030039.

Exact T1 expectation RL-minus-EXPANDED_CE: +0.053709pp new, +0.063738pp old, utility +0.001115678.

Exact T1 expectation RL-minus-EXPANDED_INIT: +0.047773pp new, +0.034213pp old, utility +0.000740163.

Expectation integrates p(a100) p(a200 | the native prefix from a100) across all144cached trajectories in each context. Each trajectory's own gain and retention hinge are averaged; the hinge is not applied after averaging old scores. This is exact for the stored probabilities and this finite native stream, not population performance or an independent repetition. The uniform argmax uses action0 due to the fixed lowest-ID tie rule, and is only a mechanical control; the primary uniform comparison uses exact uniform expectation.

The original V99 sampled RL utility is more favorable than its full-policy expectation. Changing the readout changes the apparent ranking against initialization, but neither permits dropping the original negative result. A single categorical rollout is not an estimate with established sampling precision; exact integration removes action-sampling noise for this fixed stream only. Argmax was separately preregistered, and failed rather than being silently substituted for the V99 endpoint.

## Qualification, accounting and limits

All four GPU jobs exited zero; coordinator ended and lock released. All52native state snapshots were unchanged by inference; all48saved actor state/probability vectors and all36original sampled choices reproduced exactly. All312actor probability vectors and all36argmax choices were sealed before metrics were read. All108mode/context/policy rows, conditional weighted expectations, argmax lookup rows, original samples and the verdict independently recomputed. Synthetic checks cover conditional weighting, point mass, uniform mean, tie handling and invalid probabilities.

No student/actor optimizer update, training/development image query, failed update or new image. Unique cached grid rows read576;36weighted policy rows,36argmax rows and36original sampled rows do not become108new replicates. Cumulative campaign remains419,349nativeupdates excluding8,000common-source updates (427,349including),83,149actorupdates. The preregistration's original count72saved actor vectors was corrected to48 before probability extraction; this clerical correction is retained in Git history.

Current training returns still use the old nine-action V90 behavior continuation and its selected step200 states, even though V99 now deploys twelve-action policies. The next separately frozen experiment will refresh training-only returns and reached states under a symmetric ensemble of the current expanded CE/RL policies, keeping CE/RL fitting information and computation matched. This targets continuation/state-distribution mismatch instead of another temperature/readout search. It does not use development maximizing actions as labels.

Repeated D1, shared4old+4new development query images, simulated retention and unestablished patient independence remain. Training roles retain16memory-fit images, support2/8 from8support images and4old+4new Q_train images; no hidden U labels, sealed test or real subsequent domain. No independent validation, GRPO claim or paper success. Allseed/readout/control results are public; state vectors, weights and image/role identities stay private on NAS.
