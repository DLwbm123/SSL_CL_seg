# F5 projection V2: complete result and failed practical gate

All four trajectories exited successfully. Eight target students were sealed before the 20 seen-domain evaluations. The existing completion audit passed. Formal physical optimizer calls were 21,200; source calls were zero. No retry, coefficient selection, extra evaluation or checkpoint selection occurred. Execution commit: `07b439773c3fe59b5eb8f97b8f2f7b0cc80692e2`.

| Method | Final Dice % | Old % | Incoming % | Forget pp |
|---|---:|---:|---:|---:|
|Matched prior-round F5|65.4109|63.1689|69.8950|13.6757|
|F5 + one-sided R projection|65.5126|63.3436|69.8507|13.5009|
|Historical B0|64.8041|62.1852|70.0418|14.3667|
|Historical B2|67.9011|66.2410|71.2212|10.7007|

Final is the mean across two seeds and two orders, with Final=(2Old+Incoming)/3. Forget is lower-is-better. F5 is the matched V1 control, reused after disabled-projection parity qualification. B0/B2 are historical aggregate imports, not newly trained concurrent controls.

| Seed | Order | V2 Final % | Delta vs F5 pp | Delta vs historical B2 pp |
|---|---:|---:|---:|---:|
|163|1|70.5006|-0.0541|+3.6633|
|163|2|62.4007|+0.1137|-2.2552|
|164|1|65.6455|+0.2525|-4.9176|
|164|2|63.5036|+0.0948|-6.0444|

Both seed-average Final gains against F5 are positive (+0.0298pp and +0.1736pp), but the mean gain is only +0.1017pp, below the frozen +0.5pp criterion. Order-average Old/Incoming and individual-class tolerances pass against F5. V2 does not pass the practical gate. Against historical B2 the mean Final delta is −2.3885pp: seed163 +0.7041pp, seed164 −5.4810pp. Both orders violate the Old tolerance; order2 also violates the Incoming tolerance. Six individual domain/class deltas are below −5pp; the worst is seed164/order2/Drishti cup at −16.2705pp. B2 superiority is therefore not merely a small mean difference. The historical B0 gate also fails, despite +0.7086pp overall Final, because seed164 is negative and order2 Old decreases by more than0.5pp. All paired domain/rim/cup deltas and all gate outcomes are in RESULTS.json.

## Activation and interpretation

The module was called at all 16,960 active-U updates and detected/projected conflicts at 8,183 (48.25%). This is full active-update telemetry, unlike V1 sparse snapshots. Eight stages retained their four predefined gradient/module diagnostic points and 64 extra diagnostic VJPs in total; no extra image probes were added. Every projection checks the raw-gradient half-space tolerance. These engineering properties do not guarantee Adam update alignment or old-domain retention.

Mean Old improves +0.1747pp versus F5, Incoming decreases −0.0443pp, and Forget decreases0.1747pp. One of four Final pairs is negative. The intervention is active and qualified, but its practical gain is too small and it leaves the substantial B2 gap. This evidence does not establish that gradient conflict is the main cause of that gap. V2 is a negative practical result; no success or statistical significance claim is made.

## Costs, boundary and continuation

V2 qualification used32 CPU optimizer calls,10 generated CUDA calls (including the deliberate post-optimizer failure), and2 discarded real-L smoke calls. V1+V2 totals:106,000 formal calls, source0,416 CPU qualification calls,48 generated CUDA qualification calls,18 real-L smoke calls. Earlier failures and isolated driver qualification are retained. V2 controller elapsed3090.35 seconds; the matching runtime libraries remain private on NAS.

The audit seal inventory includes the32 existing V1 targets reused for the control; the V2 experimental inventory is eight newly sealed target students, not32 new students.

This round has completed training and readout. Public delivery must be verified separately. Hourly autonomous follow-up remains active because neither the F5 nor B2 success gate passed. A subsequent round requires a new finite preregistration and qualification; this round will not be tuned or restarted. The next analysis should examine which F5 architectural/gradient permissions differ from stronger B2, rather than assume that additional projection tuning will close the gap. No next round has been started by this report.

These are repeatedly reused development results, not independent patient confirmation, SOTA or clinical evidence. No independent/sealed test was read. Anonymous aggregates, code, protocol and costs are public; patient rows, images, weights, raw private logs and runtime configuration stay on NAS.
