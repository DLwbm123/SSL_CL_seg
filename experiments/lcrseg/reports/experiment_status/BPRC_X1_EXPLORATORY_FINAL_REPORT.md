# BPRC-X1 exploratory final report

**Final status:** `BPRC_X1_PREVIOUS_UTILITY_NOT_SUPPORTED`  
**Candidate:** `X1 = B2 top-2 pairwise class-balanced / 3`  
**Seed-0 pilot authorized:** `false`  
**Optimizer steps:** `0`  
**Epistemic status:** user-authorized exploratory test outside the frozen BPRC V0.1 protocol

## 1. Question tested

The completed BPRC V0.1 audit showed that pairwise objectives had excessive relation-gradient scale. This exploratory test asked whether a single analytically fixed change could make the most promising previous-loss candidate usable:

```text
X0 = exact B0 categorical pixel mean
X1 = existing B2 top-2 pairwise class-balanced loss / C
C  = 3 semantic classes
```

The `1/3` factor was frozen before execution. No alternative scales, temperatures, lambdas, pair definitions, GT weights, or outcome-driven choices were tested. All BPRC V0.1 artifacts remained immutable.

## 2. Execution

Seeds 0/1/2 ran independently on physical GPUs 5/6/7. Each seed reused byte-identical BPRC V0.1 fixed-batch manifests and the same frozen checkpoints:

- two transitions per seed;
- 32 current-site update batches per transition;
- 16 previous-site and 16 current-site validation batches;
- stateless virtual-step norm `1e-3`;
- exact TARC/R0 loss and margin evaluators;
- 384 gradient rows, 384 virtual-step rows, and 3456 margin rows in total.

All 17 engineering checks passed. Model/checkpoint mutation, old-model gradients, hidden-GT update usage, optimizer steps, and non-finite values were all zero. Complete regression: `198 passed in 19.89s`.

## 3. Gradient-scale result

**PASS.** The fixed normalization solved the scale problem:

| Metric | Raw B2 from V0.1 | X1 = B2/3 | Gate |
|---|---:|---:|---:|
| median X1/X0 gradient ratio | `5.947623` | `1.982534` | `[0.5,2.0]` |
| p10 | `3.126392` | `1.042118` | `>=0.25` |
| p90 | `11.872448` | `3.957435` | `<=4.0` |
| non-finite | `0` | `0` | `0` |

The numerical effect was almost exactly the intended factor of three.

## 4. Previous-site utility

**FAIL. This is the first failed gate and determines the final status.**

- X1 previous-val loss delta lower than X0: `50.5208%` of 192 paired comparisons (required `>=60%`);
- median X0 delta: `-0.0004644850`;
- median X1 delta: `-0.0005317228`;
- required median bound: `<=-0.0005644850`;
- median paired X1-minus-X0: `-0.0000254232`.

The normalized top-2 loss had a small aggregate benefit, but it was neither frequent nor large enough to establish historical utility.

## 5. Current-site safety

**PASS.**

- median current loss delta X0/X1: `-0.0001477972` / `-0.0005701724`;
- median current Dice delta X0/X1: `+0.0000869036` / `+0.0002551973`.

The candidate remained safe for current-site performance, as the unnormalized pairwise candidates did.

## 6. Disc-rim margin

**FAIL independently.**

- X1 improved disc-rim margin over X0 in `32.8125%` of comparisons (required `>=60%`);
- median disc-rim X1-minus-X0: `-0.0000742510` (required `>=+0.005`);
- classwise median deltas: background `-0.0000164619`, disc-rim `-0.0000742510`, cup `+0.0001878647`.

The class-safety floor passed, but the intended disc-rim effect remained negative. Normalization reduced magnitude; it did not correct the relation-gradient direction.

## 7. Conclusion

The suggestion had one real effect: dividing B2 by three repaired gradient scale exactly as predicted. It did not repair previous-site utility or disc-rim retention. Therefore the BPRC failure is not merely a loss-magnitude problem; the top-2 old-winner relation signal is directionally misaligned with the desired historical disc-rim behavior.

Under the frozen exploratory decision rule, a 1000-step pilot would add optimizer cost without diagnostic support. It was not implemented or launched.

## 8. Canonical evidence

| Artifact | SHA-256 |
|---|---|
| `BPRC_X1_EXPLORATORY_PROTOCOL.json` | `3863b40f767ac2c72dd2ef5c4fec9a464e55c26c6af23ada6f8066fb7c86161f` |
| `BPRC_X1_DIAGNOSTIC_AUDIT.json` | `2d6c1dddec483aa0a4fce95a69b424b0850470018c836a77372ab317d06c712e` |
| `BPRC_X1_EXPLORATORY_STOP.json` | `85def4b25ecf8ee2eb82adfa2772c54a7eeaffbe862cd96b39cd38250f6c83e7` |
| `gradient_scale.csv` | `93209a8b8f9f2c391eefeb94d7b206a4f64a28447809a732711ef23fb35316d0` |
| `virtual_steps.csv` | `14968ff2173cfc5b646551fae28ee9527804b7a402323373860d8feea5b3e687` |
| `margin_analysis.csv` | `f2c5c874b757333cd4bf988a5fee556d749fce7e45e65b6a9142ab0ef6ff4eca` |
| `BPRC_X1_FULL_TESTS.log` | `fb8653d04f9619cc3ac52964377f87baa8423b4786275ee6b18908e51e969043` |
