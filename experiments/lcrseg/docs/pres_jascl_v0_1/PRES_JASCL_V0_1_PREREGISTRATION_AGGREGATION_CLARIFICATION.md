# PRES-JASCL V0.1 aggregation clarification

This clarification resolves one wording conflict in preregistration commit `cd797d55362fd997beb6a9b7d5878aa790392831`. It does not change the scientific contract, thresholds, candidates, seeds, roles, mappings, isolation rules, outputs, or hard stop.

The explicit frozen-evaluator requirement has precedence. For every seed/domain/expert or routing policy, the segmentation evaluator accumulates one 3×3 pixel confusion matrix over all authorized validation cases, ignores label 255, and then derives per-class Dice/IoU, Mean Dice, Foreground Dice, and Mean IoU exactly as the regenerated-B0 evaluator does. Case-level Dice or IoU is not averaged.

After that within-domain pixel aggregation, seeds are averaged equally inside each domain. Three-domain and REFUGE/RIM historical summaries average domains equally. D1/D3 no-domain-drop checks remain per unrounded seed/domain cell. Historical forgetting is fixed Oracle-snapshot foreground Dice minus Shared-final foreground Dice; Drishti_GS is the current-domain result and is excluded from the historical forgetting average.

Routing remains case-level: per-seed/domain accuracy is correct cases divided by cases, seeds are averaged equally within domain, and macro accuracy is the equal-domain mean. Bootstrap routing uses the same aggregation. This is the only operative interpretation of the original generic `case mean` phrase.
