# Deterministic source CE compatibility

The first native CUDA round on commit 3a438571dba4b80e1726a57fbf1b715d9126784b passed 13 tests and failed the generated-data source worker before its first optimizer call. PyTorch 2.2.1 rejects nll_loss2d under deterministic algorithms. Cumulative qualification optimizer calls: 29/256; real smoke and formal updates: zero.

The source worker now takes the CE component from the existing designated supervised_parts(log_softmax(logits), labels). It retains the same ignore-aware per-valid-pixel cross-entropy objective and the deterministic native reduction already used by target training. The Dice return value is not part of the source objective. Determinism is not disabled, no new candidate or training step is added, and the original CUDA attempt remains in the cumulative ledger. All qualification receipts must bind the new execution commit before smoke or formal dispatch.
