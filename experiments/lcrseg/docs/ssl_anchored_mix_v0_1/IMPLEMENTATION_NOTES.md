# Implementation boundary

The original training loop, canonical data access, model, head, EMA, optimizer and random streams are reused in the new namespace. New CE uses the predecessor explicit log-softmax gather, avoiding CUDA nll_loss atomic reductions. Dice sums use FP64. The complementary two-view loss collects log probabilities by source; odd U duplicates are loss-weighted before unique-source normalization. No clean-L extra objective is added.

All qualification and real execution processes instrument actual Adam/backward/model forward/EMA/HDF5/sample operations. Cumulative counters persist at optimizer, completed step, diagnostic patient and process exit boundaries; original per-training-stream counters also remain. A fatal OS kill can leave the last unflushed operation incomplete; such an attempt must not be silently declared complete.

The compact evaluator keeps one clean and four noise draws on original val images, aggregates probability/teacher precision/recall/correctness summaries and seals outputs before evaluator GT. No real gradient grid or new diagnostic backward. Main output is epoch100 student; fresh deployment is image-only.

P0 uses only old public metrics and old task/initialization/deployment metadata. Bootstrap reads the permitted old SUP final patient-score CSVs and new final score CSVs; patient indices correspond to the same sorted frozen manifest and are never public. No old model loads or forward.

P1 and P2 selection are computed from complete fixed matrices. Statistical intervals do not change gates. Optional joint bootstrap is omitted; required patient and training-seed bootstrap each have 2000 repeats.

The user subsequently specified GPU3/4/5; these override the attachment previous GPU list.
