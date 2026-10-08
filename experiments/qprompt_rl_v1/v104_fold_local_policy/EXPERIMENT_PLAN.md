# V104 fold-local CE/RL generalization

Test whether fixed-mean4 features improve matched learned policy transfer over original states. Every fold receives fresh seed601/602 initialization, fold-only feature/reward scaling,512CEwarmup then two matched512update continuations. Reuse exact existing CE/RL objectives. Compare all controls with primary exact expected training reward, argmax secondary.

The complete machine-readable preregistration fixes the protocol.61442total actor steps include2synthetic qualification steps; zero native updates/image inference/queries. No data-independent population conclusion or sequential deployment claim follows from this shared-source table.
