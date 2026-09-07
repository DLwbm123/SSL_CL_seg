# Reproduction and execution

Use the existing py38 environment on zmic44 and the pinned official classifier reference 3c93ca70784fc3a1d2a887f8d7dce5af6bc75f53. Every remote command is wrapped with bash experiments/lcrseg/scripts/with_nas_storage.sh. Inputs DATA=/home/jiangsuiyang/SSL_CL use the existing canonical DATA/h5/v1 binding.

Modules in experiments.lcrseg.single_teacher_scd_r1:
1. preflight --base NEW_NAS_ROOT --old-root V01_NAS_ROOT --data DATA
2. corpus --old-root V01_NAS_ROOT --output NEW_NAS_ROOT/corpus --data DATA --reference JASCL_REFERENCE (GPU7; three diagnostic updates)
3. qualification --output NEW_NAS_ROOT/qualification_server --device cuda:0 --reference JASCL_REFERENCE --corpus NEW_NAS_ROOT/corpus --old-failed V01_NAS_ROOT/engineering_bracket_probe_01/private_failure_vectors.pt (GPU7)
4. execute --base NEW_NAS_ROOT --qualification NEW_NAS_ROOT/qualification_server/qualification.json --data DATA --reference JASCL_REFERENCE (U0/L05/L10)
5. Only when E qualification PASS: same execute command with --include-e.
6. report --base NEW_NAS_ROOT --output NEW_NAS_ROOT/public_results.

Parents persist child command/PID/exit/time receipts. New stage2 retires only its own new stage1 rolling slots after evaluation, preserving old research. Resume is permitted only with matching source, R1 identity and checkpoint-aligned ledger. An unsupported divergence closes the attempt rather than silently duplicating steps.
