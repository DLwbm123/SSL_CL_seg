# SR-GAS V0.2 Test Report

Status: `SRGAS_V0_2_TESTS_PASSED`

- V0.2 contract tests: `23/23` passed.
- Complete repository regression suite: `171/171` passed.
- The V0.2 suite includes lag-buffer reset/use/commit/skip/resume, frozen 20% warm-up, first warm-start step zero noise, shared-noise equality/resume, L2/L3/L4 sensitivity sources, D1/D2 timing contracts, architecture equivalence, unchanged safety threshold, and a real L4 one-batch backward from a V0.1a-compatible parent followed by successful lag-buffer commit.
- The complete regression suite retains existing golden-batch, checkpoint-resume, old-model gradient, hidden-GT, anchor immutability, and SR-GAS V0.1/V0.1a coverage.

Commands were executed with `/home/jiangsuiyang/anaconda3/envs/py38/bin/python` and `PYTHONPATH=.` from `/home/jiangsuiyang/SSL_CL/code/SSL_CL_seg/experiments/lcrseg`.
