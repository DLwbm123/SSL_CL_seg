# Executed commands and evidence scope

All commands here are local Git operations or read-only admission checks.
There is no experiment launch command: none was executed or authorized as ready.

The local private bundle's `EXACT_COMMAND_RECEIPTS.json` retains literal argv,
exit status and provenance for the executed commands. It also binds the exact
executed local-input audit script and full output by SHA-256.
`DESTINATION_CONNECTION_PREFLIGHT.json` retains the candidate address, account,
port, known-host fingerprints and actual authentication output.

Public commands below substitute `DESTINATION_HOST` and `PRIVATE_AUDIT` for
private locations only; obtain their literal values from that private receipt.
These substitutions are not an alternate destination or a runnable experiment.

```sh
git switch -c codex/gate1c-v3-clean-regeneration 70ba3dfb4fc989a5149a6343d857e7d10fd2017d
git add -- experiments/lcrseg/docs/di_dmpa_jascl/GATE1C_V22_UNRECOVERABLE_CLOSURE.md experiments/lcrseg/docs/di_dmpa_jascl/GATE1C_V22_UNRECOVERABLE_CLOSURE.json
git commit -m 'docs: close unrecoverable Gate1C v2.2 without inferring a full result'
git push -u origin codex/gate1c-v3-clean-regeneration
git ls-remote --heads origin codex/gate1c-v3-clean-regeneration
ssh -G jiangsuiyang
ssh-keygen -F "$DESTINATION_HOST" -l
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o ConnectionAttempts=1 -p 22 "jiangsuiyang@$DESTINATION_HOST" 'id -a'
ssh-add -l
/opt/miniconda3/bin/python -B "$PRIVATE_AUDIT/audit_local_inputs.py"
```

The SSH probe failed authentication before a remote shell was entered.
The local input audit exited 0, called the existing
`lcrseg.acceptance.verify_checksums`, checked the six frozen seed hashes and
transfer manifest, and inspected the local official JASCL Git state.
It did not decode HDF5 labels, load models, use GPUs or run forwards.
It writes only create-only audit outputs in the new private local directory;
it does not invoke the old CLI that would write into the frozen data root.

No server-local PROCESS_START/PID/EXIT/EXECUTION_COMPLETION receipt is fabricated:
there was no remote experiment process. No historical test suite is relabeled
as freshly executed JUnit evidence.
