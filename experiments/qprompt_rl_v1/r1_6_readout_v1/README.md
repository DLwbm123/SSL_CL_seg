# R1.6 readout supervision

Frozen plan: TASK_PLAN.json. Run modes use the same registry and checkpoint schema. The historical model, matcher, augmentation, optimizer and metric are imported unchanged. Public files exclude private runtime paths and medical payloads.

Entry (through stdin, with private environment): `from r1_6_readout_v1.runner import main; main()`. Modes: prepare, qualify, smoke, forensic, train, evaluate, supervise. CPU regression: `python -m r1_6_readout_v1.test_contract`. A single coordinator grants durable attempts and task ownership. Recovery across a source commit requires an explicit validated repair binding; this implementation rejects implicit rebinding.

No training results are claimed by the initial source release. Final closeout must review all prescribed tables, checkpoint provenance and the frozen fresh-seed decision, then publish only aggregate evidence.
