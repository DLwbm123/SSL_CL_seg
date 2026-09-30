# Integration repairs

- First smoke process stopped before any optimizer call: the native factory requires a torch.device, not a string. Corrected both training and evaluator call sites. No objective/data/seed change; original window and physical ledger retained.
