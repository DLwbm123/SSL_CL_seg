import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path

spec = importlib.util.spec_from_file_location('host_ops', Path(__file__).with_name('host_migration.py'))
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)

class HandoffTests(unittest.TestCase):
    def test_disabled_print_flag_normalization(self):
        state = dict(last_epoch=2000, base_lrs=[.01], _last_lr=[.009], verbose=False)
        normalized = ops.normalize_scheduler_state(state)
        self.assertEqual(normalized, {k:v for k,v in state.items() if k != 'verbose'})
        self.assertFalse(state['verbose'])
        with self.assertRaises(ValueError): ops.normalize_scheduler_state(dict(verbose=True))

    def test_free_memory_gate_keeps_existing_jobs(self):
        config = dict(gpus=[0, 1], minimum_free_mib=8192)
        with patch.object(ops.subprocess, 'check_output', return_value='0, 40959\n1, 6000\n'):
            self.assertEqual(ops.available_cards(config), [0])
        self.assertEqual(ops.available_cards(dict(gpus=[4, 6])), [4, 6])

    def config(self):
        keys = ['a', 'b', 'c', 'd']
        return dict(gpus=[0, 1], capacity=2, expected_handoffs=keys,
                    handoffs={'a': {}}, pending_handoffs=keys[1:]), [dict(id=k, steps=127900) for k in keys]

    def test_admitted_target_and_legacy_cards(self):
        config, plan = self.config()
        ops.validate_handoffs(config, plan)
        config['gpus'] = [4, 6]
        ops.validate_handoffs(config, plan)

    def test_wrong_budget_and_duplicate_handoff_rejected(self):
        config, plan = self.config()
        config['pending_handoffs'].append('a')
        with self.assertRaises(AssertionError): ops.validate_handoffs(config, plan)
        config, plan = self.config()
        plan[0]['steps'] -= 1
        with self.assertRaises(AssertionError): ops.validate_handoffs(config, plan)

    def test_unknown_card_and_node_rejected(self):
        config, plan = self.config()
        config['gpus'] = [2, 3]
        with self.assertRaises(AssertionError): ops.validate_handoffs(config, plan)
        config, plan = self.config()
        config['expected_handoffs'][0] = 'unknown'
        with self.assertRaises(AssertionError): ops.validate_handoffs(config, plan)

if __name__ == '__main__': unittest.main()
