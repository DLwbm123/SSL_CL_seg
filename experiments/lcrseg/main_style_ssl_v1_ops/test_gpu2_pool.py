"""CPU-only checks for slot reservation and unrelated-client admission."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('pool', Path(__file__).with_name('gpu2_pool.py'))
pool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pool)


class AdmissionTests(unittest.TestCase):
    def test_inherited_none_resume_cursor_is_valid(self):
        self.assertTrue(pool.has_progress({'status': 'RUNNING', 'step': 100}, {'resumed_from': None}))
        self.assertFalse(pool.has_progress({'status': 'RUNNING', 'step': 6000}, {'resumed_from': 6000}))
        self.assertTrue(pool.has_progress({'status': 'RUNNING', 'step': 6100}, {'resumed_from': 6000}))

    def test_checkpoint_watchers_reserve_slots_before_workers_move(self):
        legacy = {'old': {'gpu': 6, 'pid': 1}}
        self.assertFalse(pool.slot_available(legacy, {'a': {}, 'b': {}}))
        moving = dict(legacy, new={'gpu': 2, 'pid': 2})
        self.assertFalse(pool.slot_available(moving, {'b': {}}))
        self.assertTrue(pool.slot_available(moving, {}))

    def test_foreign_gpu_client_blocks_admission_even_with_free_memory(self):
        cards = '2, uuid2, 70000\n'
        with patch.object(pool.subprocess, 'check_output', side_effect=[cards, 'uuid2, 999\n']):
            self.assertFalse(pool.card_available({}))

    def test_existing_admitted_worker_is_allowed(self):
        running = {'a': {'gpu': 2, 'pid': 123, 'birth': 10}}
        with patch.object(pool.resource, 'alive', return_value=True), \
             patch.object(pool.subprocess, 'check_output', side_effect=['2, uuid2, 60000\n', 'uuid2, 123\n']):
            self.assertTrue(pool.card_available(running))

    def test_preserved_colocated_task_requires_matching_identity(self):
        peer = {'pid': 999, 'birth': 20}
        with patch.object(pool.resource, 'alive', return_value=True), \
             patch.object(pool.subprocess, 'check_output', side_effect=['2, uuid2, 50000\n', 'uuid2, 999\n']):
            self.assertTrue(pool.card_available({}, [peer]))
        with patch.object(pool.resource, 'alive', return_value=False), \
             patch.object(pool.subprocess, 'check_output', side_effect=['2, uuid2, 50000\n', 'uuid2, 999\n']):
            self.assertFalse(pool.card_available({}, [peer]))


if __name__ == '__main__':unittest.main()
