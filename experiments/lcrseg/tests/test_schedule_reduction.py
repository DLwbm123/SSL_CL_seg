"""CPU-only checks for subset scheduling and honest single-seed aggregation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('reduction', ROOT / 'experiments/lcrseg/scripts/reduce_native_schedule.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReductionTest(unittest.TestCase):
    def test_subset_and_summary(self):
        plan = json.loads((ROOT / 'experiments/lcrseg/docs/five_frameworks_v1/native_execution/RESOLVED_PROTOCOL.json').read_text())
        families = ['B0_PARENT_LCTX', 'B3_PARENT_LCTX_DENSEG', 'B4_PARENT_PAS_KL_DENSEG_JOINT', 'F1', 'F2', 'F3', 'F4', 'F5']
        cancelled = [n['id'] for n in plan['nodes'] if
                     (n.get('phase') == 'D' and (n['seed'] != 162 or n['family'] not in families)) or
                     (n.get('phase') == 'C' and n.get('family') == 'F5' and n['candidate_id'] not in ['F5_C01', 'F5_C02'])]
        amendment = {'amendment_id': 'test', 'replication_families': families, 'retained_F5_candidates': ['F5_C01', 'F5_C02'],
                     'cancelled_node_ids': cancelled, 'retained_node_ids': [n['id'] for n in plan['nodes'] if n['id'] not in cancelled],
                     'comparison_references': families[:3]}
        nodes = module.reduced_nodes(plan, amendment)
        self.assertEqual(len(cancelled), 112)
        self.assertEqual(len(nodes), 328)
        self.assertEqual(sum(n.get('phase') == 'D' for n in nodes), 32)
        self.assertEqual(len(next(n for n in nodes if n['id'] == 'SELECT_F5')['dependencies']), 4)
        broken = copy.deepcopy(amendment)
        broken['retained_node_ids'].remove('SOURCE_S162')
        broken['cancelled_node_ids'].append('SOURCE_S162')
        with self.assertRaises(ValueError):
            module.reduced_nodes(plan, broken)
        receipts = {}
        for family in families:
            for order in (1, 2):
                ident = f'D__{family}__O{order}'
                receipts[ident] = {'identity': {'family': family, 'stage': 2, 'seed': 162, 'order': order},
                                  'Final': .5, 'Old': .4, 'Incoming': .7, 'Forget': .1,
                                  'seen': ['a', 'b', 'c'], 'scores': {d: {c: .5 for c in ('rim', 'cup', 'disc_union')} for d in ('a', 'b', 'c')}}
        with tempfile.TemporaryDirectory() as tmp:
            write = lambda p, v: p.write_text(json.dumps(v))
            module.summarize(Path(tmp), receipts, amendment, write)
            result = json.loads((Path(tmp) / 'replication_summary_reduced.json').read_text())
            self.assertEqual(result['table']['F1']['seed_Final'], {'162': .5})
            self.assertIsNone(result['seed_confidence_intervals'])
            receipts.pop(next(iter(receipts)))
            with self.assertRaises(ValueError):
                module.summarize(Path(tmp), receipts, amendment, write)


if __name__ == '__main__':
    unittest.main()
