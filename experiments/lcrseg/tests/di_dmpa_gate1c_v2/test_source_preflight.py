"""Stdlib-only checks; also collected by the existing pytest suite."""
import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[2]/'scripts/audit_gate1c_v22_sources.py'
spec = importlib.util.spec_from_file_location('gate1c_source_preflight', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SourcePreflightTests(unittest.TestCase):
    def test_comments_and_whitespace_are_not_behavior(self):
        result = module.identical_definitions('def f(x):\n return x + 1\n',
            '# comment\ndef f(x):\n    return x+1  # comment\n', ['f'])
        self.assertEqual(set(result), {'f'})
        self.assertEqual(len(result['f']), 64)

    def test_changed_missing_and_duplicate_definitions_fail(self):
        original = 'def f(x):\n return x + 1\n'
        for changed in ('def f(x):\n return x + 2\n', 'def g(x):\n return x + 1\n', original+original):
            with self.subTest(source=changed), self.assertRaises(ValueError):
                module.identical_definitions(original, changed, ['f'])

    def test_cache_schema_constants_cannot_change(self):
        original = "CACHE_FIELDS = ('R0', 'R1')\n"
        self.assertIn('CACHE_FIELDS', module.identical_definitions(original, original, ['CACHE_FIELDS']))
        with self.assertRaises(ValueError):
            module.identical_definitions(original, "CACHE_FIELDS = ('R0', 'R2')\n", ['CACHE_FIELDS'])


if __name__ == '__main__':
    unittest.main()
