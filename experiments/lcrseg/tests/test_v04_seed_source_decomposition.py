import inspect

from lcrseg.engine.continual_runner import ContinualRunner


def test_v04_optimization_seed_is_separate_from_frozen_split_seed() -> None:
    init_source = inspect.getsource(ContinualRunner.__init__)
    run_source = inspect.getsource(ContinualRunner.run)
    dataset_source = inspect.getsource(ContinualRunner._datasets)

    assert 'get("optimization_seed", self.seed)' in init_source
    assert "seed_everything(\n            self.optimization_seed" in run_source
    assert run_source.count("seed=self.optimization_seed") == 3
    assert "seed=self.seed" in dataset_source
    assert 'f"{self.dataset}_seed{self.seed}.json"' in init_source
    assert 'lcrseg_v1_seed{self.seed}.csv' in init_source
