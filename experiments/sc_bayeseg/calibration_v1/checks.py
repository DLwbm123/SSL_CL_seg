"""Regression for diagnostic RNG isolation; zero backward/optimizer calls."""
import torch
from lcrseg.engine.checkpoint import capture_rng_state
from lcrseg.engine.trainer import seed_everything
from sc_bayeseg.run import same
from sc_bayeseg.calibration_v1 import run


class DensityStub:
    def draw(self):pass
    def clear_draw(self):pass
    def kl(self):return torch.tensor(0.)
    def bound(self,image,y):
        value=torch.randn(2)
        return value,dict(image=value),{}


def main():
    seed_everything(163);before=capture_rng_state()
    first=run.density(DensityStub(),None,None)
    assert same(before,capture_rng_state())
    second=run.density(DensityStub(),None,None)
    assert first==second and same(before,capture_rng_state())
    original=run._evaluate
    def noisy_evaluate(*args):
        seed_everything(50163)
        return dict(value=float(torch.rand(())))
    try:
        run._evaluate=noisy_evaluate
        first=run.evaluate(None,None,None)
        assert same(before,capture_rng_state())
        assert first==run.evaluate(None,None,None) and same(before,capture_rng_state())
    finally:run._evaluate=original
    print('PASS full RNG isolation; zero backward and optimizer calls')


if __name__=='__main__':main()
