"""Source-only control: hard, class-balanced pseudo supervision on admitted pixels."""
from pathlib import Path
import torch
from ..main_style_ssl_v1.model import MainTrainer
from ..five_frameworks_v1.gate import digest


def hard_objective(logits, target, valid):
    if logits.shape != target.shape or valid.shape != logits.shape[:1]+logits.shape[2:] or valid.dtype != torch.bool:
        raise ValueError('pseudo target and mask grids must match')
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise ValueError('nonfinite pseudo objective input')
    labels = target.detach().argmax(1)
    valid = valid.detach()
    logp = logits.log_softmax(1)
    ce = -logp.gather(1, labels[:, None])[:, 0]
    counts = torch.stack([(valid & (labels == c)).sum() for c in range(logits.shape[1])])
    present = counts > 0
    terms = [(ce*(valid & (labels == c))).sum()/counts[c].clamp_min(1)
             for c in range(logits.shape[1])]
    balanced_ce = (torch.stack(terms)*present).sum()/present.sum().clamp_min(1)
    prob = logp.exp()*valid[:, None]
    dice_terms = []
    for c in range(1, logits.shape[1]):
        truth = (labels == c) & valid
        dice_terms.append(1-(2*(prob[:, c]*truth).sum()+1e-5)/(prob[:, c].sum()+truth.sum()+1e-5))
    foreground = present[1:]
    foreground_dice = (torch.stack(dice_terms)*foreground).sum()/foreground.sum().clamp_min(1)
    return balanced_ce+foreground_dice, dict(class_pixels=counts.tolist(),
        balanced_ce=float(balanced_ce.detach()), foreground_dice=float(foreground_dice.detach()),
        valid_pixels=int(valid.sum()), foreground_classes=int(foreground.sum()))


class HardPseudoTrainer(MainTrainer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not self.model.source_dense:
            raise ValueError('hard-pseudo control is source-only')

    def kl_objective(self,images,logits,target,valid):
        loss,stats = hard_objective(logits,target,valid)
        self.last['pseudo'] = stats
        return loss

    def loss_contract(self):
        return {**super().loss_contract(), 'source_u_objective':'class-balanced hard CE + present-foreground Dice',
                'source_u_implementation':digest(Path(__file__).read_text())}


def conservative_objective(logits, target, valid):
    if logits.shape != target.shape or valid.shape != logits.shape[:1]+logits.shape[2:] or valid.dtype != torch.bool:
        raise ValueError('pseudo target and mask grids must match')
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise ValueError('nonfinite pseudo objective input')
    confidence, labels = target.detach().max(1)
    kept = valid.detach() & (confidence > .95)
    ce = -logits.log_softmax(1).gather(1, labels[:, None])[:, 0]
    # Full-grid normalization prevents a few admitted pixels from receiving full-batch weight.
    loss = (ce*kept).mean()
    return loss, dict(valid_pixels=int(kept.sum()), grid_pixels=kept.numel(),
                      class_pixels=[int((kept & (labels == c)).sum()) for c in range(logits.shape[1])])


class ConservativePseudoTrainer(HardPseudoTrainer):
    def kl_objective(self, images, logits, target, valid):
        loss, stats = conservative_objective(logits, target, valid)
        self.last['pseudo'] = stats
        return loss

    def loss_contract(self):
        return {**super().loss_contract(), 'source_u_objective': 'confidence>0.95 hard CE, full-grid mean, no class balancing or Dice'}
