"""Compare p(S|sampled Z) with p(S|observed Y) under the same hierarchy."""
import math

import torch
from torch.nn import functional as F

from sc_bayeseg.model import normal_log_prob, sar, student_log_prob
from ..feedback_c1.model import FeedbackModel, standard_risk_from_logits
from ..semantic_latent_h1.model import LatentReadout


class ConditionalFeedback(FeedbackModel):
    def __init__(self,label_condition):
        super().__init__(cut=True)
        self.label_condition=label_condition
        self.unet.requires_grad_(False)

    def objectives(self,image,label,generator):
        output=self.unet(image)
        task=standard_risk_from_logits(output.logits,label)
        d=self.draw(image,generator,output)
        condition=(F.one_hot(label,2).permute(0,3,1,2).to(image.dtype)
                   if self.label_condition else d['probability'])
        observed=student_log_prob(image-d['shape']-d['appearance'],.1)
        appearance_prior=normal_log_prob(d['appearance'],torch.zeros_like(d['appearance']),
                                         torch.full_like(d['appearance'],math.log(.5)))
        shape_prior=student_log_prob(
            sar(d['shape']-self.shape_mean(condition),.9),.2)+self.sar_logdet
        segmentation_prior=student_log_prob(sar(d['z'],.9),1.)+self.sar_logdet
        bound=observed+appearance_prior+shape_prior+segmentation_prior-d['logq']
        variational=-bound.mean()/(self.size*self.size)
        return task,variational,output.logits,d


class ConditionalReadout(LatentReadout):
    def __init__(self,label_condition):
        super().__init__(use_shape=True)
        self.label_condition=label_condition
        self.base=ConditionalFeedback(label_condition)
        self.base.unet.requires_grad_(False)
