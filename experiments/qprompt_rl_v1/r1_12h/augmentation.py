"""Exact geometry/strong math from f6b9697; generators supplied by frozen schedule."""
import torch
from torch.nn import functional as F


def geometry(image, label, valid, generator):
    dimensions = []
    if torch.rand((), generator=generator) < .5:
        dimensions.append(-1)
    if torch.rand((), generator=generator) < .5:
        dimensions.append(-2)
    k = int(torch.randint(4, (), generator=generator))
    def change(x):
        if dimensions:
            x = x.flip(dimensions)
        return x.rot90(k, (-2,-1)).contiguous()
    return change(image), None if label is None else change(label), change(valid)


def strong(weak, generator):
    g = generator
    outputs = []
    weights = weak.new_tensor([.2989,.587,.114])[:,None,None]
    for image in weak:
        factors = .8 + .4 * torch.rand(3, generator=g, device=weak.device)
        x = image * factors[0]
        gray = (x * weights).sum(0,keepdim=True)
        x = (x - gray.mean()) * factors[1] + gray.mean()
        gray = (x * weights).sum(0,keepdim=True)
        x = (x - gray) * factors[2] + gray
        if torch.rand((),generator=g,device=weak.device) < .5:
            sigma = .1 + .9 * torch.rand((),generator=g,device=weak.device)
            axis = torch.arange(-2,3,device=weak.device,dtype=x.dtype)
            kernel = (-axis.square()/(2*sigma.square())).exp()
            kernel = kernel / kernel.sum()
            kernel = (kernel[:,None]*kernel[None,:]).expand(3,1,5,5)
            x = F.conv2d(F.pad(x[None],(2,2,2,2),mode="reflect"),kernel,groups=3)[0]
        if torch.rand((),generator=g,device=weak.device) < .5:
            x = x + .02 * torch.randn(x.shape,generator=g,device=x.device)
        outputs.append(x.clamp(0,1))
    return torch.stack(outputs)
