import math
import torch


def sar(x, alpha=.9):
    return x - alpha / 4 * sum(torch.roll(x, n, d) for n, d in ((1, -1), (-1, -1), (1, -2), (-1, -2)))


def sar_eigenvalues(height, width, like, alpha=.9):
    u = torch.arange(height, dtype=like.dtype, device=like.device)
    v = torch.arange(width, dtype=like.dtype, device=like.device)
    return 1 - alpha / 2 * (torch.cos(2 * math.pi * u / height)[:, None]
                            + torch.cos(2 * math.pi * v / width)[None, :])


def sar_solve(x, alpha=.9):
    eig = sar_eigenvalues(*x.shape[-2:], x, alpha)
    return torch.fft.ifft2(torch.fft.fft2(x) / eig).real


def sar_logdet(height, width, like, alpha=.9):
    return sar_eigenvalues(height, width, like, alpha).log().sum()


def sar_log_prob(x, mean, sigma=.2, alpha=.9):
    c, h, w = x.shape[-3:]
    return (-c * h * w * (.5 * math.log(2 * math.pi) + math.log(sigma))
            + c * sar_logdet(h, w, x, alpha)
            - sar(x - mean, alpha).square().flatten(1).sum(1) / (2 * sigma ** 2))


def dct_basis(h, w, order, dtype=torch.float64):
    x = torch.arange(h, dtype=dtype) + .5
    y = torch.arange(w, dtype=dtype) + .5
    return torch.stack([
        (torch.cos(math.pi * u * x / h) * math.sqrt((1 if u == 0 else 2) / h))[:, None]
        * (torch.cos(math.pi * v * y / w) * math.sqrt((1 if v == 0 else 2) / w))[None, :]
        for u in range(order) for v in range(order) if u or v])
