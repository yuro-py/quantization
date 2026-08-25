import torch
import torch.nn.functional as F


def dq(x, alpha, beta, bits=8):
    s = (alpha - beta) / (2**bits - 1)
    z = torch.round(-beta / s)
    xq = torch.clamp(torch.round(x / s) + z, 0, 2**bits - 1)
    return s * (xq - z)


def min_max(x):
    return dq(x, x.max(), x.min())


def percentile(x, p=99):
    v = x.flatten()
    return dq(x, torch.quantile(v, p / 100), torch.quantile(v, 1 - p / 100))


def mse(x, steps=100):
    best, best_err = (x.max(), x.min()), None
    for f in torch.linspace(0.3, 1.0, steps):
        r = (x.max() * f, x.min() * f)
        err = (x - dq(x, *r)).pow(2).mean()
        if best_err is None or err < best_err:
            best, best_err = r, err
    return dq(x, *best)


def cross_entropy(x, steps=100):
    target = F.softmax(x.flatten(), dim=-1)
    best, best_loss = (x.max(), x.min()), None
    for f in torch.linspace(0.3, 1.0, steps):
        xd = dq(x, x.max() * f, x.min() * f).flatten()
        loss = -(target * F.log_softmax(xd, dim=-1)).sum()
        if best_loss is None or loss < best_loss:
            best, best_loss = (x.max() * f, x.min() * f), loss
    return dq(x, *best)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(256)
    x[:16] *= 8  # moderate outliers
    for name, fn in [("min-max", min_max), ("percentile", percentile), ("mse", mse), ("cross-entropy", cross_entropy)]:
        xd = fn(x)
        top8 = x.topk(8).indices
        order = (xd[top8].argsort(descending=True).argsort() == torch.arange(8)).all().item()
        err_out = (x[:16] - xd[:16]).abs().mean().item()
        err_in = (x[16:] - xd[16:]).abs().max().item()
        print(f"{name:14s} outlier mean err: {err_out:7.2f}   inlier max err: {err_in:.4f}   top-8 order: {order}")
