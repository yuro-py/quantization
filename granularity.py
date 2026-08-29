import torch


def _dq(x, s, z, qmax):
    xq = torch.clamp(torch.round(x / s) + z, 0, qmax)
    return s * (xq - z)


def per_tensor(x, bits=8):
    qmax = 2**bits - 1
    s = (x.max() - x.min()) / qmax
    z = torch.round(-x.min() / s)
    return _dq(x, s, z, qmax)


def per_channel(x, bits=8, dim=0):
    qmax = 2**bits - 1
    dims = [d for d in range(x.dim()) if d != dim]
    mn, mx = x.amin(dim=dims, keepdim=True), x.amax(dim=dims, keepdim=True)
    s = (mx - mn) / qmax
    z = torch.round(-mn / s)
    return _dq(x, s, z, qmax)


def per_group(x, group_size=8, bits=8):
    qmax = 2**bits - 1
    shape = x.shape
    xg = x.reshape(-1, group_size)
    mn, mx = xg.amin(dim=-1, keepdim=True), xg.amax(dim=-1, keepdim=True)
    s = (mx - mn) / qmax
    z = torch.round(-mn / s)
    return _dq(xg, s, z, qmax).reshape(shape)


if __name__ == "__main__":
    torch.manual_seed(0)
    w = torch.cat([torch.randn(4, 32) * 0.01, torch.randn(4, 32) * 10.0])  # mixed channel scales
    for name, fn in [("per-tensor", per_tensor), ("per-channel", lambda t: per_channel(t, dim=0)), ("per-group", lambda t: per_group(t, group_size=8))]:
        params = {"per-tensor": 2, "per-channel": 2 * w.shape[0], "per-group": 2 * (w.numel() // 8)}[name]
        print(f"{name:12s} mean err: {(w - fn(w)).abs().mean().item():9.5f}   ({params} params)")
