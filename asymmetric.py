import torch


def quantize(x, bits=8):
    qmin, qmax = 0, 2**bits - 1
    s = (x.max() - x.min()) / (qmax - qmin)
    z = torch.round(-x.min() / s)
    xq = torch.clamp(torch.round(x / s) + z, qmin, qmax)
    return xq.to(torch.int), s, z


def dequantize(xq, s, z):
    return s * (xq.float() - z)


if __name__ == "__main__":
    x = torch.tensor([43.31, -44.93, 0.0, 22.99, -43.93, -11.35, 38.48, -20.49, -38.61, -28.02, 550000])
    xq, s, z = quantize(x)
    xd = dequantize(xq, s, z)
    print("scale:", round(s.item(), 4), "zero-point:", z.item())
    print("quantized:  ", xq.tolist())
    print("dequantized:", [round(v, 2) for v in xd.tolist()])
    print("max error:", round((x - xd).abs().max().item(), 3))
