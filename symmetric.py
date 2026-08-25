import torch


def quantize(x, bits=8):
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().max() / qmax
    xq = torch.clamp(torch.round(x / s), -qmax, qmax)
    return xq.to(torch.int), s


def dequantize(xq, s):
    return s * xq.float()


if __name__ == "__main__":
    x = torch.tensor([43.31, -44.93, 0.0, 22.99, -43.93, -11.35, 38.48, -20.49, -38.61, -28.02])
    xq, s = quantize(x)
    xd = dequantize(xq, s)
    print("scale:", round(s.item(), 4))
    print("quantized:  ", xq.tolist())
    print("dequantized:", [round(v, 2) for v in xd.tolist()])
    print("max error:", round((x - xd).abs().max().item(), 3))
