import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


class FakeQuantFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, beta, bits):
        ctx.save_for_backward(x, alpha, beta)
        s = (alpha - beta) / (2**bits - 1)
        z = torch.round(-beta / s)
        return (torch.clamp(torch.round(x / s) + z, 0, 2**bits - 1) - z) * s

    @staticmethod
    def backward(ctx, grad):  # STE: gradient 1 inside [beta, alpha], 0 outside
        x, alpha, beta = ctx.saved_tensors
        return grad * ((x >= beta) & (x <= alpha)).to(grad.dtype), None, None, None


class FakeQuant(nn.Module):
    def __init__(self, bits=8, momentum=0.9):
        super().__init__()
        self.bits, self.momentum = bits, momentum
        self.register_buffer("mn", torch.tensor(float("inf")))
        self.register_buffer("mx", torch.tensor(float("-inf")))

    @torch.no_grad()
    def observe(self, x):
        if bool(self.mx < self.mn):
            self.mn.copy_(x.min())
            self.mx.copy_(x.max())
        else:
            self.mn.mul_(self.momentum).add_((1 - self.momentum) * x.min())
            self.mx.mul_(self.momentum).add_((1 - self.momentum) * x.max())

    def forward(self, x):
        if self.training or bool(self.mx < self.mn):
            self.observe(x)
        return FakeQuantFn.apply(x, self.mx.detach(), self.mn.detach(), self.bits)


def insert_fake_quant(model, bits=8):
    for name, mod in model.named_children():
        if isinstance(mod, nn.Linear):
            setattr(model, name, nn.Sequential(FakeQuant(bits=bits), mod))
        else:
            insert_fake_quant(mod, bits)
    return model


def make_data(n=256):
    torch.manual_seed(42)
    y = torch.randint(0, 2, (n,))
    centers = torch.tensor([[-1.5] + [0.0] * 15, [1.5] + [0.0] * 15])
    return torch.randn(n, 16) + centers[y], y


def train(model, x, y, epochs=200):
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        opt.step()


@torch.no_grad()
def accuracy(model, x, y):
    return (model(x).argmax(1) == y).float().mean().item()


if __name__ == "__main__":
    x, y = make_data()
    xt, yt, xv, yv = x[:192], y[:192], x[192:], y[192:]

    float_model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
    train(float_model, xt, yt)

    ptq_model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2))
    ptq_model.load_state_dict(float_model.state_dict())
    insert_fake_quant(ptq_model)
    ptq_model.train()
    for i in range(0, len(xt), 64):  # calibrate observers
        with torch.no_grad():
            ptq_model(xt[i : i + 64])
    ptq_model.eval()

    qat_model = insert_fake_quant(nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 2)))
    train(qat_model, xt, yt)
    qat_model.eval()

    print(f"float acc: {accuracy(float_model, xv, yv):.3f}")
    print(f"ptq   acc: {accuracy(ptq_model, xv, yv):.3f}")
    print(f"qat   acc: {accuracy(qat_model, xv, yv):.3f}")
