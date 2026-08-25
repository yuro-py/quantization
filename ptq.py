import torch
import torch.nn as nn

torch.manual_seed(0)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class Int8Linear(nn.Module):
    # weights: symmetric per-channel int8, bias: int32, input quantized on the fly (dynamic)
    def __init__(self, lin):
        super().__init__()
        w = lin.weight.data
        self.s_w = w.abs().amax(dim=1, keepdim=True) / 127.0
        self.wq = torch.round(w / self.s_w).clamp(-127, 127).to(torch.int8)
        self.b = nn.Parameter(lin.bias.data.clone(), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        s_x = (x.max() - x.min()) / 255.0
        z_x = torch.round(-x.min() / s_x)
        xq = (torch.round(x / s_x) + z_x).clamp(0, 255).to(torch.int32)
        acc = xq @ self.wq.to(torch.int32).T
        acc -= (z_x.to(torch.int32) * self.wq.to(torch.int32).sum(dim=1)).unsqueeze(0)
        acc += torch.round(self.b / (s_x * self.s_w.flatten())).to(torch.int32).unsqueeze(0)
        return (s_x * self.s_w.flatten()).unsqueeze(0) * acc.float()


def to_int8(model):
    qmodel = Net()
    qmodel.fc1 = Int8Linear(model.fc1)
    qmodel.fc2 = Int8Linear(model.fc2)
    return qmodel


if __name__ == "__main__":
    model = Net()
    data = torch.randn(64, 32)
    with torch.no_grad():
        y_ref = model(data)
        y_int = to_int8(model)(data)
    print("max abs diff:", (y_ref - y_int).abs().max().item())
    print("argmax match:", (y_ref.argmax(1) == y_int.argmax(1)).float().mean().item())
