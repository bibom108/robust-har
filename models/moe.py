import torch
import torch.nn.functional as F
import torch.nn as nn
import models.resnet


class SwitchMoE(nn.Module):

    def __init__(self, device, expert_weight: list, dim: int = 6):
        super().__init__()

        self.dim = dim

        self.num_experts = len(expert_weight)
        self.experts = nn.ModuleList(
            [
                torch.load(file).eval()
                for file in expert_weight
            ]
        )

        self.gate = models.resnet.resnet18(num_classes=self.num_experts).to(device)


    def forward(self, x):
        expert_outputs = torch.stack([expert(x) for expert in self.experts], -1)

        gate_scores = torch.unsqueeze(F.softmax(self.gate(x), dim=-1), -1)
        top_k_scores, top_k_indices = gate_scores.topk(1, dim=-1)

        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )

        return torch.squeeze(torch.matmul(expert_outputs, mask))


# def test():
#     net = SwitchMoE(["logs/clean_training/0.0/0.0/weight.pt", "logs/robust_train_full/0.25/0.5/weight.pt"])
#     y = net(torch.randn(128, 9, 128))
#     print(y.size())

# test()
