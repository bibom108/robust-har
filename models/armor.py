import torch
import torch.nn.functional as F
from torch import Tensor, nn
import models.network as net
import copy


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Armor(nn.Module):

    def __init__(
        self,
        args,
        out = 6
    ):
        super().__init__()

        self.num_experts = len(args.sigma_train)

        self.experts = nn.ModuleList(
            [
                net.Network()
                for sigma_train in args.sigma_train
            ]
        )

        self.weight = nn.Parameter(torch.randn(out, self.num_experts, dtype=torch.float32) * 0.02)
        self.args = args


    def load_experts_weight(self):
        for i, sigma_train in enumerate(self.args.sigma_train):
            weight_file = "logs/" + self.args.adv + f"/{sigma_train}/weight.pt"
            self.experts[i] = torch.load(weight_file).to(DEVICE)


    def forward(self, x):
        expert_outputs = [expert(x) for expert in self.experts] # 21x128x6

        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1) # 128x6x21

        out = stacked_expert_outputs * self.weight

        out = torch.sum(out, dim=-1)

        return out, 0 # 128x6
