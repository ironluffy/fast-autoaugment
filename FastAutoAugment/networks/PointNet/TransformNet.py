import torch
import torch.nn as nn

from .base import Conv1dBlock, LinearBlock


class TransformNet(nn.Module):
    def __init__(self, K):
        super(TransformNet, self).__init__()
        self.K = K
        self.conv1 = Conv1dBlock(in_channels=K, out_channels=64)
        self.conv2 = Conv1dBlock(in_channels=64, out_channels=128)
        self.conv3 = Conv1dBlock(in_channels=128, out_channels=1024)
        self.linear1 = LinearBlock(in_features=1024, out_features=512)
        self.linear2 = LinearBlock(in_features=512, out_features=256)
        self.linear3 = LinearBlock(in_features=256, out_features=K * K, activation=False, batch_norm=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        iden = torch.eye(self.K).view(1, self.K * self.K).repeat(x.size(0), 1)
        iden = iden.to(device=x.device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)

        return x
