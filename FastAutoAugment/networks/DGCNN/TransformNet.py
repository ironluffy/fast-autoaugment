import torch
import torch.nn as nn

from .base import Conv1dBlock, Conv2dBlock, LinearBlock, get_graph_feature


class TransformNet(nn.Module):
    def __init__(self, k=20):
        super(TransformNet, self).__init__()
        self.k = k
        self.conv1 = Conv2dBlock(in_channels=6, out_channels=64)
        self.conv2 = Conv2dBlock(in_channels=64, out_channels=128)
        self.conv3 = Conv1dBlock(in_channels=128, out_channels=1024)
        self.linear1 = LinearBlock(in_features=1024, out_features=512)
        self.linear2 = LinearBlock(in_features=512, out_features=256, bias=True)
        self.linear3 = LinearBlock(in_features=256, out_features=9, activation=False, batch_norm=False, bias=True)

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x, _ = torch.max(x, dim=-1, keepdim=False)

        x = self.conv3(x)
        x, _ = torch.max(x, dim=-1, keepdim=False)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)

        iden = torch.eye(3).view(1, 9).repeat(x.size(0), 1)
        iden = iden.to(device=x.device)
        x = x + iden
        x = x.view(x.size(0), 3, 3)

        return x
