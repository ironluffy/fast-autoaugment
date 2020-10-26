import torch
import torch.nn as nn


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, batch_norm=True):
        super(Conv1dBlock, self).__init__()
        modules = [nn.Conv1d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1, bias=False)]
        if batch_norm:
            modules.append(nn.BatchNorm1d(out_channels))

        if activation:
            modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv_block(x)

        return x


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, batch_norm=True):
        super(Conv2dBlock, self).__init__()
        modules = [nn.Conv2d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1, bias=False)]
        if batch_norm:
            modules.append(nn.BatchNorm2d(out_channels))

        if activation:
            modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv_block(x)

        return x


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=True, batch_norm=True, bias=False):
        super(LinearBlock, self).__init__()
        modules = [nn.Linear(in_features=in_features,
                             out_features=out_features,
                             bias=bias)]

        if batch_norm:
            modules.append(nn.BatchNorm1d(out_features))

        if activation:
            modules.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.linear_block = nn.Sequential(*modules)

    def forward(self, x):
        x = self.linear_block(x)

        return x


class EdgeConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, batch_norm=True, k=20):
        super(EdgeConv2dBlock, self).__init__()
        self.k = k
        self.conv_block = Conv2dBlock(in_channels=in_channels * 2,
                                      out_channels=out_channels,
                                      activation=activation,
                                      batch_norm=batch_norm)

    def forward(self, x):
        x = get_graph_feature(x, k=self.k)
        x = self.conv_block(x)
        x = x.max(dim=-1, keepdim=False)[0]

        return x


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x[:, 6:], k=k)
    device = torch.device(x.device)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature  # (batch_size, 2*num_dims, num_points, k)
