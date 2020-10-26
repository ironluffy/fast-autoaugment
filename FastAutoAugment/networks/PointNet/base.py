import torch.nn as nn


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, batch_norm=True):
        super(Conv1dBlock, self).__init__()
        modules = [nn.Conv1d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1, bias=True)]
        if batch_norm:
            modules.append(nn.BatchNorm1d(out_channels))

        if activation:
            modules.append(nn.ReLU(inplace=True))

        self.conv_block = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv_block(x)

        return x


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=True, batch_norm=True):
        super(LinearBlock, self).__init__()
        modules = [nn.Linear(in_features=in_features,
                             out_features=out_features)]

        if batch_norm:
            modules.append(nn.BatchNorm1d(out_features))

        if activation:
            modules.append(nn.ReLU(inplace=True))

        self.linear_block = nn.Sequential(*modules)

    def forward(self, x):
        x = self.linear_block(x)

        return x
