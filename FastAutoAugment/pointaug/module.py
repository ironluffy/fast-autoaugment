import numpy as np
import torch
from torch import nn

torch.autograd.set_detect_anomaly(True)


class BaseModule(nn.Module):
    """ Base module class with some basic additions to the pytorch Module class
    """

    @property
    def nb_params(self):
        """This property is used to return the number of trainable parameters for a given layer
        It is useful for debugging and reproducibility.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._nb_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._nb_params


def weight_variable(shape):
    initial = torch.empty(shape, dtype=torch.float)
    torch.nn.init.xavier_normal_(initial)
    return initial


class Seq(nn.Sequential):
    def __init__(self):
        super().__init__()
        self._num_modules = 0

    def append(self, module):
        self.add_module(str(self._num_modules), module)
        self._num_modules += 1
        return self


class FastBatchNorm1d(BaseModule):
    def __init__(self, num_features, momentum=0.1, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, momentum=momentum, **kwargs)

    def _forward_dense(self, x):
        return self.batch_norm(x)

    def _forward_sparse(self, x):
        """ Batch norm 1D is not optimised for 2D tensors. The first dimension is supposed to be
        the batch and therefore not very large. So we introduce a custom version that leverages BatchNorm1D
        in a more optimised way
        """
        x = x.unsqueeze(2)
        x = x.transpose(0, 2)
        x = self.batch_norm(x)
        x = x.transpose(0, 2)
        return x.squeeze()

    def forward(self, x):
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError("Non supported number of dimensions {}".format(x.dim()))


class Linear(Seq):
    def __init__(self, in_channels, out_channels, bias=True, bn=True, activation=nn.LeakyReLU(negative_slope=0.01)):
        super().__init__()
        self.append(nn.Linear(in_channels, out_channels, bias=bias))
        if bn:
            self.append(nn.BatchNorm1d(out_channels))
        if activation:
            self.append(activation)


class Conv1D(Seq):
    def __init__(self, in_channels, out_channels, bias=True, bn=True, activation=nn.LeakyReLU(negative_slope=0.01)):
        super().__init__()
        self.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias))
        if bn:
            self.append(nn.BatchNorm1d(out_channels))
        if activation:
            self.append(activation)


class MLP(Seq):
    def __init__(self, channels, bias=False, bn=True, activation=nn.LeakyReLU(negative_slope=0.01)):
        super().__init__()
        for i in range(len(channels) - 1):
            self.append(Linear(channels[i], channels[i + 1], bn=bn, bias=bias, activation=activation))


class MLP1D(Seq):
    def __init__(self, channels, bias=False, bn=True, activation=nn.LeakyReLU(negative_slope=0.01)):
        super().__init__()
        for i in range(len(channels) - 1):
            self.append(Conv1D(channels[i], channels[i + 1], bn=bn, bias=bias, activation=activation))


nn_raising_config = [3, 64, 64, 128, 1024]
nn_rotation_config = [1024*2, 512, 256, 4, 9]
nn_translation_config = [1024*3, 1024, 512, 64, 3, 3]

class AugmentationModule(nn.Module):
    """
    PointAugment: an Auto-Augmentation Framework for Point Cloud Classification
    https://arxiv.org/pdf/2002.10876.pdf
    """

    def __init__(self, conv_type="DENSE"):
        super(AugmentationModule, self).__init__()
        self._conv_type = conv_type

        if conv_type == "DENSE":
            # per point feature extraction
            self.nn_raising = MLP1D(nn_raising_config)
            # shape-wise regression
            self.nn_rotation = MLP1D(nn_rotation_config)
            # point-wise regression
            self.nn_translation = MLP1D(nn_translation_config)
        else:
            self.nn_raising = MLP(nn_raising_config)
            self.nn_rotation = MLP(nn_rotation_config)
            self.nn_translation = MLP(nn_translation_config)

    def forward(self, data):

        if self._conv_type == "DENSE":
            batch_size = data.size(0)
            num_points = data.size(2)
            F = self.nn_raising(data)
            G, _ = F.max(-1)
            noise_rotation = torch.randn(G.size()).to(G.device)
            noise_translation = torch.randn(F.size()).to(F.device)

            feature_rotation = [noise_rotation, G]
            feature_translation = [F, G.unsqueeze(-1).repeat((1, 1, num_points)), noise_translation]

            features_rotation = torch.cat(feature_rotation, dim=1).unsqueeze(-1)
            features_translation = torch.cat(feature_translation, dim=1)

            M = self.nn_rotation(features_rotation).view((batch_size, 3, 3))
            D = self.nn_translation(features_translation).permute(0, 2, 1)

            new_data = data.clone()
            new_data.pos = D + new_data.pos @ M
        else:
            batch_size = data.pos.shape[0]
            num_points = data.pos.shape[1]
            F = self.nn_raising(data.pos.permute(0, 2, 1))
            G, _ = F.max(-1)
            noise_rotation = torch.randn(G.size()).to(G.device)
            noise_translation = torch.randn(F.size()).to(F.device)

            feature_rotation = [noise_rotation, G]
            feature_translation = [F, G.unsqueeze(-1).repeat((1, 1, num_points)), noise_translation]

            features_rotation = torch.cat(feature_rotation, dim=1).unsqueeze(-1)
            features_translation = torch.cat(feature_translation, dim=1)

            M = self.nn_rotation(features_rotation).view((batch_size, 3, 3))
            D = self.nn_translation(features_translation).permute(0, 2, 1)

            new_data = data.clone()
            new_data.pos = D + new_data.pos @ M

        return new_data


if __name__ == "__main__":
    model = AugmentationModule()
    in_point = torch.randn(32, 3, 1024)
    print(model(in_point))