import torch
import torch.nn as nn

from .base import Conv1dBlock, EdgeConv2dBlock, LinearBlock
from .TransformNet import TransformNet


class DGCNNCommonFeat(nn.Module):
    def __init__(self, k=20):
        super(DGCNNCommonFeat, self).__init__()
        self.spatial_transform = TransformNet(k=k)

    def forward(self, x):
        transform_matrix = self.spatial_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transform_matrix)
        x = x.transpose(2, 1)

        return x


class DGCNNClassificationFeat(nn.Module):
    def __init__(self, k=20):
        super(DGCNNClassificationFeat, self).__init__()
        self.edge_conv1 = EdgeConv2dBlock(in_channels=3, out_channels=64, k=k)
        self.edge_conv2 = EdgeConv2dBlock(in_channels=64, out_channels=64, k=k)
        self.edge_conv3 = EdgeConv2dBlock(in_channels=64, out_channels=128, k=k)
        self.edge_conv4 = EdgeConv2dBlock(in_channels=128, out_channels=256, k=k)

        self.mlp = Conv1dBlock(in_channels=64 + 64 + 128 + 256, out_channels=1024)

    def forward(self, x):
        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)
        x4 = self.edge_conv4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.mlp(x)
        x, _ = torch.max(x, dim=2, keepdim=False)

        return x


class DGCNNClassificationClassifier(nn.Module):
    def __init__(self, num_class=10):
        super(DGCNNClassificationClassifier, self).__init__()
        self.mlp = nn.Sequential(
            LinearBlock(in_features=1024, out_features=512),
            nn.Dropout(p=0.5),
            LinearBlock(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.5),
            LinearBlock(in_features=256, out_features=num_class, activation=False, batch_norm=False, bias=True)
        )

    def forward(self, x):
        x = self.mlp(x)

        return x


class DGCNNClassification(nn.Module):
    def __init__(self, num_class=10, k=20):
        super(DGCNNClassification, self).__init__()
        self.common_feature_extractor = DGCNNCommonFeat(k=k)
        self.classification_feature_extractor = DGCNNClassificationFeat(k=k)
        self.classification_classifier = DGCNNClassificationClassifier(num_class=num_class)

    def forward(self, x):
        x = self.common_feature_extractor(x)
        x = self.classification_feature_extractor(x)
        x = self.classification_classifier(x)

        return x


class DGCNNClassificationFeatV2(nn.Module):
    def __init__(self, k=20):
        super(DGCNNClassificationFeatV2, self).__init__()
        self.edge_conv1 = EdgeConv2dBlock(in_channels=3, out_channels=64, k=k)
        self.edge_conv2 = EdgeConv2dBlock(in_channels=64, out_channels=64, k=k)
        self.edge_conv3 = EdgeConv2dBlock(in_channels=64, out_channels=128, k=k)
        self.edge_conv4 = EdgeConv2dBlock(in_channels=128, out_channels=256, k=k)

        self.mlp = Conv1dBlock(in_channels=64 + 64 + 128 + 256, out_channels=1024)

    def forward(self, x):
        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)
        x4 = self.edge_conv4(x3)

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.mlp(x)
        x, idx = torch.max(x, dim=2, keepdim=False)

        return x, idx


class DGCNNClassificationV2(nn.Module):
    def __init__(self, num_class=10, k=20):
        super(DGCNNClassificationV2, self).__init__()
        self.common_feature_extractor = DGCNNCommonFeat(k=k)
        self.classification_feature_extractor = DGCNNClassificationFeatV2(k=k)
        self.classification_classifier = DGCNNClassificationClassifier(num_class=num_class)

    def forward(self, x):
        x = self.common_feature_extractor(x)
        x, idx = self.classification_feature_extractor(x)
        x = self.classification_classifier(x)

        return x, idx
