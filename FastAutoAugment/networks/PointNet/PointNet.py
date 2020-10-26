import torch
import torch.nn as nn

from .base import Conv1dBlock, LinearBlock
from .TransformNet import TransformNet


class PointNetCommonFeat(nn.Module):
    def __init__(self):
        super(PointNetCommonFeat, self).__init__()
        self.input_transform = TransformNet(3)
        self.feature_transform = TransformNet(64)

        self.mlp = nn.Sequential(
            Conv1dBlock(in_channels=3, out_channels=64),
            Conv1dBlock(in_channels=64, out_channels=64),
        )

    def forward(self, x):
        transform_matrix = self.input_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transform_matrix)
        x = x.transpose(2, 1)

        x = self.mlp(x)

        transform_matrix = self.feature_transform(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, transform_matrix)
        x = x.transpose(2, 1)

        return x


class PointNetClassificationFeat(nn.Module):
    def __init__(self):
        super(PointNetClassificationFeat, self).__init__()
        self.mlp = nn.Sequential(
            Conv1dBlock(in_channels=64, out_channels=64),
            Conv1dBlock(in_channels=64, out_channels=128),
            Conv1dBlock(in_channels=128, out_channels=1024),
        )

    def forward(self, x):
        x = self.mlp(x)
        x, _ = torch.max(x, dim=2, keepdim=False)

        return x


class PointNetClassificationClassifier(nn.Module):
    def __init__(self, num_class=10):
        super(PointNetClassificationClassifier, self).__init__()
        self.mlp = nn.Sequential(
            LinearBlock(in_features=1024, out_features=512),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=512, out_features=256),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=256, out_features=num_class, activation=False, batch_norm=False)
        )

    def forward(self, x):
        x = self.mlp(x)

        return x


class PointNetClassification(nn.Module):
    def __init__(self, num_class=10):
        super(PointNetClassification, self).__init__()
        self.common_feature_extractor = PointNetCommonFeat()
        self.classification_feature_extractor = PointNetClassificationFeat()
        self.classification_classifier = PointNetClassificationClassifier(num_class=num_class)

    def forward(self, x):
        x = self.common_feature_extractor(x)
        x = self.classification_feature_extractor(x)
        x = self.classification_classifier(x)

        return x


class PointNetClassificationV2(nn.Module):
    def __init__(self, num_class=2):
        super(PointNetClassificationV2, self).__init__()
        self.common_feature_extractor = self.mlp = nn.Sequential(
            Conv1dBlock(in_channels=3, out_channels=64),
            Conv1dBlock(in_channels=64, out_channels=64),
        )
        self.classification_feature_extractor = PointNetClassificationFeat()
        self.classification_classifier = PointNetClassificationClassifier(num_class=num_class)

    def forward(self, x):
        x = self.common_feature_extractor(x)
        x = self.classification_feature_extractor(x)
        x = self.classification_classifier(x)

        return x


class PointNetClassificationV3(nn.Module):
    def __init__(self, num_class=2):
        super(PointNetClassificationV3, self).__init__()
        self.common_feature_extractor = self.mlp = nn.Sequential(
            Conv1dBlock(in_channels=3, out_channels=64),
            Conv1dBlock(in_channels=64, out_channels=64),
        )
        self.classification_feature_extractor = nn.Sequential(
            Conv1dBlock(in_channels=64, out_channels=64),
            Conv1dBlock(in_channels=64, out_channels=128),
            Conv1dBlock(in_channels=128, out_channels=256),
        )
        self.classification_classifier = nn.Sequential(
            LinearBlock(in_features=256, out_features=128),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=128, out_features=64),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=64, out_features=num_class, activation=False, batch_norm=False)
        )

    def forward(self, x):
        x = self.common_feature_extractor(x)
        x = self.classification_feature_extractor(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = self.classification_classifier(x)

        return x


class PointNetClassificationV4(nn.Module):
    def __init__(self, num_class=2):
        super(PointNetClassificationV4, self).__init__()
        self.common_feature_extractor = self.mlp = nn.Sequential(
            Conv1dBlock(in_channels=3, out_channels=64),
            Conv1dBlock(in_channels=64, out_channels=64),
        )
        self.classification_feature_extractor = nn.Sequential(
            Conv1dBlock(in_channels=64, out_channels=64),
            Conv1dBlock(in_channels=64, out_channels=128),
        )
        self.classification_classifier = nn.Sequential(
            LinearBlock(in_features=128, out_features=64),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=64, out_features=32),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=32, out_features=num_class, activation=False, batch_norm=False)
        )

    def forward(self, x):
        x = self.common_feature_extractor(x)
        x = self.classification_feature_extractor(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = self.classification_classifier(x)

        return x


class PointNetClassificationV5(nn.Module):
    def __init__(self, num_class=2):
        super(PointNetClassificationV5, self).__init__()
        self.common_feature_extractor = self.mlp = nn.Sequential(
            Conv1dBlock(in_channels=3, out_channels=16),
            Conv1dBlock(in_channels=16, out_channels=32),
        )
        self.classification_feature_extractor = nn.Sequential(
            Conv1dBlock(in_channels=32, out_channels=32),
            Conv1dBlock(in_channels=32, out_channels=64),
        )
        self.classification_classifier = nn.Sequential(
            LinearBlock(in_features=64, out_features=32),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=32, out_features=16),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=16, out_features=num_class, activation=False, batch_norm=False)
        )

    def forward(self, x):
        x = self.common_feature_extractor(x)
        x = self.classification_feature_extractor(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = self.classification_classifier(x)

        return x


class PointNetClassificationV6(nn.Module):
    def __init__(self, num_class=2):
        super(PointNetClassificationV6, self).__init__()
        self.common_feature_extractor = self.mlp = nn.Sequential(
            Conv1dBlock(in_channels=3, out_channels=16),
            Conv1dBlock(in_channels=16, out_channels=32),
        )
        self.classification_feature_extractor = nn.Sequential(
            Conv1dBlock(in_channels=32, out_channels=32),
            Conv1dBlock(in_channels=32, out_channels=32),
        )
        self.classification_classifier = nn.Sequential(
            LinearBlock(in_features=32, out_features=16),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=16, out_features=8),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=8, out_features=num_class, activation=False, batch_norm=False)
        )

    def forward(self, x):
        x = self.common_feature_extractor(x)
        x = self.classification_feature_extractor(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = self.classification_classifier(x)

        return x


class PointNetClassificationV7(nn.Module):
    def __init__(self, num_class=2):
        super(PointNetClassificationV7, self).__init__()
        self.common_feature_extractor = self.mlp = nn.Sequential(
            Conv1dBlock(in_channels=3, out_channels=16),
            Conv1dBlock(in_channels=16, out_channels=32),
        )
        self.classification_feature_extractor = nn.Sequential(
            Conv1dBlock(in_channels=32, out_channels=32),
        )
        self.classification_classifier = nn.Sequential(
            LinearBlock(in_features=32, out_features=16),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=16, out_features=8),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=8, out_features=num_class, activation=False, batch_norm=False)
        )

    def forward(self, x):
        x = self.common_feature_extractor(x)
        x = self.classification_feature_extractor(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = self.classification_classifier(x)

        return x


class PointNetClassificationV8(nn.Module):
    def __init__(self, num_class=2):
        super(PointNetClassificationV8, self).__init__()
        self.common_feature_extractor = self.mlp = nn.Sequential(
            Conv1dBlock(in_channels=3, out_channels=16),
            Conv1dBlock(in_channels=16, out_channels=32),
        )
        self.classification_classifier = nn.Sequential(
            LinearBlock(in_features=32, out_features=16),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=16, out_features=8),
            nn.Dropout(p=0.3),
            LinearBlock(in_features=8, out_features=num_class, activation=False, batch_norm=False)
        )

    def forward(self, x):
        x = self.common_feature_extractor(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = self.classification_classifier(x)

        return x


class PointNetClassificationFeatureExtractor(nn.Module):
    def __init__(self):
        super(PointNetClassificationFeatureExtractor, self).__init__()
        self.common_feature_extractor = PointNetCommonFeat()
        self.classification_feature_extractor = PointNetClassificationFeat()

    def forward(self, x):
        x = self.common_feature_extractor(x)
        x = self.classification_feature_extractor(x)

        return x