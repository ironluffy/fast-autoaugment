import os
import gdown
import zipfile
import torch.utils.data as data

import utils.point_cloud_utils as pcu

from .ModelNet import ModelNet
from .ShapeNet import ShapeNet
from .ScanNet import ScanNet

URL = 'https://drive.google.com/uc?id=1-LfJWL5geF9h0Z2QpdTL0n4lShy8wy2J'
ZIP_NAME = 'PointDA_data.zip'
FOLDER_NAME = 'PointDA_data'
DOMAIN_DATASET = {
    'modelnet': ModelNet,
    'shapenet': ShapeNet,
    'scannet': ScanNet,
}
SAMPLING_METHOD = {
    'fixed': pcu.fixed_point_sample,
    'random': pcu.random_point_sample,
    'farthest': pcu.farthest_point_sample,
}


class PointDA(data.Dataset):
    def __init__(self, root, domain, partition, num_points=1024, sampling_method='random', download=False, test_cnt=1,
                 transforms=None):
        if download:
            self.download(root)
        self.domain = domain
        self.num_points = num_points
        self.partition = partition
        if partition in ['train', 'trainval']:
            self.dataset = DOMAIN_DATASET[domain](root=root, partition=partition, seed=None)
            self.targets = self.dataset.targets
        elif partition in ['test', 'val']:
            self.dataset = data.ConcatDataset([
                DOMAIN_DATASET[domain](root=root, partition=partition, seed=seed) for seed in range(test_cnt)])
            self.targets = []
            for seed in range(test_cnt):
                self.targets.extend(DOMAIN_DATASET[domain](root=root, partition=partition, seed=seed).targets)
        else:
            raise NotImplementedError
        self.sampling_method = sampling_method
        self.transforms = transforms

    def __getitem__(self, idx):
        item = self.dataset[idx]
        point_cloud = item['point_cloud']
        label = item['label']
        seed = item['seed']
        point_cloud = point_cloud.unsqueeze(dim=0)

        if point_cloud.size(2) > self.num_points:
            point_cloud = SAMPLING_METHOD[self.sampling_method](point_cloud, self.num_points, seed=seed)[0]

        if self.partition in ['train', 'trainval']:
            point_cloud = pcu.random_rotate_one_axis(point_cloud, axis='z')
            point_cloud = pcu.jitter(point_cloud)
            point_cloud = pcu.point_permutate(point_cloud)

        if self.transforms is not None:
            transformed = point_cloud.clone().detach()
            for transform in self.transforms:
                transformed = transform(point_cloud)
            transformed = transformed.squeeze()
            point_cloud = point_cloud.squeeze()
            output = {
                'point_cloud': point_cloud,
                'transformed': transformed,
                'label': label,
            }

        else:
            point_cloud = point_cloud.squeeze()
            output = {
                'point_cloud': point_cloud,
                'label': label,
            }

        return output

    def __len__(self):
        return len(self.dataset)

    def get_info(self):
        if isinstance(self.dataset, data.ConcatDataset):
            info = {}
            for dataset in self.dataset.datasets:
                cur_info = dataset.get_info()
                for key in cur_info.keys():
                    if key in info:
                        info[key] += cur_info[key]
                    else:
                        info[key] = cur_info[key]
        else:
            info = self.dataset.get_info()

        return info

    @staticmethod
    def download(root):
        os.makedirs(root, exist_ok=True)
        if os.path.exists(os.path.join(root, FOLDER_NAME)):
            return
        if not os.path.exists(os.path.join(root, ZIP_NAME)):
            gdown.download(URL, os.path.join(root, ZIP_NAME), quiet=False)

        file = zipfile.ZipFile(os.path.join(root, ZIP_NAME))
        file.extractall(root)
        file.close()
