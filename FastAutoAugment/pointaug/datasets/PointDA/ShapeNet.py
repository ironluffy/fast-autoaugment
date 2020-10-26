import os
import glob
import numpy as np
import torch
import torch.utils.data as data

import utils.point_cloud_utils as pcu

idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
                4: "chair", 5: "lamp", 6: "monitor",
                7: "plant", 8: "sofa", 9: "table"}
label_to_idx = {"bathtub": 0, "bed": 1, "bookshelf": 2, "cabinet": 3,
                "chair": 4, "lamp": 5, "monitor": 6,
                "plant": 7, "sofa": 8, "table": 9}


class ShapeNet(data.Dataset):
    def __init__(self, root, partition, seed=None):
        assert partition in ['train', 'val', 'trainval', 'test']
        self.root = os.path.join(root, 'PointDA_data', 'shapenet')
        self.partition = partition
        self.seed = seed

        if partition in ['test']:
            point_cloud_dir_list = sorted(glob.glob(os.path.join(self.root, '*', 'test', '*.npy')))
        elif partition in ['train', 'val', 'trainval']:
            point_cloud_dir_list = sorted(glob.glob(os.path.join(self.root, '*', 'train', '*.npy')))
        else:
            raise NotImplementedError

        if partition == 'train':
            point_cloud_dir_list = [point_cloud_dir_list[i]
                                    for i in range(len(point_cloud_dir_list)) if i % 10 < 8]
        elif partition == 'val':
            point_cloud_dir_list = [point_cloud_dir_list[i]
                                    for i in range(len(point_cloud_dir_list)) if i % 10 >= 8]

        self.point_cloud_list = np.array([np.load(point_cloud_dir) for point_cloud_dir in point_cloud_dir_list])
        self.point_cloud_list = np.swapaxes(self.point_cloud_list, 1, 2).astype(np.float64)
        self.point_cloud_list = torch.as_tensor(self.point_cloud_list)
        self.point_cloud_list = pcu.normalize(self.point_cloud_list).to(torch.float32)
        self.label_list = [label_to_idx[_dir.split('/')[-3]] for _dir in point_cloud_dir_list]
        self.point_cloud_list = self.rotate_point_cloud(self.point_cloud_list, self.label_list)
        self.targets = self.label_list

    def __getitem__(self, idx):
        output = {
            'point_cloud': self.point_cloud_list[idx],
            'label': self.label_list[idx],
            'seed': self.seed,
        }

        return output

    def __len__(self):
        return len(self.point_cloud_list)

    @staticmethod
    def rotate_point_cloud(point_cloud_list, label_list):
        for batch, label in enumerate(label_list):
            if label != label_to_idx['plant']:
                point_cloud_list[batch:batch+1] = pcu.rotate_shape(point_cloud_list[batch:batch+1], 'x', -np.pi / 2)

        return point_cloud_list

    def get_info(self):
        unique, counts = np.unique(self.label_list, return_counts=True)
        return dict(zip([idx_to_label[idx] for idx in unique], counts))
