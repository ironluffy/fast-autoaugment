import os
import glob
import h5py
import numpy as np
import torch
import torch.utils.data as data

import FastAutoAugment.utils.point_cloud_utils as pcu

idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
                4: "chair", 5: "lamp", 6: "monitor",
                7: "plant", 8: "sofa", 9: "table"}
label_to_idx = {"bathtub": 0, "bed": 1, "bookshelf": 2, "cabinet": 3,
                "chair": 4, "lamp": 5, "monitor": 6,
                "plant": 7, "sofa": 8, "table": 9}


class ScanNet(data.Dataset):
    def __init__(self, root, partition, seed=None):
        assert partition in ['train', 'val', 'trainval', 'test']
        self.root = os.path.join(root, 'PointDA_data', 'scannet')
        self.partition = partition
        self.seed = seed

        self.point_cloud_list, self.label_list = self.load_data_h5py()

        if partition == 'train':
            self.point_cloud_list = [self.point_cloud_list[i]
                                     for i in range(len(self.point_cloud_list)) if i % 10 < 8]
            self.label_list = [self.label_list[i] for i in range(len(self.label_list)) if i % 10 < 8]
        elif partition == 'val':
            self.point_cloud_list = [self.point_cloud_list[i]
                                     for i in range(len(self.point_cloud_list)) if i % 10 >= 8]
            self.label_list = [self.label_list[i] for i in range(len(self.label_list)) if i % 10 >= 8]

        self.point_cloud_list = np.swapaxes(self.point_cloud_list, 1, 2).astype(np.float64)
        self.point_cloud_list = torch.as_tensor(self.point_cloud_list)
        self.point_cloud_list = pcu.normalize(self.point_cloud_list).to(torch.float32)
        self.point_cloud_list = self.rotate_point_cloud(self.point_cloud_list)
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

    def load_data_h5py(self):
        all_data = []
        all_label = []
        if self.partition in ['test']:
            file_list = sorted(glob.glob(os.path.join(self.root, 'test_*.h5')))
        elif self.partition in ['train', 'val', 'trainval']:
            file_list = sorted(glob.glob(os.path.join(self.root, 'train_*.h5')))
        else:
            raise NotImplementedError
        for h5_name in file_list:
            f = h5py.File(h5_name, 'r')
            data = f['data'][:]
            label = f['label'][:]
            f.close()
            all_data.append(data)
            all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data[:, :, :3], all_label.tolist()

    @staticmethod
    def rotate_point_cloud(point_cloud_list):
        point_cloud_list = pcu.rotate_shape(point_cloud_list, 'x', -np.pi / 2)

        return point_cloud_list

    def get_info(self):
        unique, counts = np.unique(self.label_list, return_counts=True)
        return dict(zip([idx_to_label[idx] for idx in unique], counts))
