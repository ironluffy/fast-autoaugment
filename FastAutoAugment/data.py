import torch
import random
import logging
import torch.distributed as dist

from FastAutoAugment.datasets import PointDA
from torch.utils.data import SubsetRandomSampler, Sampler, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from theconf import Config as C, ConfigArgumentParser
from FastAutoAugment.utils.point_augmentations import apply_augment
from FastAutoAugment.common import get_logger

logger = get_logger('Fast AutoAugment')
logger.setLevel(logging.INFO)


def get_dataloaders(dataset, batch, dataroot, split=0.15, cv_num=5, split_idx=0, multinode=False, target_lb=-1,
                    target=False):
    if 'pointda':
        transform_train = []
        transform_test = []
    else:
        raise ValueError('dataset=%s' % dataset)

    total_aug = augs = None
    if isinstance(C.get()['aug'], list):
        logger.debug('augmentation provided.')
        # transform_train.append(Augmentation(C.get()['aug']))
    else:
        logger.debug('augmentation: %s' % C.get()['aug'])

        if C.get()['aug'] in ['default']:
            pass
        else:
            raise ValueError('not found augmentations. %s' % C.get()['aug'])

    if dataset == 'pointda':
        total_trainset = PointDA(root=dataroot, domain=C.get()['target' if target else 'source'], partition='trainval',
                                 num_points=1024, sampling_method='random', download=True, transforms=transform_train)
        testset = PointDA(root=dataroot, domain=C.get()['target' if target else 'source'], partition='test',
                          num_points=1024, sampling_method='random', download=True)
    else:
        raise ValueError('invalid dataset name=%s' % dataset)

    if total_aug is not None and augs is not None:
        total_trainset.set_preaug(augs, total_aug)
        print('set_preaug-')

    train_sampler = None
    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=cv_num, test_size=split, random_state=0)
        sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)

        if target_lb >= 0:
            train_idx = [i for i in train_idx if total_trainset.targets[i] == target_lb]
            valid_idx = [i for i in valid_idx if total_trainset.targets[i] == target_lb]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)

        if multinode:
            train_sampler = torch.utils.data.distributed.DistributedSampler(Subset(total_trainset, train_idx),
                                                                            num_replicas=dist.get_world_size(),
                                                                            rank=dist.get_rank())
    else:
        valid_sampler = SubsetSampler([])

        if multinode:
            train_sampler = torch.utils.data.distributed.DistributedSampler(total_trainset,
                                                                            num_replicas=dist.get_world_size(),
                                                                            rank=dist.get_rank())
            logger.info(f'----- dataset with DistributedSampler  {dist.get_rank()}/{dist.get_world_size()}')

    trainloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=True if train_sampler is None else False, num_workers=0,
        pin_memory=True,
        sampler=train_sampler, drop_last=True)
    validloader = torch.utils.data.DataLoader(
        total_trainset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True,
        sampler=valid_sampler, drop_last=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True,
        drop_last=False
    )
    return train_sampler, trainloader, validloader, testloader


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, pnt):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                pnt = apply_augment(pnt, name, level)
        return pnt


class SubsetSampler(Sampler):
    r"""Samples elements from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)
