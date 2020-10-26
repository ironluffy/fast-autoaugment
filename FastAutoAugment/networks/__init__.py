import torch

from torch import nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.backends.cudnn as cudnn

from .PointNet import PointNetClassification, PointNetClassificationV7
from .DGCNN.DGCNN import DGCNNClassification


def get_model(conf, num_class=10, local_rank=-1):
    name = conf['type']

    if name == 'pointnet':
        model = PointNetClassification(num_class=num_class)
    elif name == 'dgcnn':
        model = DGCNNClassification(num_class=num_class)
    elif name == 'pointnet_dc':
        model = PointNetClassificationV7(num_class=2)

    else:
        raise NameError('no model named, %s' % name)

    if local_rank >= 0:
        device = torch.device('cuda', local_rank)
        model = model.to(device)
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = model.cuda()
    #         model = DataParallel(model)

    cudnn.benchmark = True
    return model


def num_class(dataset):
    return {
        'pointda': 2,
    }[dataset]
