import os
import torch
import numpy
import random
from datetime import datetime
import argparse
import aug_eval
from utils.logger.ExperimentLogger import ExperimentLogger
from utils.point_augmentations import augment_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='aug_trained', type=str)
    parser.add_argument('--data_dir', default='..', type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--dc_model', type=str, default='pointnetv7',
                        choices=['pointnet', 'dgcnn', 'pointnetv5', 'pointnetv7'])
    parser.add_argument('--source_domain', type=str, choices=['modelnet', 'scannet', 'shapenet'])
    parser.add_argument('--target_domain', type=str, choices=['modelnet', 'scannet', 'shapenet'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--source_train_sampling', type=str, default='random')
    parser.add_argument('--source_val_sampling', type=str, default='fixed')
    parser.add_argument('--source_test_sampling', type=str, default='fixed')
    parser.add_argument('--target_train_sampling', type=str, default='random')
    parser.add_argument('--target_val_sampling', type=str, default='fixed')
    parser.add_argument('--target_test_sampling', type=str, default='fixed')
    parser.add_argument('--num-op', type=int, default=2)
    parser.add_argument('--num_cv', type=int, default=5)
    parser.add_argument('--num-policy', type=int, default=5)
    parser.add_argument('--num-search', type=int, default=100)
    parser.add_argument('--tag', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--aug_all', action='store_true')
    parser.add_argument('--random_range', type=float, default=0.3)
    parser.add_argument('--trs_deter', action='store_true')
    parser.add_argument('--only_eval', action='store_true')
    parser.add_argument('--use_emd_false', action='store_false')
    parser.add_argument('--model', type=str, default='pointnet')
    parser.add_argument('--emd_coeff', type=int, default=10)
    parser.add_argument('--ablated', type=str, choices=['dc', 'emd'])

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)

    torch.random.manual_seed(args.random_seed)
    numpy.random.seed(args.random_seed)

    if args.aug_all:
        args.save_dir = os.path.join(args.save_dir, 'aug_all_{}_{}'.format(args.source_domain,
                                                                           args.target_domain))
    else:
        args.save_dir = os.path.join(args.save_dir,
                                     'emd{0:2.0f}_{1}_{2}_{3}_op{4}_ncv{5}_npy{6}_{7}'.format(args.emd_coeff,
                                                                                              args.model,
                                                                                              args.source_domain,
                                                                                              args.target_domain,
                                                                                              args.num_op, args.num_cv,
                                                                                              args.num_policy,
                                                                                              args.tag))
    logger = ExperimentLogger(args.save_dir, exist_ok=True)

    random.seed(int(args.tag))
    all_augment = [random.sample([(fn.__name__, random.random(), random.random()) for fn, v1, v2 in augment_list()], 3)]
    args.all_augment = all_augment
    logger.save_args(args)

    if args.only_eval:
        aug_eval.test(args, logger)
    else:
        aug_eval.train(args, logger)
        aug_eval.test(args, logger)
