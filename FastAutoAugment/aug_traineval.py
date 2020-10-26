import os
import torch
import numpy
from datetime import datetime
import argparse
import aug_eval
from utils.logger.ExperimentLogger import ExperimentLogger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='aug_trained', type=str)
    parser.add_argument('--data_dir', default='..', type=str)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--source_domain', type=str, choices=['modelnet', 'scannet', 'shapenet'])
    parser.add_argument('--target_domain', type=str, choices=['modelnet', 'scannet', 'shapenet'])
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=200)
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
    parser.add_argument('--tag', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--model', type=str, default='pointnet')

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)

    torch.random.manual_seed(args.random_seed)
    numpy.random.seed(args.random_seed)

    args.save_dir = os.path.join(args.save_dir,
                                 '{}_{}_op{}_ncv{}_npy{}_{}'.format(args.source_domain, args.target_domain,
                                                                    args.num_op, args.num_cv,
                                                                    args.num_policy, args.tag))
    logger = ExperimentLogger(args.save_dir, exist_ok=True)
    logger.save_args(args)

    aug_eval.train(args, logger)
    aug_eval.test(args, logger)
