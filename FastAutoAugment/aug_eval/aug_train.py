import os
import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from aug_eval.datasets import PointDA
from aug_eval.aug_test import test_model
from utils.point_augmentations import apply_augment, all_augment
from utils import metrics

from networks.PointNet import PointNetClassification
from networks.DGCNN.DGCNN import DGCNN as DGCNNClassification


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, pnt):
        org_size = pnt.size(0)
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                pnt = apply_augment(pnt, name, level)
        return pnt


def train(args, logger):
    total_epoch = args.epoch
    lr = 1e-3
    weight_decay = 5e-5
    batch_size = args.batch_size
    num_class = 10
    num_points = 1024

    if os.path.isfile(
            'aug_final/{}_{}2{}_op{}_ncv{}_npy{}_ns{}.pth'.format(args.dc_model, args.source_domain, args.target_domain,
                                                                  args.num_op,
                                                                  args.num_cv,
                                                                  args.num_policy, args.num_search)):
        aug_load = torch.load(
            'aug_final/{}_{}2{}_op{}_ncv{}_npy{}_ns{}.pth'.format(args.dc_model, args.source_domain, args.target_domain,
                                                                  args.num_op,
                                                                  args.num_cv,
                                                                  args.num_policy, args.num_search))
    else:
        aug_load = torch.load(
            'aug_final/{}_{}2{}_op{}_ncv{}_npy{}.pth'.format(args.dc_model, args.source_domain, args.target_domain,
                                                             args.num_op,
                                                             args.num_cv,
                                                             args.num_policy))

    if args.aug_all:
        aug = all_augment
    else:
        aug = Augmentation(aug_load)

    # Dataset
    source_trainset = PointDA(root=args.data_dir, domain=args.source_domain, partition='train',
                              num_points=num_points, sampling_method=args.source_train_sampling, download=True)
    source_valset = PointDA(root=args.data_dir, domain=args.source_domain, partition='val',
                            num_points=num_points, sampling_method=args.source_val_sampling, download=True)
    logger.log_dataset(source_trainset)
    logger.log_dataset(source_valset)

    # DataLoader
    source_trainloader = DataLoader(source_trainset, num_workers=args.num_workers,
                                    batch_size=batch_size, shuffle=True)
    source_valloader = DataLoader(source_valset, num_workers=args.num_workers,
                                  batch_size=batch_size)

    target_testset = PointDA(root=args.data_dir, domain=args.target_domain, partition='test',
                             num_points=num_points, sampling_method=args.target_test_sampling, download=True)
    # DataLoader
    target_testloader = DataLoader(target_testset, num_workers=args.num_workers,
                                   batch_size=batch_size)

    # Model
    if args.model == 'dgcnn':
        model = DGCNNClassification()
        torch.backends.cudnn.enabled = False
    elif args.model == 'pointnet':
        model = PointNetClassification()
    else:
        raise NotImplementedError
    model = nn.DataParallel(model)
    model = model.to(args.device)
    logger.log_model_architecture(model)

    # Optimizer / Scheduler / Loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, total_epoch)
    criterion = nn.CrossEntropyLoss()

    # Train
    source_best_val_sample_acc = source_best_val_class_acc = 0.0

    for cur_epoch in range(total_epoch):
        model.train()

        train_pred_list = torch.zeros([0], dtype=torch.long).to(args.device)
        train_label_list = torch.zeros([0], dtype=torch.long).to(args.device)
        train_loss_sum = 0.0
        for data in tqdm(source_trainloader):
            optimizer.zero_grad()
            point_clouds = data['point_cloud'].to(args.device)
            with torch.no_grad():
                point_clouds = aug(point_clouds)
            labels = data['label'].to(args.device)

            pred = model(point_clouds)

            loss = criterion(pred, labels)
            loss.backward()

            optimizer.step()

            train_loss_sum += (loss.item() * labels.size(0))
            train_pred_list = torch.cat([train_pred_list, pred.max(dim=1)[1]], dim=0)
            train_label_list = torch.cat([train_label_list, labels], dim=0)
        scheduler.step()

        # Calculate metric
        train_loss = train_loss_sum / train_label_list.size(0)
        train_sample_accuracy, train_class_accuracy, train_accuracy_per_class = \
            metrics.calculate_accuracy_all(pred_list=train_pred_list, label_list=train_label_list, num_class=num_class)

        # Validation
        val_pred_list, val_label_list, val_loss_sum = \
            test_model(model=model, aug_model=aug, dataloader=source_valloader, criterion=criterion, device=args.device)

        trg_test_pred_list, trg_test_label_list, trg_test_loss_sum = \
            test_model(model=model, aug_model=None, dataloader=target_testloader, criterion=criterion,
                       device=args.device)

        # Calculate metric
        val_loss = val_loss_sum / val_label_list.size(0)
        val_sample_accuracy, val_class_accuracy, val_accuracy_per_class = \
            metrics.calculate_accuracy_all(pred_list=val_pred_list, label_list=val_label_list, num_class=num_class)

        trg_test_loss = trg_test_loss_sum / val_label_list.size(0)
        trg_test_sample_accuracy, trg_test_class_accuracy, trg_test_accuracy_per_class = \
            metrics.calculate_accuracy_all(pred_list=trg_test_pred_list, label_list=trg_test_label_list,
                                           num_class=num_class)

        # Save Checkpoint
        save_dict = {
            'epoch': cur_epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
        # logger.save_checkpoint(checkpoint=save_dict, tag=cur_epoch)
        if val_sample_accuracy > source_best_val_sample_acc:
            source_best_val_sample_acc = val_sample_accuracy
            logger.save_checkpoint(checkpoint=save_dict, tag='Best_sample_val')
        if val_class_accuracy > source_best_val_class_acc:
            source_best_val_class_acc = val_class_accuracy
            logger.save_checkpoint(checkpoint=save_dict, tag='Best_class_val')

        # Logging
        logger.log_epoch(epoch=cur_epoch, total_epoch=total_epoch)
        logger.log_classification(partition='train', loss=train_loss,
                                  class_accuracy=train_class_accuracy,
                                  sample_accuracy=train_sample_accuracy,
                                  accuracy_per_class=train_accuracy_per_class)
        logger.log_classification(partition='val', loss=val_loss,
                                  class_accuracy=val_class_accuracy,
                                  sample_accuracy=val_sample_accuracy,
                                  accuracy_per_class=val_accuracy_per_class)

        logger.log_classification(partition='test', loss=trg_test_loss,
                                  class_accuracy=trg_test_class_accuracy,
                                  sample_accuracy=trg_test_sample_accuracy,
                                  accuracy_per_class=trg_test_accuracy_per_class)

        logger.log_tensorlog(epoch=cur_epoch, log_dict={
            'Train/class_accuracy': train_class_accuracy,
            'Train/sample_accuracy': train_sample_accuracy,
            'Train/loss': train_loss,
            'Val/class_accuracy': val_class_accuracy,
            'Val/sample_accuracy': val_sample_accuracy,
            'Val/loss': val_loss,
            'TT/class_accuracy': trg_test_class_accuracy,
            'TT/sample_accuracy': trg_test_sample_accuracy,
            'TT/loss': trg_test_loss,
            'Val/best_class_accuracy': source_best_val_class_acc,
            'Val/best_sample_accuracy': source_best_val_sample_acc,
        })
