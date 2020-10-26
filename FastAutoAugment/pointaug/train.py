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
from networks.PointNet import PointNetClassification
from utils.point_augmentations import apply_augment



def train(args, logger):
    total_epoch = args.epoch
    lr = 1e-3
    weight_decay = 5e-5
    batch_size = args.batch_size
    num_class = 10
    num_points = 1024

    aug_load = torch.load('aug_final/{}2{}.pth'.format(args.source_domain, args.target_domain))
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

    # Model
    model = PointNetClassification(10)
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
            test_model(model=model, dataloader=source_valloader, criterion=criterion, device=args.device)

        # Calculate metric
        val_loss = val_loss_sum / val_label_list.size(0)
        val_sample_accuracy, val_class_accuracy, val_accuracy_per_class = \
            metrics.calculate_accuracy_all(pred_list=val_pred_list, label_list=val_label_list, num_class=num_class)

        # Save Checkpoint
        save_dict = {
            'epoch': cur_epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
        logger.save_checkpoint(checkpoint=save_dict, tag=cur_epoch)
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

        logger.log_tensorlog(epoch=cur_epoch, log_dict={
            'Train/class_accuracy': train_class_accuracy,
            'Train/sample_accuracy': train_sample_accuracy,
            'Train/loss': train_loss,
            'Val/class_accuracy': val_class_accuracy,
            'Val/sample_accuracy': val_sample_accuracy,
            'Val/loss': val_loss,
            'Val/best_class_accuracy': source_best_val_class_acc,
            'Val/best_sample_accuracy': source_best_val_sample_acc,
        })
