import os
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from utils import metrics
from aug_eval.datasets import PointDA
from .module import AugmentationModule
from torch.utils.data import DataLoader
from .test import test_model
from .dc_test import dc_test_model
from .aug_test import aug_test_model
from networks.DGCNN.DGCNN import DGCNN as DGCNNClassification
from torch.optim.lr_scheduler import CosineAnnealingLR
from networks.PointNet import PointNetClassification, PointNetClassificationV7


def dc_train(args, logger):
    total_epoch = 50
    lr = 1e-3
    weight_decay = 5e-5
    batch_size = args.batch_size
    num_class = 2
    num_points = 1024

    domain_classifier = PointNetClassificationV7(2)
    domain_classifier = nn.DataParallel(domain_classifier)
    domain_classifier = domain_classifier.to(args.device)

    # Dataset
    source_trainset = PointDA(root=args.data_dir, domain=args.source_domain, partition='train',
                              num_points=num_points, sampling_method=args.source_train_sampling, download=True)
    source_valset = PointDA(root=args.data_dir, domain=args.source_domain, partition='val',
                            num_points=num_points, sampling_method=args.source_val_sampling, download=True)
    target_trainset = PointDA(root=args.data_dir, domain=args.target_domain, partition='train',
                              num_points=num_points, sampling_method=args.target_train_sampling, download=True)
    target_valset = PointDA(root=args.data_dir, domain=args.target_domain, partition='val',
                            num_points=num_points, sampling_method=args.target_val_sampling, download=True)

    logger.log_dataset(source_trainset)
    logger.log_dataset(target_trainset)
    logger.log_dataset(source_valset)
    logger.log_dataset(target_valset)

    # DataLoader
    source_trainloader = DataLoader(source_trainset, num_workers=args.num_workers,
                                    batch_size=batch_size, shuffle=True)
    source_valloader = DataLoader(source_valset, num_workers=args.num_workers,
                                  batch_size=batch_size)
    target_trainloader = DataLoader(target_trainset, num_workers=args.num_workers,
                                    batch_size=batch_size, shuffle=True)
    target_valloader = DataLoader(target_valset, num_workers=args.num_workers,
                                  batch_size=batch_size)

    # Optimizer / Scheduler / Loss
    optimizer = optim.Adam(domain_classifier.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, total_epoch)
    criterion = nn.CrossEntropyLoss()

    # Train
    source_best_val_sample_acc = source_best_val_class_acc = 0.0

    for cur_epoch in range(total_epoch):
        domain_classifier.train()

        train_pred_list = torch.zeros([0], dtype=torch.long).to(args.device)
        train_label_list = torch.zeros([0], dtype=torch.long).to(args.device)
        train_loss_sum = 0.0
        for src_data, trg_data in tqdm(zip(source_trainloader, target_trainloader)):
            optimizer.zero_grad()
            src_point_clouds = src_data['point_cloud'].to(args.device)
            trg_point_clouds = trg_data['point_cloud'].to(args.device)
            point_clouds = torch.cat([src_point_clouds, trg_point_clouds], dim=0)
            src_labels = torch.zeros_like(src_data['label'], dtype=torch.int64).to(args.device)
            trg_labels = torch.ones_like(trg_data['label'], dtype=torch.int64).to(args.device)
            labels = torch.cat([src_labels, trg_labels], dim=0)

            pred = domain_classifier(point_clouds)

            loss = criterion(pred, labels)
            loss.backward()

            optimizer.step()

            train_loss_sum += (loss.item() * labels.size(0))
            train_pred_list = torch.cat([train_pred_list, pred.max(dim=1)[1]], dim=0)
            train_label_list = torch.cat([train_label_list, labels], dim=0)
        scheduler.step()

        train_loss = train_loss_sum / train_label_list.size(0)
        train_sample_accuracy, train_class_accuracy, train_accuracy_per_class = \
            metrics.calculate_accuracy_all(pred_list=train_pred_list, label_list=train_label_list, num_class=num_class)

        # Validation
        val_pred_list, val_label_list, val_loss_sum = \
            dc_test_model(model=domain_classifier, dataloader=zip(source_valloader, target_valloader),
                          criterion=criterion, device=args.device)

        # Calculate metric
        val_loss = val_loss_sum / val_label_list.size(0)
        val_sample_accuracy, val_class_accuracy, val_accuracy_per_class = \
            metrics.calculate_accuracy_all(pred_list=val_pred_list, label_list=val_label_list, num_class=num_class)

        # Save Checkpoint
        save_dict = {
            'epoch': cur_epoch,
            'model': domain_classifier.module.state_dict(),
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
        # logger.save_checkpoint(checkpoint=save_dict, tag=cur_epoch)
        if val_sample_accuracy > source_best_val_sample_acc:
            source_best_val_sample_acc = val_sample_accuracy
            torch.save(save_dict, os.path.join(logger.save_root, '..', 'domain_classifier',
                                               '{}_{}.pth'.format(args.source_domain, args.target_domain)))

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
            'DC/Train/class_accuracy': train_class_accuracy,
            'DC/Train/sample_accuracy': train_sample_accuracy,
            'DC/Train/loss': train_loss,
            'DC/Val/class_accuracy': val_class_accuracy,
            'DC/Val/sample_accuracy': val_sample_accuracy,
            'DC/Val/loss': val_loss,
            'DC/Val/best_class_accuracy': source_best_val_class_acc,
            'DC/Val/best_sample_accuracy': source_best_val_sample_acc,
        })

    print('Training finished: domain classifier')
    del domain_classifier


def aug_train(args, logger):
    total_epoch = 50
    lr = 1e-3
    weight_decay = 5e-5
    batch_size = args.batch_size
    num_points = 1024
    num_class = 2

    aug_model = AugmentationModule()
    aug_model = nn.DataParallel(aug_model)
    aug_model = aug_model.to(args.device)

    domain_classifier = PointNetClassificationV7(2)
    if not os.path.isfile(os.path.join(logger.save_root, '..', 'domain_classifier',
                                       '{}_{}.pth'.format(args.source_domain, args.target_domain))):
        dc_train(args, logger)
    ckpt = torch.load(os.path.join(logger.save_root, '..', 'domain_classifier',
                                   '{}_{}.pth'.format(args.source_domain, args.target_domain)))
    print('Loaded: Domain classifier')
    domain_classifier.load_state_dict(ckpt['model'])
    domain_classifier = nn.DataParallel(domain_classifier)
    domain_classifier = domain_classifier.to(args.device).eval()

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

    # Optimizer / Scheduler / Loss
    optimizer = optim.Adam(aug_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, total_epoch)
    criterion = nn.CrossEntropyLoss()

    # Train
    source_best_val_sample_acc = source_best_val_class_acc = 0.0

    for cur_epoch in range(total_epoch):
        aug_model.train()

        train_pred_list = torch.zeros([0], dtype=torch.long).to(args.device)
        train_label_list = torch.zeros([0], dtype=torch.long).to(args.device)
        train_loss_sum = 0.0
        for data in tqdm(source_trainloader):
            optimizer.zero_grad()
            point_clouds = data['point_cloud'].to(args.device)
            point_clouds = aug_model(point_clouds)
            labels = torch.ones_like(data['label'], dtype=torch.int64).to(args.device)

            pred = domain_classifier(point_clouds)

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
            aug_test_model(model=domain_classifier, aug_model=aug_model, dataloader=source_valloader,
                           criterion=criterion, device=args.device)

        # Calculate metric
        val_loss = val_loss_sum / val_label_list.size(0)
        val_sample_accuracy, val_class_accuracy, val_accuracy_per_class = \
            metrics.calculate_accuracy_all(pred_list=val_pred_list, label_list=val_label_list, num_class=num_class)

        # Save Checkpoint
        save_dict = {
            'epoch': cur_epoch,
            'model': aug_model.module.state_dict(),
            'optimizer': optimizer,
            'scheduler': scheduler,
        }
        # logger.save_checkpoint(checkpoint=save_dict, tag=cur_epoch)
        if val_sample_accuracy > source_best_val_sample_acc:
            source_best_val_sample_acc = val_sample_accuracy
            torch.save(save_dict, os.path.join(logger.save_root, '..', 'pointaug',
                                               '{}_{}.pth'.format(args.source_domain, args.target_domain)))

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

    print("Training finished: Augmentation model")
    del domain_classifier
    del aug_model


def train(args, logger):
    total_epoch = args.epoch
    lr = 1e-3
    weight_decay = 5e-5
    batch_size = args.batch_size
    num_class = 10
    num_points = 1024

    aug_model = AugmentationModule()

    if not os.path.isfile(os.path.join(logger.save_root, '..', 'pointaug',
                                       '{}_{}.pth'.format(args.source_domain, args.target_domain))):
        aug_train(args, logger)
    ckpt = torch.load(
        os.path.join(logger.save_root, '..', 'pointaug', '{}_{}.pth'.format(args.source_domain, args.target_domain)))
    print("Loaded: augmentation model")
    aug_model.load_state_dict(ckpt['model'])
    aug_model = nn.DataParallel(aug_model)
    aug_model = aug_model.to(args.device).eval()
    total_epoch = 1

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
                point_clouds = aug_model(point_clouds)
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
