import torch
import torch.nn as nn

from datasets import PointDA
from utils import metrics
from torch.utils.data import DataLoader
from networks.PointNet import PointNetClassification, PointNetClassificationV7


def dc_test(args, logger):
    # Logger
    logger.change_log_file('test.log')

    # Const
    batch_size = args.batch_size
    num_class = 2
    num_points = 1024

    # Dataset
    source_testset = PointDA(root=args.data_dir, domain=args.source_domain, partition='test',
                             num_points=num_points, sampling_method=args.source_test_sampling, download=True)
    target_testset = PointDA(root=args.data_dir, domain=args.target_domain, partition='test',
                             num_points=num_points, sampling_method=args.target_test_sampling, download=True)
    logger.log_dataset(source_testset)
    logger.log_dataset(target_testset)

    # DataLoader
    source_testloader = DataLoader(source_testset, num_workers=args.num_workers,
                                   batch_size=batch_size)
    target_testloader = DataLoader(target_testset, num_workers=args.num_workers,
                                   batch_size=batch_size)

    # Model
    model = PointNetClassificationV7(2)
    model = nn.DataParallel(model)
    model = model.to(args.device)
    logger.log_model_architecture(model)

    criterion = nn.CrossEntropyLoss()

    # ===Best Class Acc===
    # Model Load
    load_dict = logger.load_checkpoint('Best_class_val_dc')
    model.module.load_state_dict(load_dict['model'])

    # Test Model
    test_pred_list, test_label_list, test_loss_sum = \
        test_model(model=model, dataloader=zip(source_testloader, target_testloader), criterion=criterion, device=args.device)

    # Calculate metric
    test_loss = test_loss_sum / test_label_list.size(0)
    test_sample_accuracy, test_class_accuracy, test_accuracy_per_class = \
        metrics.calculate_accuracy_all(pred_list=test_pred_list, label_list=test_label_list, num_class=num_class)

    # Log
    logger.log.write('Test on Best Class ACC')
    logger.log_classification(partition='test', loss=test_loss,
                              class_accuracy=test_class_accuracy,
                              sample_accuracy=test_sample_accuracy,
                              accuracy_per_class=test_accuracy_per_class)

    # ===Best Sample Acc===
    # Model Load
    load_dict = logger.load_checkpoint('Best_sample_val_dc')
    model.module.load_state_dict(load_dict['model'])

    # Test Model
    test_pred_list, test_label_list, test_loss_sum = \
        dc_test_model(model=model, dataloader=zip(source_testloader, target_testloader), criterion=criterion, device=args.device)

    # Calculate metric
    test_loss = test_loss_sum / test_label_list.size(0)
    test_sample_accuracy, test_class_accuracy, test_accuracy_per_class = \
        metrics.calculate_accuracy_all(pred_list=test_pred_list, label_list=test_label_list, num_class=num_class)

    # Log
    logger.log.write('Test on Best Sample ACC')
    logger.log_classification(partition='test', loss=test_loss,
                              class_accuracy=test_class_accuracy,
                              sample_accuracy=test_sample_accuracy,
                              accuracy_per_class=test_accuracy_per_class)


def dc_test_model(model, dataloader, criterion, device):
    pred_list = torch.zeros([0], dtype=torch.long).to(device)
    label_list = torch.zeros([0], dtype=torch.long).to(device)
    loss_sum = 0.0
    with torch.no_grad():
        model.eval()
        for src_data, trg_data in dataloader:
            src_point_clouds = src_data['point_cloud'].to(device)
            trg_point_clouds = trg_data['point_cloud'].to(device)
            point_clouds = torch.cat([src_point_clouds, trg_point_clouds], dim=0)
            src_labels = torch.zeros_like(src_data['label'], dtype=torch.int64).to(device)
            trg_labels = torch.ones_like(trg_data['label'], dtype=torch.int64).to(device)
            labels = torch.cat([src_labels, trg_labels], dim=0)

            pred = model(point_clouds)

            loss = criterion(pred, labels)

            loss_sum += (loss.item() * labels.size(0))
            pred_list = torch.cat([pred_list, pred.max(dim=1)[1]], dim=0)
            label_list = torch.cat([label_list, labels], dim=0)

    return pred_list, label_list, loss_sum