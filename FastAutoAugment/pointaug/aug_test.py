import torch


def aug_test_model(model, aug_model, dataloader, criterion, device):
    pred_list = torch.zeros([0], dtype=torch.long).to(device)
    label_list = torch.zeros([0], dtype=torch.long).to(device)
    loss_sum = 0.0
    with torch.no_grad():
        aug_model.eval()
        model.eval()
        for data in dataloader:
            point_clouds = data['point_cloud'].to(device)
            point_clouds = aug_model(point_clouds)
            labels = torch.ones_like(data['label'], dtype=torch.int64).to(device)

            pred = model(point_clouds)

            loss = criterion(pred, labels)

            loss_sum += (loss.item() * labels.size(0))
            pred_list = torch.cat([pred_list, pred.max(dim=1)[1]], dim=0)
            label_list = torch.cat([label_list, labels], dim=0)

    return pred_list, label_list, loss_sum
