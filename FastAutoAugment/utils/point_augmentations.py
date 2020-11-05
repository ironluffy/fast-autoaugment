import torch
import random
import numpy as np
import utils.point_cloud_utils as pcu
from utils.emd.emd_module import emdModule


def random_crop_plane(point_clouds, level=0.6, random_range=0.1):
    B, C, N = point_clouds.shape
    device = point_clouds.device
    level = 1.0 - level
    min_rad = level * (1 - random_range)
    cropped_point_clouds = torch.zeros([0], dtype=torch.float32, device=device)

    for idx in range(B):
        point_cloud = point_clouds[idx, :].unsqueeze(0)
        rad = np.random.uniform(min_rad, min_rad + random_range)
        dir_vec = torch.rand([1, 3, 1]).to(device)
        pos_vec = (dir_vec / pcu.get_distance(dir_vec).unsqueeze(dim=2) * rad)
        threshold_val = (point_cloud - pos_vec).permute(0, 2, 1).bmm(dir_vec)

        point_cloud = point_cloud[:, :, threshold_val.squeeze() < 0]

        new_point_cloud = point_cloud
        while new_point_cloud.size(2) < N:
            if new_point_cloud.size(2) == 0:
                new_point_cloud = point_clouds[idx, :].unsqueeze(0)
                break
            new_point_cloud = torch.cat([new_point_cloud, pcu.jitter(point_cloud)], dim=2)
        new_point_cloud, _ = pcu.random_point_sample(new_point_cloud, N)
        cropped_point_clouds = torch.cat([cropped_point_clouds, new_point_cloud], dim=0)

    cropped_point_clouds = pcu.normalize(cropped_point_clouds)

    return cropped_point_clouds


def random_crop_sphere(point_clouds, level=0.6, random_range=0.1):
    B, C, N = point_clouds.shape
    device = point_clouds.device
    min_rad = level * (0.8 - random_range)
    cropped_point_clouds = torch.zeros([0], dtype=torch.float32, device=device)

    for idx in range(B):
        point_cloud = point_clouds[idx, :].unsqueeze(0)
        iter = 0
        while True:
            if iter > 10:
                new_point_cloud = point_cloud
                break
            else:
                iter += 1
            random_index = np.random.randint(N)
            view_point = point_cloud[:, :, random_index].unsqueeze(2)
            distance = pcu.get_distance(point_cloud.sub(view_point))

            temp_point_cloud = point_cloud.transpose(2, 1)
            max_rad = min_rad + random_range
            mask_f = (distance > np.random.uniform(min_rad, max_rad))

            temp_point_cloud = temp_point_cloud[mask_f, :].unsqueeze(0).transpose(2, 1)
            new_point_cloud = temp_point_cloud

            if new_point_cloud.size(2) > 50:
                break

        while new_point_cloud.size(2) < N:
            new_point_cloud = torch.cat([new_point_cloud, pcu.jitter(temp_point_cloud)], dim=2)

        new_point_cloud, _ = pcu.random_point_sample(new_point_cloud, N)
        cropped_point_clouds = torch.cat([cropped_point_clouds, new_point_cloud], dim=0)

    cropped_point_clouds = pcu.normalize(cropped_point_clouds)

    return cropped_point_clouds


def random_crop_sphere_reverse(point_clouds, level=0.6, random_range=0.1):
    B, C, N = point_clouds.shape
    device = point_clouds.device
    level = 1.0 - level
    min_rad = level * (1 - (random_range + 0.1)) + 0.1
    cropped_point_clouds = torch.zeros([0], dtype=torch.float32, device=device)

    for idx in range(B):
        point_cloud = point_clouds[idx, :].unsqueeze(0)
        iter = 0
        while True:
            if iter > 10:
                new_point_cloud = point_cloud
                break
            else:
                iter += 1
            random_index = np.random.randint(N)
            view_point = point_cloud[:, :, random_index].unsqueeze(2)
            distance = pcu.get_distance(point_cloud.sub(view_point))

            temp_point_cloud = point_cloud.transpose(2, 1)
            max_rad = min_rad + random_range
            mask_f = (distance < np.random.uniform(min_rad, max_rad))

            temp_point_cloud = temp_point_cloud[mask_f, :].unsqueeze(0).transpose(2, 1)
            new_point_cloud = temp_point_cloud

            if new_point_cloud.size(2) > 50:
                break

        while new_point_cloud.size(2) < N:
            new_point_cloud = torch.cat([new_point_cloud, pcu.jitter(temp_point_cloud)], dim=2)

        new_point_cloud, _ = pcu.random_point_sample(new_point_cloud, N)
        cropped_point_clouds = torch.cat([cropped_point_clouds, new_point_cloud], dim=0)

    cropped_point_clouds = pcu.normalize(cropped_point_clouds)

    return cropped_point_clouds


def naive_point_mix_up(point_clouds, level=1.0, random_range=0.3):
    B, _, N = point_clouds.shape
    device = point_clouds.device

    shuffled_index = torch.randperm(B).to(device)
    shuffled_pc = point_clouds[shuffled_index, :]

    lam = (1 - random_range) * level + torch.FloatTensor(B).uniform_(0, random_range)
    lam = lam.to(device)

    num_pts_a = N - torch.round(lam * N)
    num_pts_b = torch.round(lam * N)

    mixed_point_clouds = torch.zeros([0], dtype=torch.float32).to(device)
    for idx in range(B):
        if num_pts_a[idx] == 0 or num_pts_b[idx] == 0:
            mixed_pc = point_clouds[idx].unsqueeze(0)
            mixed_point_clouds = torch.cat([mixed_point_clouds, mixed_pc], dim=0)
            continue

        point_cloud_a, _ = pcu.random_point_sample(point_clouds[idx, :].unsqueeze(0), int(num_pts_a[idx]))
        point_cloud_b, _ = pcu.random_point_sample(shuffled_pc[idx, :].unsqueeze(0), int(num_pts_b[idx]))
        mixed_pc = torch.cat((point_cloud_a, point_cloud_b), 2)

        points_perm = torch.randperm(N).to(device)
        mixed_pc = mixed_pc[:, :, points_perm]

        mixed_point_clouds = torch.cat([mixed_point_clouds, mixed_pc], dim=0)

    # label_a = labels.clone()
    # label_b = labels[shuffled_index].clone()

    mixed_point_clouds = pcu.normalize(mixed_point_clouds)

    return mixed_point_clouds  # , (label_a, label_b, lam)


def point_mix_up(point_clouds, level=1.0, random_range=0.1):
    emd_loss2 = emdModule()
    B, _, N = point_clouds.shape
    device = point_clouds.device

    shuffled_index = torch.randperm(B).to(device)
    shuffled_point_clouds = point_clouds[shuffled_index, :].to(device)

    lam = (1 - random_range) * level + torch.FloatTensor(B).uniform_(0, random_range)
    lam = lam.to(device)
    emd_loss, emd_matching_idx = emd_loss2(point_clouds.permute(0, 2, 1),
                                           shuffled_point_clouds.permute(0, 2, 1), 0.05, 3000)

    emd_matching_idx = emd_matching_idx.type(torch.LongTensor).to(device)
    matching_point_clouds = torch.gather(shuffled_point_clouds, 2, torch.stack([emd_matching_idx] * 3, dim=1))

    mixup_point_cloud = matching_point_clouds * lam.view(-1, 1, 1) + point_clouds * (1 - lam).view(-1, 1, 1)

    # label_a = labels.clone()
    # label_b = labels[shuffled_index].clone()

    mixup_point_cloud = pcu.normalize(mixup_point_cloud)

    return mixup_point_cloud  # , (label_a, label_b, lam)


def point_mix_up_random_access(point_clouds, level=1.0, random_range=0.1):
    B, _, N = point_clouds.shape
    device = point_clouds.device

    shuffled_index = torch.randperm(B).to(device)
    shuffled_point_clouds = point_clouds[shuffled_index, :].to(device)

    lam = (1 - random_range) * level + torch.FloatTensor(B).uniform_(0, random_range)
    lam = lam.to(device)

    mixup_point_cloud = shuffled_point_clouds * lam.view(-1, 1, 1) + point_clouds * (1 - lam).view(-1, 1, 1)

    # label_a = labels.clone()
    # label_b = labels[shuffled_index].clone()

    mixup_point_cloud = pcu.normalize(mixup_point_cloud)

    return mixup_point_cloud  # , (label_a, label_b, lam)


def squeeze_z(point_clouds, level=0.6, random_range=0.1):
    B, C, N = point_clouds.shape
    device = point_clouds.device
    level = 1.0 - level
    min_squeeze_param = level * (1 - random_range)
    squeeze_param = torch.FloatTensor(B, 1).uniform_(min_squeeze_param, min_squeeze_param + random_range)

    scale_vector = torch.cat([torch.ones(point_clouds.size(0), 1),
                              torch.ones(point_clouds.size(0), 1), squeeze_param], dim=1).to(device)

    squeeze_point_cloud = point_clouds * scale_vector.unsqueeze(-1)

    squeeze_point_cloud = pcu.normalize(squeeze_point_cloud)

    return squeeze_point_cloud


def squeeze_xy(point_clouds, level=0.6, random_range=0.1):
    B, C, N = point_clouds.shape
    device = point_clouds.device
    level = 1.0 - level
    min_squeeze_param = level * (1 - random_range)
    squeeze_param = torch.FloatTensor(B, 1).uniform_(min_squeeze_param, min_squeeze_param + random_range)
    random_theta = torch.FloatTensor(B, 1).uniform_(0, 2 * np.pi)
    scale_vector = torch.cat([torch.cos(random_theta) * squeeze_param,
                              torch.sin(random_theta) * squeeze_param,
                              torch.ones(B, 1)], dim=1).to(device)

    squeeze_point_cloud = point_clouds * scale_vector.unsqueeze(-1)
    squeeze_point_cloud = pcu.normalize(squeeze_point_cloud)

    return squeeze_point_cloud


def squeeze_sphere(point_clouds, level=0.6, random_range=0.1):
    B, C, N = point_clouds.shape
    device = point_clouds.device
    level = 1.0 - level
    min_squeeze_param = level * (1 - random_range)
    squeeze_ratio = torch.FloatTensor(B, 1, 1).uniform_(min_squeeze_param, min_squeeze_param + random_range).to(device)
    viewpoint_rad = torch.FloatTensor(B, 1, 1).uniform_().to(device) * (1 - squeeze_ratio)

    phi = torch.FloatTensor(B, 1).uniform_(0, np.pi * 2)
    theta = torch.FloatTensor(B, 1).uniform_(0, np.pi)

    squeezed_point_clouds = point_clouds * squeeze_ratio

    view_point = torch.cat([torch.sin(theta) * torch.cos(phi),
                            torch.sin(theta) * torch.sin(phi),
                            torch.cos(theta)], dim=1).unsqueeze(-1).to(device)
    view_point = view_point * viewpoint_rad

    augmented_point_clouds = squeezed_point_clouds + view_point

    return augmented_point_clouds


def sparse(point_clouds, level=0.6, random_range=0.1):
    B, C, N = point_clouds.shape
    device = point_clouds.device
    min_rad = level * (1 - random_range)
    augmented_point_clouds = torch.zeros([0], dtype=torch.float32, device=device)

    for idx in range(B):
        point_cloud = point_clouds[idx, :].unsqueeze(0)
        random_index = np.random.randint(N)
        view_point = point_cloud[:, :, random_index].unsqueeze(2)
        distance = pcu.get_distance(point_cloud.sub(view_point))

        max_rad = min_rad + random_range
        mask_f = (distance < np.random.uniform(min_rad, max_rad))

        part_point_cloud = point_cloud.transpose(2, 1)
        part_point_cloud = part_point_cloud[mask_f, :].unsqueeze(0).transpose(2, 1)
        part_point_cloud = pcu.jitter(part_point_cloud, sigma=0.02, clip=0.05)

        if part_point_cloud.size(2) == N:
            augmented_point_clouds = torch.cat([augmented_point_clouds, part_point_cloud], dim=0)
        else:
            point_cloud, _ = pcu.random_point_sample(point_cloud, num_points=N - part_point_cloud.size(2))
            point_cloud = torch.cat([point_cloud, part_point_cloud], dim=2)
            point_cloud = pcu.point_permutate(point_cloud)
            augmented_point_clouds = torch.cat([augmented_point_clouds, point_cloud], dim=0)

    augmented_point_clouds = pcu.normalize(augmented_point_clouds)

    return augmented_point_clouds


def global_sparse(point_clouds, level=0.6, random_range=0.1):
    B, C, N = point_clouds.shape
    device = point_clouds.device
    min_param = level * (0.9 - random_range) + 0.1
    augmented_point_clouds = torch.zeros([0], dtype=torch.float32, device=device)

    lam = torch.FloatTensor(B, 1).uniform_(min_param, min_param + random_range).to(device)
    num_pts_a = N - torch.round(lam * N)
    num_pts_b = torch.round(lam * N)

    for idx in range(B):
        if num_pts_a[idx] == N:
            point_cloud = point_clouds[idx, :]
            augmented_point_clouds = torch.cat([augmented_point_clouds, point_cloud], dim=0)
            continue

        temp_point_cloud = torch.zeros([0], dtype=torch.float32, device=device)
        part_point_cloud, _ = pcu.random_point_sample(point_clouds[idx, :].unsqueeze(0), int(num_pts_a[idx]))

        while True:
            temp_point_cloud = torch.cat([temp_point_cloud, part_point_cloud], dim=2)
            if temp_point_cloud.size(-1) > num_pts_b[idx]:
                break

        temp_point_cloud, _ = pcu.random_point_sample(temp_point_cloud, int(num_pts_b[idx]))
        point_cloud = torch.cat([temp_point_cloud, part_point_cloud], dim=2)
        point_cloud = pcu.point_permutate(point_cloud)
        augmented_point_clouds = torch.cat([augmented_point_clouds, point_cloud], dim=0)

    augmented_point_clouds = pcu.normalize(augmented_point_clouds)

    return augmented_point_clouds


def equalize(point_clouds, level):
    return point_clouds


def mix_up_loss(pred, mixup_label, criterion):
    # Use criterion without reduction (e.g) nn.CrossEntropyLoss(reduction='none'))
    mixup_label = mixup_label.to(pred.device)
    label_a, label_b, lam = mixup_label
    loss = lam * criterion(pred, label_a) + (1 - lam) * criterion(pred, label_b)

    return loss


def augment_list(for_autoaug=True):  # 16 oeprations and their ranges
    l = [
        (random_crop_plane, 0, 1.0),
        (random_crop_sphere, 0, 1.0),
        (random_crop_sphere_reverse, 0, 1.0),
        (naive_point_mix_up, 0, 1.0),
        (point_mix_up, 0, 1.0),
        (squeeze_xy, 0, 1.0),
        (squeeze_z, 0, 1.0),
        (squeeze_sphere, 0, 1.0),
        (equalize, 0, 1.0),
        (sparse, 0, 1.0),
        (global_sparse, 0, 1.0)
    ]
    if for_autoaug:
        l += [
        ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}
all_augment = [[(fn, random.random(), random.random()) for fn, v1, v2 in augment_list()]]


def get_augment(name):
    return augment_dict[name]


def apply_augment(pc, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(pc, level * (high - low) + low)
