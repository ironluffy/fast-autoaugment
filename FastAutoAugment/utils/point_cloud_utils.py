import torch
import numpy as np


def get_distance(point_cloud):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :return: (B, N) Distance of each point in the point   cloud.
    """
    distance = point_cloud.pow(2).sum(axis=1).sqrt()
    return distance


def normalize(point_cloud):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :return: (B, C, N) Normalized point cloud.
    """
    if point_cloud.size(2) == 0:
        return point_cloud
    centroid = point_cloud.mean(axis=2, keepdims=True)
    point_cloud = point_cloud - centroid
    farthest_distance = get_distance(point_cloud).unsqueeze(dim=1).max(axis=2, keepdim=True)[0]
    point_cloud = point_cloud / farthest_distance
    return point_cloud


def farthest_point_sample(point_cloud, num_points, seed=None):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :param num_points: Number of samples.
    :returns: (B, C, num_points) Sampled point cloud. /
                  (B, num_points) Sampled point cloud indices.
    """
    device = point_cloud.device
    B, C, N = point_cloud.shape
    if N <= num_points:
        return point_cloud
    centroid_indices = torch.zeros(B, num_points, dtype=torch.long).to(device)
    distance_dp = torch.ones(B, N).to(device) * 1e10
    if seed is not None:
        np.random.seed(seed=seed)
    farthest = torch.tensor(np.random.randint(0, N, (B,))).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(num_points):
        centroid_indices[:, i] = farthest
        centroid = point_cloud[batch_indices, :, farthest].unsqueeze(dim=2)
        distance = get_distance(point_cloud - centroid)
        mask = distance < distance_dp
        distance_dp[mask] = distance[mask]
        farthest = distance_dp.max(1)[1]
    batch_indices = torch.stack([batch_indices] * num_points, dim=1)
    centroids = point_cloud[batch_indices, :, centroid_indices].permute(0, 2, 1)
    return centroids, centroid_indices


def random_point_sample(point_cloud, num_points, seed=None):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :param num_points: Number of samples.
    :param seed: if you want fixed, set seed.
    :returns: (B, C, num_points) Sampled point cloud. /
                  (B, num_points) Sampled point cloud indices.
    """
    device = point_cloud.device
    B, C, N = point_cloud.shape
    if N <= num_points:
        return point_cloud, None
    if seed is not None:
        np.random.seed(seed=seed)
    random_sample_indices = torch.tensor([np.random.choice(N, num_points, replace=False) for b in range(B)]).to(device)
    if B == 0:
        print(point_cloud.size())
    batch_indices = torch.stack([torch.arange(B)] * num_points, dim=1).to(device)
    random_samples = point_cloud[batch_indices, :, random_sample_indices].permute(0, 2, 1)
    return random_samples, random_sample_indices


def fixed_point_sample(point_cloud, num_points, seed=0):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :param num_points: Number of samples.
    :param seed: if you want fixed, set seed.
    :returns: (B, C, num_points) Sampled point cloud. /
                  (B, num_points) Sampled point cloud indices.
    """
    fixed_samples, fixed_sample_indices = random_point_sample(point_cloud, num_points, seed)
    return fixed_samples, fixed_sample_indices


def random_rotate_one_axis(point_cloud, axis):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :param axis: Axis to do random rotation.
    :return: (B, C, N) Rotated point cloud.
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    return rotate_shape(point_cloud, axis, rotation_angle)


def jitter(point_cloud, sigma=0.01, clip=0.02):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :param sigma:
    :param clip:
    :return: (B, C, N) Jittered point cloud.
    """
    random_noise = torch.clamp(sigma * torch.randn(point_cloud.shape).to(device=point_cloud.device), -clip, clip)
    point_cloud = point_cloud + random_noise
    return point_cloud


def rotate_shape(point_cloud, axis, rotation_angle):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :param axis: Axis to do random rotation.
    :param rotation_angle: Rotation angle.
    :return: Rotated point cloud.
    """
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    point_cloud = point_cloud.transpose(2, 1)
    if axis == 'x':
        R_x = torch.tensor([[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]], dtype=torch.float32)
        R_x = R_x.to(device=point_cloud.device)
        point_cloud = torch.matmul(point_cloud, R_x)
    elif axis == 'y':
        R_y = torch.tensor([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]], dtype=torch.float32)
        R_y = R_y.to(device=point_cloud.device)
        point_cloud = torch.matmul(point_cloud, R_y)
    elif axis == 'z':
        R_z = torch.tensor([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]], dtype=torch.float32)
        R_z = R_z.to(device=point_cloud.device)
        point_cloud = torch.matmul(point_cloud, R_z)
    else:
        raise NotImplementedError
    point_cloud = point_cloud.transpose(2, 1)
    return point_cloud


def rotate_shape_grad(point_cloud, axis, rotation_angle):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :param axis: Axis to do random rotation.
    :param rotation_angle: Rotation angle.
    :return: Rotated point cloud.
    """
    cosval = torch.cos(rotation_angle)
    sinval = torch.sin(rotation_angle)
    point_cloud = point_cloud.transpose(2, 1)
    if axis == 'x':
        R_x = torch.tensor([[1, 0, 0], [0, cosval, -sinval], [0, sinval, cosval]], dtype=torch.float32)
        R_x = R_x.to(device=point_cloud.device)
        point_cloud = torch.matmul(point_cloud, R_x)
    elif axis == 'y':
        R_y = torch.tensor([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]], dtype=torch.float32)
        R_y = R_y.to(device=point_cloud.device)
        point_cloud = torch.matmul(point_cloud, R_y)
    elif axis == 'z':
        R_z = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 1]]] * rotation_angle.size(0), dtype=torch.float32,
                           device=point_cloud.device)
        R_z[:, 0, 0:1] = cosval
        R_z[:, 0, 1:2] = -sinval
        R_z[:, 1, 0:1] = sinval
        R_z[:, 1, 1:2] = cosval
        point_cloud = torch.matmul(point_cloud, R_z)
    else:
        raise NotImplementedError
    point_cloud = point_cloud.transpose(2, 1)
    return point_cloud


def point_permutate(point_cloud):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :return: Permutated point cloud.
    """
    B, C, N = point_cloud.shape
    indicies = torch.randperm(N)
    point_cloud = point_cloud[:, :, indicies]
    return point_cloud


def occlude_sampling(point_cloud, num_points=1024, rad=0.1, seed=None):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :param num_points: Number of samples.
    :param rad: cutting radius.
    :return: Occluded point cloud.
    """
    if point_cloud.size(0) != 1:
        raise NotImplementedError
    device = point_cloud.device
    dir_vec = torch.rand([1, 3, 1]).to(device)
    pos_vec = (dir_vec / get_distance(dir_vec).unsqueeze(dim=2) * rad)
    threshold_val = (point_cloud - pos_vec).permute(0, 2, 1).bmm(dir_vec)
    point_cloud = point_cloud[:, :, threshold_val.squeeze() < 0]
    new_point_cloud = point_cloud
    while new_point_cloud.size(2) < num_points:
        new_point_cloud = torch.cat([new_point_cloud, jitter(point_cloud)], dim=2)
    return new_point_cloud


def occlude_sampling_viewpoint(point_cloud, num_points=1024, viewpoint_rad=0.7, occlude_range=0.3, min_rad=0.3,
                               reverse=True, occlude_type='random'):
    """
    :param point_cloud: (B, C, N) Point cloud data.
    :param num_points: Number of samples.
    :param viewpoint_rad: viewpoint space.
    :param occlude_range: random occlude radius range.
    :param min_rad: min value of occlude radius.
    :param reverse: masking inside(F)/outside(T) of occlude sphere
    :param occlude_type: random / viewpoint
    :return: Occluded point cloud.
    """
    if point_cloud.size(0) != 1:
        raise NotImplementedError
    while (1):
        if occlude_type == 'viewpoint':
            phi = np.random.uniform(0, np.pi * 2)
            theta = np.random.uniform(0, np.pi * 2)
            view_point = torch.Tensor([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]) \
                         * viewpoint_rad
            view_point = view_point.unsqueeze(0).unsqueeze(2)
            distance = get_distance(point_cloud.sub(view_point))
        elif occlude_type == 'random':
            random_index = np.random.randint(point_cloud.size(2))
            view_point = point_cloud[:, :, random_index].unsqueeze(2)
            distance = get_distance(point_cloud.sub(view_point))
        else:
            raise NotImplementedError
        temp_point_cloud = point_cloud.transpose(2, 1)
        max_rad = min_rad + occlude_range
        if not reverse:
            mask_f = (distance > np.random.uniform(min_rad, max_rad))
        else:
            mask_f = (distance < np.random.uniform(min_rad, max_rad))
        temp_point_cloud = temp_point_cloud[mask_f, :].unsqueeze(0).transpose(2, 1)
        new_point_cloud = temp_point_cloud
        if new_point_cloud.size(2) > 100:
            break
    while new_point_cloud.size(2) < num_points:
        new_point_cloud = torch.cat([new_point_cloud, jitter(temp_point_cloud)], dim=2)
    new_point_cloud = normalize(new_point_cloud)
    return new_point_cloud
