import torch
import open3d
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
import imageio
from torch.autograd import Variable
from src.utils import point_cloud_utils as pcu
from sklearn.metrics import confusion_matrix


def weights_init(m):
    classname = m.__class__.__name__
    if classname in ('Conv1d', 'Linear'):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def pc_to_grid(point_cloud, grid_rate):
    B, C, N = point_cloud.shape
    device = point_cloud.device
    grid_pc = point_cloud.to(device)
    for c in range(C):
        point_matrix = grid_pc[:, c, :]  # (B, N)
        sorted_matrix = torch.sort(point_matrix, dim=-1)  # (B, N)
        indices_matrix = torch.stack([sorted_matrix[1]] * 3, dim=1)
        grid_pc = torch.gather(grid_pc, -1, indices_matrix).view(B, C, pow(grid_rate, c + 1), -1)
    return grid_pc


def pc_to_regular_grid(point_cloud, grid_rate):
    B, C, N = point_cloud.shape
    device = point_cloud.device
    grid_scale = torch.stack([torch.linspace(-1, 1, grid_rate + 1)] * 3, dim=0).to(device)
    seg_masks = torch.zeros([0], dtype=torch.bool).to(device)
    seg_mean = torch.zeros([0], dtype=torch.float).to(device)
    seg_std = torch.zeros([0], dtype=torch.float).to(device)
    for idx in list(itertools.product(list(range(grid_rate)), repeat=3)):
        idx = torch.tensor(idx).unsqueeze(1).to(device)
        seg_mask = (torch.gather(grid_scale, 1, idx) < point_cloud) * \
                   (torch.gather(grid_scale, 1, idx + 1) >= point_cloud)
        # seg_masks = torch.cat([seg_masks, seg_mask], dim=1)
        seg_mean = torch.cat([seg_mean, (point_cloud * seg_mask).mean(-1).unsqueeze(-1)], dim=2)
        seg_std = torch.cat([seg_std, (point_cloud * seg_mask).std(-1).unsqueeze(-1)], dim=2)
    seg_mean = torch.stack([seg_mean.mean(dim=1)] * seg_mean.size(2), dim=1) \
               - torch.stack([seg_mean.mean(dim=1)] * seg_mean.size(2), dim=2)
    seg_std = torch.stack([seg_std.mean(dim=1)] * seg_std.size(2), dim=1) \
              - torch.stack([seg_std.mean(dim=1)] * seg_std.size(2), dim=2)
    return torch.stack([seg_mean, seg_std], dim=1)


def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False,
                        marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10,
                        azim=240, axis=None, title=None, *args, **kwargs):
    plt.switch_backend('agg')
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis
    if title is not None:
        plt.title(title)
    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)
    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        # Multiply with 0.7 to squeeze free-space.
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()
    if not show_axis:
        plt.axis('off')
    if 'c' in kwargs:
        plt.colorbar(sc)
    if show:
        plt.show()
    return fig


def plot_3d_colormap(point_clouds, max_points, max_count, show_axis=True, in_u_sphere=False,
                     marker='.', s=10, alpha=.8, figsize=(10, 10), elev=10,
                     azim=240, axis=None, title=None, *args, **kwargs):
    plt.switch_backend('agg')
    x, y, z = point_clouds
    m_x, m_y, m_z = max_points
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        # ax2 = fig.add_subplot(122, projection='3d')
    else:
        ax = axis
        fig = axis
    if title is not None:
        plt.title(title)
    sc_pc = ax.scatter(x, y, z, marker=marker, c='lightgray', s=s, alpha=alpha)
    sc_max_pc = ax.scatter(m_x, m_y, m_z, marker=marker, c=max_count, cmap='rainbow', s=s, alpha=alpha)
    plt.colorbar(sc_max_pc, label='max_count')
    ax.view_init(elev=elev, azim=azim)
    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.5, 0.5)
    else:
        # Multiply with 0.7 to squeeze free-space.
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()
    if not show_axis:
        plt.axis('off')
    return fig


def colormap_save(dataloader, model, device, domain, save_dir, num_class, max_num_sample, target_domain=None):
    idx_to_label = {0: "bathtub", 1: "bed", 2: "bookshelf", 3: "cabinet",
                    4: "chair", 5: "lamp", 6: "monitor",
                    7: "plant", 8: "sofa", 9: "table"}
    sample_num = torch.zeros([num_class], dtype=torch.int).to(device)
    with torch.no_grad():
        model.eval()
        for data in dataloader:
            point_clouds = data['point_cloud'].to(device)
            labels = data['label'].to(device)
            pred, max_idx = model(point_clouds)
            if domain == 'source':
                save_path = os.path.join(save_dir, 'src')
                mask = (labels == pred.max(dim=1)[1])
                point_clouds = point_clouds[mask, :]
                labels = labels[mask]
                max_idx = max_idx[mask, :]
            elif domain == 'target':
                save_path = os.path.join(save_dir, 'trg_{}'.format(target_domain))
                pred_labels = pred.max(dim=1)[1]
            else:
                raise NotImplementedError
            point_clouds = point_clouds.cpu()
            for k in range(point_clouds.size(0)):
                class_idx = int(labels[k])
                if domain == 'target': class_idx = int(pred_labels[k])
                if sample_num[class_idx] == max_num_sample: continue
                sample_num[class_idx] += 1
                class_label = idx_to_label[class_idx]
                image_path = os.path.join(save_path, '{}'.format(class_label))
                os.makedirs(image_path, exist_ok=True)
                max_list, max_count = np.unique(max_idx[k].cpu(), return_counts=True)
                max_list = torch.tensor(max_list)
                max_count = (max_count - max_count.min()) / (max_count.max() - max_count.min())
                max_pc = torch.gather(point_clouds[k, :, :], 1,
                                      torch.stack([max_list] * 3, dim=0))
                # Colormap
                if domain == 'source':
                    img_title = '{}'.format(class_label)
                else:
                    true_label = idx_to_label[int(labels[k])]
                    img_title = 'true label : {}\npred label : {}'.format(true_label, class_label)
                fig = plot_3d_colormap(point_clouds[k, :, :], max_pc, max_count,
                                       in_u_sphere=True, show=False, title=img_title)
                fig.savefig(os.path.join(image_path, '{}.png'.format(sample_num[class_idx])))
                plt.close(fig)
            if sample_num.sum() == max_num_sample * num_class: break


def image_save(point_cloud, save_dir, save_folder, save_name, img_title, batch_idx=0, folder_numbering=True):
    for k in range(point_cloud.size(0)):
        fig = plot_3d_point_cloud(point_cloud[k][0], point_cloud[k][1], point_cloud[k][2],
                                  in_u_sphere=True, show=False,
                                  title='{}'.format(img_title))
        if folder_numbering:
            save_path = os.path.join(save_dir, '{}_{}'.format(save_folder, batch_idx * point_cloud.size(0) + k))
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, '{}.png'.format(save_name)))
        else:
            save_path = os.path.join(save_dir, '{}'.format(save_folder))
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(
                os.path.join(save_path, '{}_{}.png'.format(save_name, batch_idx * point_cloud.size(0) + k)))
        plt.close(fig)


def make_training_sample(point_cloud):
    B, C, N = point_cloud.shape
    device = point_cloud.device
    sample = torch.randn(B, C, int(N / 4)).to(device)
    sigma = [0.1, 0.15, 0.2]
    clip = [0.2, 0.3, 0.4]
    for i in range(3):
        jittering_sample = pcu.jitter(point_cloud, sigma=sigma[i], clip=clip[i])[:, :, torch.randperm(N)[:int(N / 4)]]
        sample = torch.cat([sample, jittering_sample], dim=2)
    sample_dist = point_cloud_distance_cp(point_cloud, sample, sampling=True).squeeze(dim=-1)
    return sample, sample_dist


def knn_point_sampling(point_cloud, target_points, sample_num):
    device = point_cloud.device
    B, C, N = point_cloud.shape
    _, _, M = target_points.shape
    point_cloud_matrix = torch.stack([point_cloud] * M, dim=2)
    target_points_matrix = torch.stack([target_points] * N, dim=3)
    distance_matrix = (point_cloud_matrix - target_points_matrix).pow(2).sum(dim=1).sqrt().to(device)
    knn_matrix = torch.topk(distance_matrix, sample_num, largest=False)
    knn_indices_matrix = torch.stack([knn_matrix[1]] * 3, dim=1)
    knn_points_matrix = torch.gather(point_cloud_matrix, 3, knn_indices_matrix)
    return knn_points_matrix


def point_cloud_distance_svd(point_cloud, target_points, k=5, p=0.01, sampling=False):
    if point_cloud.shape != target_points.shape:
        raise NotImplementedError
    device = point_cloud.device
    B, C, N = point_cloud.shape
    knn_points_matrix = knn_point_sampling(point_cloud, target_points, k)
    p_hat_matrix = torch.mean(knn_points_matrix, dim=3)
    p_matrix = (knn_points_matrix - p_hat_matrix.unsqueeze(dim=3))
    M_matrix = torch.matmul(p_matrix.permute(0, 2, 1, 3), p_matrix.permute(0, 2, 3, 1)) / k
    U_matrix, S_matrix, V_matrix = torch.svd(M_matrix)
    norm_matrix = U_matrix[:, :, :, 2]
    random_point_matrix = torch.gather(knn_points_matrix, 3,
                                       torch.randint(k, knn_points_matrix.shape)[:, :, :, 0:1].to(device)).squeeze()
    tangent_dist_matrix = torch.abs(torch.matmul(norm_matrix.unsqueeze(dim=2),
                                                 (target_points - random_point_matrix).permute(0, 2, 1).unsqueeze(3)))
    # regularize
    if sampling:
        return tangent_dist_matrix
    else:
        point_cloud_matrix = torch.stack([point_cloud] * N, dim=2)
        points_matrix = torch.stack([point_cloud] * N, dim=3)
        self_dist_matrix = (point_cloud_matrix - points_matrix).pow(2).sum(dim=1).sqrt()
        knn_matrix = torch.topk(self_dist_matrix, k, largest=False, sorted=True)
        reg = torch.clamp(torch.mean(knn_matrix[0]), min=0.1)
        loss = tangent_dist_matrix.mean() + (1 / reg) * p
        return loss


def point_cloud_distance_cp(point_cloud, target, k=3, sampling=False):
    if point_cloud.shape != target.shape:
        raise NotImplementedError
    knn_points_matrix = knn_point_sampling(point_cloud, target, k)
    # Cross product
    ref_point = knn_points_matrix[:, :, :, 0]
    cross_norm_matrix = torch.cross((ref_point - knn_points_matrix[:, :, :, 1]).transpose(2, 1),
                                    (ref_point - knn_points_matrix[:, :, :, 2]).transpose(2, 1))
    normalize_norm = torch.mul(cross_norm_matrix,
                               1 / torch.stack([cross_norm_matrix.pow(2).sum(axis=2).sqrt()] * 3, dim=2))
    cross_tangent_dist_matrix = torch.abs(torch.matmul(normalize_norm.unsqueeze(dim=2),
                                                       (target - ref_point).transpose(2, 1).unsqueeze(dim=3)))
    if sampling:
        return cross_tangent_dist_matrix
    else:
        loss2 = cross_tangent_dist_matrix.mean()
        return loss2


def point_cloud_segmentation_tangent_loss(point_clouds, pred, knn_num, device):
    tangent_loss_sum = 0.0
    num_seg_class = pred.size(1)
    part_mean_points = torch.zeros([0], dtype=torch.float).to(device)
    weighted_part_mean_points = torch.zeros([0], dtype=torch.float).to(device)
    for seg_class in range(num_seg_class):
        weight = torch.softmax(pred, dim=1)[:, seg_class, :]
        weight, weight_index = torch.topk(weight, k=knn_num)
        part_pc = torch.gather(point_clouds, 2, torch.stack([weight_index] * 3, dim=1))
        weight_part_pc = part_pc * torch.stack([weight] * 3, dim=1)
        p_matrix = (part_pc - part_pc.mean(dim=2).unsqueeze(-1)) * torch.stack([weight] * 3, dim=1)
        cov_matrix = torch.matmul(p_matrix[:, :, :17], p_matrix[:, :, :17].transpose(2, 1)) / knn_num
        try:
            U, S, V = torch.svd(cov_matrix.cpu())
        except:
            import ipdb;
            ipdb.set_trace()
        U_matrix = torch.stack([U[:, :, 2].to(device)] * knn_num, dim=1).unsqueeze(2)
        tangent_dist = torch.abs(torch.matmul(U_matrix, p_matrix.transpose(2, 1).unsqueeze(-1)))
        tangent_loss_sum += tangent_dist.mean()
        part_mean_points = torch.cat([part_mean_points, part_pc.mean(dim=2).unsqueeze(1)], dim=1)
        weighted_part_mean_points = torch.cat([weighted_part_mean_points, weight_part_pc.mean(dim=2).unsqueeze(1)],
                                              dim=1)
    tangent_loss = tangent_loss_sum / num_seg_class
    return tangent_loss, part_mean_points, weighted_part_mean_points


def point_cloud_segmentation_std_loss(point_clouds, part_mean_points, pred):
    num_seg_class = pred.size(1)
    sum_part_std = 0.0
    weight_matrix = torch.softmax(pred, dim=1)
    for seg_class in range(num_seg_class):
        distance_matrix = (point_clouds - part_mean_points[:, seg_class, :].unsqueeze(-1)).pow(2).sum(dim=1).sqrt()
        weighted_distance = (weight_matrix[:, seg_class, :] * distance_matrix).mean(dim=1)
        seg_class_std = weighted_distance / weight_matrix[:, seg_class, :].mean(dim=1)
        sum_part_std += seg_class_std.mean()
    part_std = sum_part_std / num_seg_class
    return part_std


def point_cloud_segmentation_contrastive_loss(point_clouds, pred, theta_regressor, emd_loss, device):
    num_seg_class = pred.size(1)
    random_idx = torch.randperm(point_clouds.size(0)).to(device)
    target_point_cloud = torch.stack([point_clouds[0, :, :]] * point_clouds.size(0), dim=0)
    theta = theta_regressor(torch.cat([point_clouds, target_point_cloud], dim=1))
    aligned_point_clouds = pcu.rotate_shape(point_clouds, 'z', theta)
    shuffled_point_clouds = aligned_point_clouds[random_idx, :, :].to(device)
    emd_loss, emd_matching_idx = emd_loss(aligned_point_clouds.permute(0, 2, 1),
                                          shuffled_point_clouds.permute(0, 2, 1), 0.05, 3000)
    emd_matching_idx = emd_matching_idx.type(torch.LongTensor).to(device)
    pos_loss = 0.0
    neg_loss = 0.0
    for seg_class in range(num_seg_class):
        softmax_weight = torch.softmax(pred, dim=1)[:, seg_class, :]
        for sc in range(num_seg_class):
            shuffled_weight = torch.softmax(pred, dim=1)[:, sc, :][random_idx, :]
            shuffled_weight = torch.gather(shuffled_weight, 1, emd_matching_idx)
            max_weight = torch.cat([softmax_weight.unsqueeze(0), shuffled_weight.unsqueeze(0)],
                                   dim=0).max(dim=0)[0]
            max_weight, seg_class_idx = torch.topk(max_weight, 50, dim=1)
            if sc == seg_class:
                pos_loss += (max_weight * torch.gather(emd_loss, 1, seg_class_idx)).mean()
            else:
                neg_loss += (max_weight * torch.gather(emd_loss, 1, seg_class_idx)).mean()
    pos_loss = pos_loss / num_seg_class
    neg_loss = neg_loss / (num_seg_class * num_seg_class - num_seg_class)
    return pos_loss, neg_loss


def segmentation_cosine_similarity_contrastive_loss(point_clouds, pred, sim_feature_extractor, device, tau=1.0):
    cosine_similarity_loss = torch.nn.CosineSimilarity(dim=-1)
    seg_mask = torch.zeros([0], dtype=torch.bool).to(device)
    sim_feature = torch.zeros([0], dtype=torch.float).to(device)
    segmentation_label = torch.max(pred, dim=1)[1]  # (B, N)
    num_seg_class = pred.size(1)
    softmax_layer = torch.nn.Softmax(dim=1)
    pred = softmax_layer(pred)
    for seg_class in range(num_seg_class):
        seg_class_mask = (segmentation_label == seg_class).unsqueeze(1)
        seg_class_sim_feature = sim_feature_extractor(point_clouds, seg_class_mask).unsqueeze(1)
        seg_mask = torch.cat([seg_mask, seg_class_mask], dim=1)
        sim_feature = torch.cat([sim_feature, seg_class_sim_feature], dim=1)
    while 1:
        rand_seg_idx = torch.randperm(num_seg_class)
        if torch.sum(rand_seg_idx == torch.tensor(list(range(num_seg_class)))) == 0: break
    rand_batch_idx = torch.randperm(pred.size(0))
    rand_sim_feature = sim_feature[rand_batch_idx]
    pos_loss = cosine_similarity_loss(sim_feature, rand_sim_feature) / tau
    neg_loss = cosine_similarity_loss(sim_feature, rand_sim_feature[:, rand_seg_idx, :]) / tau
    return pos_loss.mean(), neg_loss.mean()


def cosine_sim_loss(pred, labels, criterion, tau):
    B = pred.size(0)
    device = pred.device
    pos_pred = torch.zeros([0], dtype=torch.float).to(device)
    neg_pred = torch.zeros([0], dtype=torch.float).to(device)
    for b in range(B):
        rand_idx = torch.randperm(B).to(device)
        pos_idx = (labels[rand_idx] == labels[b]).nonzero()[0]
        neg_idx = (labels[rand_idx] != labels[b]).nonzero()[:int(B / 4) - 1].squeeze(-1)
        pos_pred = torch.cat([pos_pred, pred[rand_idx, :][pos_idx, :]], dim=0)
        try:
            neg_pred = torch.cat([neg_pred, pred[rand_idx, :][neg_idx, :].unsqueeze(0)], dim=0)
        except:
            continue
    sample_pred = torch.cat([pos_pred.unsqueeze(dim=1), neg_pred], dim=1)
    similarity_matrix = torch.nn.CosineSimilarity(dim=-1)(torch.stack([pred] * sample_pred.size(1), dim=1),
                                                          sample_pred) / tau
    sim_labels = torch.zeros(similarity_matrix.size(0), dtype=torch.long).to(device)
    loss = criterion(similarity_matrix, sim_labels)
    positives = similarity_matrix[:, 0].mean()
    negatives = similarity_matrix[:, 1:].mean()
    return positives, negatives, loss


def grid_colormap(point_grid, color, save_dir):
    matplotlib.use('TkAgg')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    point_grid = point_grid.cpu()
    x = point_grid[:, 0, :]
    y = point_grid[:, 1, :]
    z = point_grid[:, 2, :]
    c = color.cpu()
    img = ax.scatter(x, y, z, s=1.5, c=c, cmap=plt.hot())
    fig.colorbar(img)
    plt.savefig(save_dir)


def optimize_visualize(point_cloud, encoder, decoder, learning_rate, num_epoch, knn_num, save_dir, batch_idx=0):
    B, C, N = point_cloud.shape
    device = point_cloud.device
    z = Variable(torch.randn(B, C, N).cuda(), requires_grad=True).to(device)
    save_dir = os.path.join(save_dir, 'optimize_visualize')
    for epoch in range(num_epoch):
        knn_sampling = knn_point_sampling(point_cloud, z, knn_num)
        source_latent_vector = encoder(knn_sampling)
        loss = torch.abs(decoder(z, source_latent_vector)).mean()
        loss.backward()
        if loss < 0.05:
            learning_rate = 1
        elif loss < 0.01:
            learning_rate = 0.1
        elif loss < 0.001:
            learning_rate = 0.01
        with torch.no_grad():
            my_vector_size = torch.stack([z.pow(2).sum(axis=1).sqrt()] * 3, dim=1)
            my_norm = z / my_vector_size
            my_grad = (z.grad * my_norm).sum(axis=1)
            my_grad = my_norm * torch.stack([my_grad] * 3, dim=1)
            z -= my_grad * learning_rate
        z.grad.zero_()
        if epoch % 100 == 0:
            image_save(z.detach().cpu(), save_dir, 'test', 'epoch_{}'.format(epoch), 'epoch : {}'.format(epoch),
                       batch_idx=batch_idx)


def grid_visualize(point_clouds, encoder, decoder, grid_scale, threshold, knn_num, save_dir, batch_idx=0):
    B, C, N = point_clouds.shape
    device = point_clouds.device
    with torch.no_grad():
        scale = torch.linspace(-1.0, 1.0, grid_scale)
        point_grid = torch.stack([torch.cartesian_prod(scale, scale, scale).transpose(1, 0)] * B, dim=0).to(device)
        partial_size = 100
        test_pred = torch.Tensor([]).to(device)
        for i in range(int((grid_scale ** 3) / partial_size)):
            partial_point_grid = point_grid[:, :, i * partial_size:(i + 1) * partial_size]
            temp_latent_vector = encoder(knn_point_sampling(point_clouds, partial_point_grid, knn_num))
            test_pred = torch.cat([test_pred, decoder(partial_point_grid, temp_latent_vector).squeeze(dim=-1)
                                   ], dim=2)
        for b in range(B):
            test_pred_sample = test_pred[b, :, :]
            masked_index = (test_pred_sample.squeeze() < threshold).nonzero()
            pred_pc = torch.gather(point_grid[b, :, :], 1, torch.stack([masked_index.squeeze()] * 3, dim=0)) \
                .unsqueeze(dim=0)
            if pred_pc.size(2) > N:
                pred_pc, _ = pcu.random_point_sample(pred_pc, N)
            elif pred_pc.size(2) < N:
                new_pred_pc = pred_pc
                while new_pred_pc.size(2) < N:
                    new_pred_pc = torch.cat([new_pred_pc, pcu.jitter(pred_pc)], dim=2)
                pred_pc, _ = pcu.random_point_sample(new_pred_pc, N)
            # pcu.visualize(point_clouds)
            # pcu.visualize(pred_pc)
            image_save(pred_pc.detach().cpu(), save_dir, 'grid_visualize', 'prediction', 'predict_pc',
                       batch_idx=batch_idx * B + b, folder_numbering=False)


def visualize_animation(point_cloud):
    if point_cloud.size(0) != 1:
        raise NotImplementedError
    pcd = open3d.geometry.PointCloud()
    permute = [0, 2, 1]
    point_cloud = point_cloud[:, permute, :]
    pcd.points = open3d.utility.Vector3dVector(np.array(point_cloud.squeeze(axis=0).permute(1, 0).cpu()))

    # def capture_image(vis):
    #     image = vis.capture_screen_float_buffer()
    #     plt.imsave(os.path.join(save_dir, '{}_{}.png'.format(save_name, len(os.listdir(save_dir)))),
    #                np.asarray(image))
    #     return False
    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        # capture_image(vis)
        return False

    open3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)


def save_gif(point_cloud, save_name, save_path, save_num=1):
    if point_cloud.size(0) > save_num:
        raise NotImplementedError
    for k in range(point_cloud.size(0)):
        img_list = []
        img_path_list = []
        point_cloud_sample = point_cloud[k, :, :].unsqueeze(0)
        for i in range(20):
            point_cloud_sample = point_cloud_sample.cpu()
            fig = plot_3d_point_cloud(point_cloud_sample[0][0], point_cloud_sample[0][1], point_cloud_sample[0][2],
                                      in_u_sphere=True, show=False, show_axis=False)
            point_cloud_sample = pcu.rotate_shape(point_cloud_sample, 'z', rotation_angle=18 * np.pi / 180)
            img_path = os.path.join(save_path, '{}.png'.format(i))
            fig.savefig(img_path)
            img_path_list.append(img_path)
            img_list.append(imageio.imread(img_path))
        plt.close(fig)
        imageio.mimsave(os.path.join(save_path, '{}_{}.gif'.format(save_name, str(k))), img_list, fps=7)
        for img_file in img_path_list:
            if os.path.exists(img_file):
                os.remove(img_file)


def save_confusion_matrix(pred_list, labels_list, num_class, save_path, save_name, cmap=None, title=None,
                          normalize=True):
    plt.switch_backend('agg')
    cm = confusion_matrix(labels_list.cpu(), pred_list.cpu())
    accuracy = np.trace(cm) / float(np.sum(cm))
    mis_class = 1 - accuracy
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if title is None:
        title = 'Confusion matrix'
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(num_class))
    plt.yticks(np.arange(num_class))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, mis_class))
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, '{}.png'.format(save_name)))
    plt.close()


def save_cos_sim_confusion_matrix(sim_confusion_matrix, num_class, save_path, save_name, cmap=None, title=None,
                                  normalize=False):
    plt.switch_backend('agg')
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if title is None:
        title = 'Confusion matrix'
    plt.figure(figsize=(8, 6))
    plt.imshow(sim_confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(num_class))
    plt.yticks(np.arange(num_class))
    if normalize:
        sim_confusion_matrix = sim_confusion_matrix.type(torch.float) / sim_confusion_matrix.sum(axis=1)[:, np.newaxis]
    thresh = sim_confusion_matrix.max() / 1.5 if normalize else sim_confusion_matrix.max() / 2
    for i, j in itertools.product(range(sim_confusion_matrix.shape[0]), range(sim_confusion_matrix.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(sim_confusion_matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if sim_confusion_matrix[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:0.4f}".format(sim_confusion_matrix[i, j]),
                     horizontalalignment="center",
                     color="white" if sim_confusion_matrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nsimilarity value')
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, '{}.png'.format(save_name)))
    plt.close()
