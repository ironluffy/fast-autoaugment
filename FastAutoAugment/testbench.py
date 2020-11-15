import os
import torch
from tqdm import tqdm
from datasets import PointDA
from torch.utils.data import DataLoader
from utils.emd.emd_module import emdModule
from aug_eval.aug_train import Augmentation

if __name__ == "__main__":
    root = 'aug_final_rc_plane_pointnetv7'
    src_domain = 'shapenet'
    trg_domain = 'scannet'
    dc_model = 'pointnetv7'
    num_op = 2
    num_cv = 1
    rnd_range = 0.1
    num_policy = 5
    num_search = 100
    use_emd_fale = True
    policy_path = os.path.join(root,
                               '{0}_{1}2{2}_op{3}_ncv{4}_npy{5}_ns{6}_rnd{7:0.2f}_{8}.pth'.format(dc_model, src_domain,
                                                                                                  trg_domain, num_op,
                                                                                                  num_cv, num_policy,
                                                                                                  num_search,
                                                                                                  rnd_range,
                                                                                                  use_emd_fale))
    load_dict = torch.load(policy_path)
    policies = load_dict['final_policy']
    print(load_dict)
    aug_model = Augmentation(policies)

    sub_policy_prob_dict = {}
    sub_policy_level_dict = {}
    sub_policy_count_dict = {}
    for policy in policies:
        for sub_policy in policy:
            name, prob, level = sub_policy
            if name in sub_policy_count_dict.keys():
                sub_policy_prob_dict[name] += prob
                sub_policy_level_dict[name] += level
                sub_policy_count_dict[name] += 1
            else:
                sub_policy_prob_dict[name] = prob
                sub_policy_level_dict[name] = level
                sub_policy_count_dict[name] = 1

    for key in sub_policy_count_dict.keys():
        sub_policy_prob_dict[key] /= sub_policy_count_dict[key]
        sub_policy_level_dict[key] /= sub_policy_count_dict[key]

    source_trainset = PointDA(root='..', domain=src_domain, partition='train',
                              num_points=1024, sampling_method='random', download=True)
    source_valset = PointDA(root='..', domain=src_domain, partition='val',
                            num_points=1024, sampling_method='random', download=True)

    # DataLoader
    source_trainloader = DataLoader(source_trainset, num_workers=0,
                                    batch_size=64, shuffle=True)
    source_valloader = DataLoader(source_valset, num_workers=0,
                                  batch_size=64)

    # emd_loss = emdModule().cuda()
    # total_emd_loss = 0
    # for data in tqdm(source_trainloader):
    #     with torch.no_grad():
    #         with torch.no_grad():
    #             point_clouds = data['point_cloud'].cuda()
    #             trans_pc = aug_model(point_clouds)
    #
    #             loss_emd = (torch.mean(
    #                 emd_loss(point_clouds.permute(0, 2, 1), trans_pc.permute(0, 2, 1), 0.05, 3000)[0])) * 10000
    #             total_emd_loss += loss_emd.item()
    #
    # print(total_emd_loss / 10000)

    print(policies)

    print('prob')
    for key, value in sub_policy_prob_dict.items():
        print(key, value)

    print('level')

    for key, value in sub_policy_level_dict.items():
        print(key, value)

    print('count')
    for key, value in sub_policy_count_dict.items():
        print(key, value)

    scores = torch.load(policy_path)
