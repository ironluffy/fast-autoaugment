import os
import torch

if __name__ == "__main__":
    root = 'aug_final'
    src_domain = 'modelnet'
    trg_domain = 'shapenet'
    dc_model = 'pointnet'
    num_op = 5
    num_cv = 1
    num_policy = 5
    num_search = 200
    policy_path = os.path.join(root,
                               '{}_{}2{}_op{}_ncv{}_npy{}_ns{}.pth'.format(dc_model, src_domain, trg_domain, num_op,
                                                                           num_cv, num_policy, num_search))
    policies = torch.load(policy_path)
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