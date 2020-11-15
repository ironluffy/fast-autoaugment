import os
import copy
import torch


if __name__ == "__main__":
    root1 = os.path.join('/home/ironluffy/Downloads')
    root2 = os.path.join('/home/ironluffy/augment_list')

    dm_list = ['modelnet', 'scannet', 'shapenet']

    for src_dm in dm_list:
        trg_list = copy.copy(dm_list)
        trg_list.remove(src_dm)
        for trg_dm in trg_list:
            org_load = torch.load(os.path.join(root1, '{}2{}_op2_ncv1_npy30.pth'.format(src_dm, trg_dm)))
            second_load = torch.load(os.path.join(root1, '{}2{}_op2_ncv1_npy30.pth'.format(src_dm, trg_dm)))
            print(org_load==second_load)
