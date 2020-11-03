#!/bin/bash
num_op=3
num_cv=1
num_policy=20
num_search=200
tag=1
#CUDA_VISIBLE_DEVICES=4 python3 aug_traineval.py --source_domain modelnet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} --num-search=${num_search} --model=dgcnn
#CUDA_VISIBLE_DEVICES=4 python3 aug_traineval.py --source_domain scannet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} --num-search=${num_search} --model=dgcnn
#CUDA_VISIBLE_DEVICES=4 python3 aug_traineval.py --source_domain shapenet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} --num-search=${num_search} --model=dgcnn

CUDA_VISIBLE_DEVICES=5 python3 aug_traineval.py --source_domain modelnet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} --num-search=${num_search} --model=dgcnn
CUDA_VISIBLE_DEVICES=5 python3 aug_traineval.py --source_domain scannet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} --num-search=${num_search} --model=dgcnn
CUDA_VISIBLE_DEVICES=5 python3 aug_traineval.py --source_domain shapenet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} --num-search=${num_search} --model=dgcnn
