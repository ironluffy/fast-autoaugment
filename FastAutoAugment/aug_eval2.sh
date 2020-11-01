#!/bin/bash
num_op=2
num_cv=1
num_policy=20
tag=1
CUDA_VISIBLE_DEVICES=6,7 python3 aug_traineval.py --source_domain shapenet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} --model=dgcnn
CUDA_VISIBLE_DEVICES=6,7 python3 aug_traineval.py --source_domain shapenet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} --model=dgcnn
