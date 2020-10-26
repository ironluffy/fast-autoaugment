#!/bin/bash
num_op=2
num_cv=1
num_policy=5
tag=1
CUDA_VISIBLE_DEVICES=0,1 python3 aug_traineval.py --source_domain modelnet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} --model=dgcnn&
#CUDA_VISIBLE_DEVICES=0,1 python3 aug_traineval.py --source_domain modelnet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} &
#CUDA_VISIBLE_DEVICES=0,1 python3 aug_traineval.py --source_domain scannet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} &
#CUDA_VISIBLE_DEVICES=2,3 python3 aug_traineval.py --source_domain scannet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} &
#CUDA_VISIBLE_DEVICES=2,3 python3 aug_traineval.py --source_domain shapenet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} &
#CUDA_VISIBLE_DEVICES=2,3 python3 aug_traineval.py --source_domain shapenet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} &

#CUDA_VISIBLE_DEVICES=4,5 python3 aug_traineval.py --source_domain modelnet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} &
#CUDA_VISIBLE_DEVICES=4,5 python3 aug_traineval.py --source_domain modelnet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} &
#CUDA_VISIBLE_DEVICES=4,5 python3 aug_traineval.py --source_domain scannet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} &
#CUDA_VISIBLE_DEVICES=6,7 python3 aug_traineval.py --source_domain scannet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} &
#CUDA_VISIBLE_DEVICES=6,7 python3 aug_traineval.py --source_domain shapenet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} &
#CUDA_VISIBLE_DEVICES=6,7 python3 aug_traineval.py --source_domain shapenet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --tag=${tag} &