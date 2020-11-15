#!/bin/bash
num_op=2
num_cv=1
num_policy=30
num_search=100
random_range=0.1
emd_coeff=0
dc_model=pointnetv7
model=pointnet
tag=1
CUDA_VISIBLE_DEVICES=0,1 python3 aug_traineval.py --source_domain shapenet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model} --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=0,1 python3 aug_traineval.py --source_domain shapenet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=0,1 python3 aug_traineval.py --source_domain modelnet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=0,1 python3 aug_traineval.py --source_domain modelnet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=0,1 python3 aug_traineval.py --source_domain scannet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=0,1 python3 aug_traineval.py --source_domain scannet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &

tag=2
CUDA_VISIBLE_DEVICES=2,3 python3 aug_traineval.py --source_domain shapenet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model} --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=2,3 python3 aug_traineval.py --source_domain shapenet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=2,3 python3 aug_traineval.py --source_domain modelnet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=2,3 python3 aug_traineval.py --source_domain modelnet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=2,3 python3 aug_traineval.py --source_domain scannet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=2,3 python3 aug_traineval.py --source_domain scannet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &

tag=3
CUDA_VISIBLE_DEVICES=6,7 python3 aug_traineval.py --source_domain shapenet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model} --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=6,7 python3 aug_traineval.py --source_domain shapenet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=6,7 python3 aug_traineval.py --source_domain modelnet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=6,7 python3 aug_traineval.py --source_domain modelnet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=6,7 python3 aug_traineval.py --source_domain scannet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &
CUDA_VISIBLE_DEVICES=6,7 python3 aug_traineval.py --source_domain scannet --target_domain shapenet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model}  --random_range=${random_range} --aug_all &

