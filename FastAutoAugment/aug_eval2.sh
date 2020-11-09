#!/bin/bash
num_op=3
num_cv=1
num_policy=20
num_search=100
random_range=0.1
dc_model=pointnetv7
model=dgcnn
tag=1
sleep 3h
CUDA_VISIBLE_DEVICES=8,9 python3 aug_traineval.py --source_domain modelnet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --random_range=${random_range} --model=${model}
CUDA_VISIBLE_DEVICES=8,9 python3 aug_traineval.py --source_domain scannet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --random_range=${random_range} --model=${model}
CUDA_VISIBLE_DEVICES=8,9 python3 aug_traineval.py --source_domain shapenet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --random_range=${random_range} --model=${model}
tag=2
CUDA_VISIBLE_DEVICES=8,9 python3 aug_traineval.py --source_domain modelnet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --random_range=${random_range} --model=${model}
CUDA_VISIBLE_DEVICES=8,9 python3 aug_traineval.py --source_domain scannet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --random_range=${random_range} --model=${model}
CUDA_VISIBLE_DEVICES=8,9 python3 aug_traineval.py --source_domain shapenet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --random_range=${random_range} --model=${model}
tag=3
CUDA_VISIBLE_DEVICES=8,9 python3 aug_traineval.py --source_domain modelnet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --random_range=${random_range} --model=${model}
CUDA_VISIBLE_DEVICES=8,9 python3 aug_traineval.py --source_domain scannet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --random_range=${random_range} --model=${model}
CUDA_VISIBLE_DEVICES=8,9 python3 aug_traineval.py --source_domain shapenet --target_domain modelnet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --random_range=${random_range} --model=${model}


