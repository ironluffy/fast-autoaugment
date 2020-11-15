#!/bin/bash
num_op=3
num_cv=1
num_policy=20
num_search=100
random_range=0.1
dc_model=pointnetv7
model=dgcnn
emd_coeff=10
tag=1
CUDA_VISIBLE_DEVICES=0,1 python3 aug_traineval.py --source_domain shapenet --target_domain scannet --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} --num-search=${num_search} --tag=${tag} --dc_model=${dc_model} --model=${model} --random_range=${random_range} --emd_coeff=${emd_coeff} --ablated=dc&

