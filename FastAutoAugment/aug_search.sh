#!/bin/bash
num_op=2
num_cv=1
num_policy=30
CUDA_VISIBLE_DEVICES=4,5 python3 search.py -c ../confs/pointNet_model2scan.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
CUDA_VISIBLE_DEVICES=4,5 python3 search.py -c ../confs/pointNet_model2shape.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
CUDA_VISIBLE_DEVICES=4,5 python3 search.py -c ../confs/pointNet_scan2model.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
CUDA_VISIBLE_DEVICES=6,7 python3 search.py -c ../confs/pointNet_scan2shape.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
CUDA_VISIBLE_DEVICES=6,7 python3 search.py -c ../confs/pointNet_shape2scan.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
CUDA_VISIBLE_DEVICES=6,7 python3 search.py -c ../confs/pointNet_shape2model.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
