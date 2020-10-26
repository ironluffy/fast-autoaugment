#!/bin/bash
num_cv=1
num_op=5
num_policy=5
python3 search.py -c ../confs/pointNet_model2scan.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
python3 search.py -c ../confs/pointNet_model2shape.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
python3 search.py -c ../confs/pointNet_scan2model.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
python3 search.py -c ../confs/pointNet_scan2shape.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
#python3 search.py -c ../confs/pointNet_shape2scan.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
#python3 search.py -c ../confs/pointNet_shape2model.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
#CUDA_VISIBLE_DEVICES=0,1 python3 search.py -c ../confs/pointNet_model2scan.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
#CUDA_VISIBLE_DEVICES=0,1 python3 search.py -c ../confs/pointNet_model2shape.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
#CUDA_VISIBLE_DEVICES=0,1 python3 search.py -c ../confs/pointNet_scan2model.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
#CUDA_VISIBLE_DEVICES=2,3 python3 search.py -c ../confs/pointNet_scan2shape.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
#CUDA_VISIBLE_DEVICES=2,3 python3 search.py -c ../confs/pointNet_shape2scan.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &
#CUDA_VISIBLE_DEVICES=2,3 python3 search.py -c ../confs/pointNet_shape2model.yaml --dataroot .. --num-op=${num_op} --num_cv=${num_cv} --num-policy=${num_policy} &

#CUDA_VISIBLE_DEVICES=0,1 python3 search.py -c ../confs/pointNet_model2scan.yaml --dataroot .. &
#CUDA_VISIBLE_DEVICES=0,1 python3 search.py -c ../confs/pointNet_model2shape.yaml --dataroot .. &
#CUDA_VISIBLE_DEVICES=0,1 python3 search.py -c ../confs/pointNet_scan2model.yaml --dataroot .. &
#CUDA_VISIBLE_DEVICES=2,3 python3 search.py -c ../confs/pointNet_scan2shape.yaml --dataroot .. &
#CUDA_VISIBLE_DEVICES=2,3 python3 search.py -c ../confs/pointNet_shape2scan.yaml --dataroot .. &
#CUDA_VISIBLE_DEVICES=2,3 python3 search.py -c ../confs/pointNet_shape2model.yaml --dataroot .. &