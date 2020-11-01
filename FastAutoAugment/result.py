import os
import argparse

st_dict = {
    'Supervised': ['shapenet_shapenet', 'scannet_scannet', 'modelnet_modelnet',
                   'scannet_scannet', 'modelnet_modelnet', 'shapenet_shapenet'],
    'Naive_occlusion': ['shapenet_scannet'],
    'Other': ['modelnet_shapenet', 'modelnet_scannet', 'shapenet_modelnet', 'shapenet_scannet',
              'scannet_modelnet', 'scannet_shapenet'],
    'Viewpoint_occlusion': ['modelnet_shapenet', 'modelnet_scannet', 'shapenet_modelnet', 'shapenet_scannet',
                            'scannet_modelnet', 'scannet_shapenet'],
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='PointNet')
parser.add_argument('--method', type=str, default='pseudo_classifier_occlusion')
parser.add_argument('--keyword1', type=str, default=None)
parser.add_argument('--keyword2', type=str, default=None)

args = parser.parse_args()

model = args.model
method = args.method
keyword1 = args.keyword1
keyword2 = args.keyword2

tags_list = ['', '1', '2', '3']
root = './aug_trained'

for i in range(4):
    for st in (st_dict[method] if method in st_dict.keys() else st_dict['Other']):
        if args.keyword1 is None:
            tag = '{}_{}'.format(st, i)
        else:
            if args.keyword2 is None:
                tag = '{}_{}'.format(args.keyword1, i)
            else:
                tag = '{}_{}_{}'.format(args.keyword2, args.keyword1, i)


        exp_root = os.path.join(root, model, method, st, tag)
        if not os.path.exists(exp_root):
            print('not start, ', end='')
            continue

        exp_root = os.path.join(exp_root, 'test.log')
        if not os.path.exists(exp_root):
            print('running, ', end='')
            continue

        with open(exp_root, 'r') as f:
            logs = [line.replace('\n', '') for line in f.readlines()]
        for idx in range(len(logs)):
            if 'Best Sample ACC' in logs[idx]:
                for idx2 in range(idx, len(logs)):
                    if 'test sample accuracy' in logs[idx2] or 'test_sample_accuracy' in logs[idx2]:
                        print('%.2lf, ' % (float(logs[idx2].split('/')[1].split(': ')[1]) * 100), end='')
                        break
                break
    print()