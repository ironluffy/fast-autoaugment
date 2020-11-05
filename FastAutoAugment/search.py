import copy
import os
import sys
import time
import random
import torch
import torch.nn as nn
import numpy as np
import ray
import gorilla

from tqdm import tqdm
from hyperopt import hp
from ray.tune.trial import Trial
from ray.tune.suggest import HyperOptSearch
from ray.tune.trial_runner import TrialRunner
from metrics import Accumulator
from collections import OrderedDict, defaultdict
from train import train_and_eval
from data import get_dataloaders
from theconf import Config as C, ConfigArgumentParser
from FastAutoAugment.utils.point_augmentations import augment_list
from ray.tune import register_trainable, run_experiments
from networks import get_model, num_class
from common import get_logger, add_filehandler
from archive import remove_deplicates, policy_decoder
from utils.emd.emd_module import emdModule
from utils.point_augmentations import apply_augment

top1_valid_by_cv = defaultdict(lambda: list)


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, pnt):
        org_size = pnt.size(0)
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                pnt = apply_augment(pnt, name, level)
        return pnt


def step_w_log(self):
    original = gorilla.get_original_attribute(ray.tune.trial_runner.TrialRunner, 'step')

    # log
    cnts = OrderedDict()
    for status in [Trial.RUNNING, Trial.TERMINATED, Trial.PENDING, Trial.PAUSED, Trial.ERROR]:
        cnt = len(list(filter(lambda x: x.status == status, self._trials)))
        cnts[status] = cnt
    best_top1_acc = 0.
    for trial in filter(lambda x: x.status == Trial.TERMINATED, self._trials):
        if not trial.last_result:
            continue
        best_top1_acc = max(best_top1_acc, trial.last_result['top1_valid'])
    print('iter', self._iteration, 'top1_acc=%.3f' % best_top1_acc, cnts, end='\r')
    return original(self)


patch = gorilla.Patch(ray.tune.trial_runner.TrialRunner, 'step', step_w_log, settings=gorilla.Settings(allow_hit=True))
gorilla.apply(patch)

logger = get_logger('Fast AutoAugment')


def _get_path(dataset, model, tag):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        'models/%s_%s_%s.model' % (dataset, model, tag))


@ray.remote(num_gpus=2, max_calls=2)
def train_model(config, dataroot, augment, cv_ratio_test, cv_num, cv_fold, save_path=None, skip_exist=False,
                is_dc=False):
    C.get()
    C.get().conf = config
    C.get()['aug'] = augment

    result = train_and_eval(None, os.path.abspath(dataroot), cv_ratio_test, cv_num, cv_fold, save_path=save_path,
                            only_eval=skip_exist)
    return C.get()['model']['type'], cv_fold, result


def eval_tta(config, augment, reporter):
    C.get()
    C.get().conf = config
    cv_ratio_test, cv_fold, save_path = augment['cv_ratio_test'], augment['cv_fold'], augment['save_path']

    # setup - provided augmentation rules
    C.get()['aug'] = policy_decoder(augment, augment['num_policy'], augment['num_op'])

    # eval
    model = get_model(C.get()['model'], num_class(C.get()['dataset']))
    model = nn.DataParallel(model).cuda()
    ckpt = torch.load(save_path + '.pth')
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    src_loaders = []
    trg_loaders = []
    for _ in range(augment['num_policy']):
        # _, tl, validloader, tl2 = get_dataloaders(C.get()['dataset'], C.get()['batch'], augment['dataroot'],
        #                                           cv_ratio_test, split_idx=cv_fold)
        _, src_tl, src_validloader, src_ttl = get_dataloaders(C.get()['dataset'],
                                                              C.get()['batch'],
                                                              augment['dataroot'],
                                                              cv_ratio_test,
                                                              cv_num,
                                                              split_idx=cv_fold,
                                                              target=False)

        src_loaders.append(iter(src_validloader))
        del src_tl, src_ttl

    start_t = time.time()
    metrics = Accumulator()
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    emd_loss = nn.DataParallel(emdModule()).cuda()
    aug_oper = Augmentation(C.get()['aug'])

    losses = []
    corrects = []
    for loader in src_loaders:
        data = next(loader)
        point_cloud = data['point_cloud'].cuda()
        label = torch.ones_like(data['label'], dtype=torch.int64).cuda()
        trans_pc = aug_oper(point_cloud)

        pred = model(trans_pc)

        loss_emd = (torch.mean(emd_loss(point_cloud, trans_pc)[0])) / 0.001
        loss = loss_fn(pred, label) + loss_emd
        losses.append(loss.detach().cpu().numpy())

        pred = pred.max(dim=1)[1]
        pred = pred.t()
        correct = float(torch.sum(pred == label).item()) / pred.size(0) * 100
        corrects.append(correct)
        del loss, correct, pred, data, label

    losses = np.concatenate(losses)
    losses_min = np.min(losses, axis=0).squeeze()
    corrects_max = max(corrects)
    metrics.add_dict({
        'minus_loss': -1 * np.sum(losses_min),
        'correct': np.sum(corrects_max),
        # 'cnt': len(corrects_max)
    })
    del corrects, corrects_max

    del model
    # metrics = metrics / 'cnt'
    gpu_secs = (time.time() - start_t) * torch.cuda.device_count()
    # print(metrics)
    reporter(minus_loss=metrics['minus_loss'], top1_valid=metrics['correct'], elapsed_time=gpu_secs, done=True)
    return metrics['minus_loss']


if __name__ == '__main__':
    import json
    from pystopwatch2 import PyStopwatch

    w = PyStopwatch()

    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--dataroot', type=str, default='../data', help='torchvision data folder')
    parser.add_argument('--until', type=int, default=5)
    parser.add_argument('--num-op', type=int, default=2)
    parser.add_argument('--num_cv', type=int, default=5)
    parser.add_argument('--num-policy', type=int, default=5)
    parser.add_argument('--num-search', type=int, default=100)
    parser.add_argument('--cv-ratio', type=float, default=0.4)
    parser.add_argument('--dc_model', type=str, default='pointnetv7',
                        choices=['pointnet', 'dgcnn', 'pointnetv5', 'pointnetv7'])
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--decay', type=float, default=-1)
    parser.add_argument('--per-class', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()

    if args.decay > 0:
        logger.info('decay=%.4f' % args.decay)
        C.get()['optimizer']['decay'] = args.decay

    C.get()['model']['type'] = args.dc_model
    add_filehandler(logger, os.path.join('models', '%s_%s_cv%.1f.log' % (
        C.get()['dataset'], C.get()['model']['type'], args.cv_ratio)))
    logger.info('configuration...')
    logger.info(json.dumps(C.get().conf, sort_keys=True, indent=4))
    logger.info('initialize ray...')
    # ray.init(redis_address=args.redis)
    ray.init(num_gpus=2)

    num_result_per_cv = args.topk
    cv_num = args.num_cv
    copied_c = copy.deepcopy(C.get().conf)

    logger.info('search augmentation policies, dataset=%s model=%s' % (C.get()['dataset'], C.get()['model']['type']))
    logger.info('----- Train without Augmentations cv=%d ratio(test)=%.1f -----' % (cv_num, args.cv_ratio))
    w.start(tag='train_no_aug')
    paths = [_get_path(C.get()['dataset'], C.get()['model']['type'],
                       'ratio{:.1f}_fold{}_{}2{}_op{}_ncv{}_npy{}'.format(args.cv_ratio, i, C.get()['source'],
                                                                          C.get()['target'], args.num_op, args.num_cv,
                                                                          args.num_policy)) for i in range(cv_num)]

    print(paths)
    reqs = [
        train_model.remote(copy.deepcopy(copied_c), args.dataroot, C.get()['aug'], args.cv_ratio, cv_num, i,
                           save_path=paths[i], skip_exist=True)
        for i in range(cv_num)]

    tqdm_epoch = tqdm(range(C.get()['epoch']))
    is_done = False
    for epoch in tqdm_epoch:
        time.sleep(15)
        while True:
            epochs_per_cv = OrderedDict()
            for cv_idx in range(cv_num):
                try:
                    latest_ckpt = torch.load(paths[cv_idx] + '.pth')
                    if 'epoch' not in latest_ckpt:
                        epochs_per_cv['cv%d' % (cv_idx + 1)] = C.get()['epoch']
                        continue
                    epochs_per_cv['cv%d' % (cv_idx + 1)] = latest_ckpt['epoch']
                except Exception as e:
                    continue
            tqdm_epoch.set_postfix(epochs_per_cv)
            if len(epochs_per_cv) == cv_num and min(epochs_per_cv.values()) >= C.get()['epoch']:
                is_done = True
            if len(epochs_per_cv) == cv_num and min(epochs_per_cv.values()) >= epoch:
                break
            # time.sleep(10)
        if is_done:
            break

    logger.info('getting results...')
    pretrain_results = ray.get(reqs)
    for r_model, r_cv, r_dict in pretrain_results:
        logger.info('model=%s cv=%d top1_train=%.4f top1_valid=%.4f' % (
            r_model, r_cv + 1, r_dict['top1_train'], r_dict['top1_valid']))
    logger.info('processed in %.4f secs' % w.pause('train_no_aug'))

    if args.until == 1:
        sys.exit(0)

    logger.info('----- Search Test-Time Augmentation Policies -----')
    w.start(tag='search')

    ops = augment_list(False)
    space = {}
    for i in range(args.num_policy):
        for j in range(args.num_op):
            space['policy_%d_%d' % (i, j)] = hp.choice('policy_%d_%d' % (i, j), list(range(0, len(ops))))
            space['prob_%d_%d' % (i, j)] = hp.uniform('prob_%d_ %d' % (i, j), 0.0, 1.0)
            space['level_%d_%d' % (i, j)] = hp.uniform('level_%d_ %d' % (i, j), 0.0, 1.0)

    final_policy_set = []
    total_computation = 0
    reward_attr = 'minus_loss'  # top1_valid or minus_loss
    for _ in range(1):  # run multiple times.
        for cv_fold in range(cv_num):
            name = "search_%s_%s_fold%d_ratio%.1f" % (
                C.get()['dataset'], C.get()['model']['type'], cv_fold, args.cv_ratio)
            print(name)
            register_trainable(name, lambda augs, rpt: eval_tta(copy.deepcopy(copied_c), augs, rpt))
            # register_trainable(name, eval_tta)
            algo = HyperOptSearch(space, max_concurrent=20, reward_attr=reward_attr)

            exp_config = {
                paths[cv_fold]: {
                    'run': name,
                    'num_samples': 4 if args.smoke_test else args.num_search,
                    'resources_per_trial': {'gpu': 1},
                    'stop': {'training_iteration': args.num_policy},
                    'config': {
                        'dataroot': os.path.abspath(args.dataroot), 'save_path': paths[cv_fold],
                        'cv_ratio_test': args.cv_ratio, 'cv_fold': cv_fold,
                        'num_op': args.num_op, 'num_policy': args.num_policy
                    },
                }
            }
            results = run_experiments(exp_config, search_alg=algo, scheduler=None, verbose=0, queue_trials=True,
                                      resume=args.resume, raise_on_failed_trial=False)
            print()
            results = [x for x in results if x.last_result is not None]
            results = sorted(results, key=lambda x: x.last_result[reward_attr], reverse=True)

            # calculate computation usage
            for result in results:
                total_computation += result.last_result['elapsed_time']

            for result in results[:num_result_per_cv]:
                final_policy = policy_decoder(result.config, args.num_policy, args.num_op)
                logger.info('loss=%.12f top1_valid=%.4f %s' % (
                    result.last_result['minus_loss'], result.last_result['top1_valid'], final_policy))

                final_policy = remove_deplicates(final_policy)
                final_policy_set.extend(final_policy)
    torch.save(final_policy_set,
               './aug_final/{}_{}2{}_op{}_ncv{}_npy{}_ns{}.pth'.format(args.dc_model, C.get()['source'],
                                                                       C.get()['target'],
                                                                       args.num_op, args.num_cv,
                                                                       args.num_policy, args.num_search))
