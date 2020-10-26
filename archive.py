from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from FastAutoAugment.utils.point_augmentations import get_augment, augment_list



def float_parameter(level, maxval):
    return float(level) * maxval / PARAMETER_MAX


def int_parameter(level, maxval):
    return int(float_parameter(level, maxval))


def no_duplicates(f):
    def wrap_remove_duplicates():
        policies = f()
        return remove_deplicates(policies)

    return wrap_remove_duplicates


def remove_deplicates(policies):
    s = set()
    new_policies = []
    for ops in policies:
        key = []
        for op in ops:
            key.append(op[0])
        key = '_'.join(key)
        if key in s:
            continue
        else:
            s.add(key)
            new_policies.append(ops)

    return new_policies


def policy_decoder(augment, num_policy, num_op):
    op_list = augment_list(False)
    policies = []
    for i in range(num_policy):
        ops = []
        for j in range(num_op):
            op_idx = augment['policy_%d_%d' % (i, j)]
            op_prob = augment['prob_%d_%d' % (i, j)]
            op_level = augment['level_%d_%d' % (i, j)]
            ops.append((op_list[op_idx][0].__name__, op_prob, op_level))
        policies.append(ops)
    return policies
