import os
import torch
import copy
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def get_lr_scheduler(lr_policy, optimizer, max_iter=None):
    if lr_policy['name'] == "Poly":
        assert max_iter > 0
        num_groups = len(optimizer.param_groups)

        def lambda_f(cur_iter):
            return (1 - (cur_iter * 1.0) / max_iter) ** lr_policy['power']

        scheduler = LambdaLR(optimizer, lr_lambda=[lambda_f] * num_groups)
    else:
        raise NotImplementedError("lr policy not supported")

    return scheduler


def displacement_error(pred, pred_gt, mode='raw'):
    assert pred_gt.shape == pred.shape
    loss = pred_gt - pred
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred, pred_gt, mode='raw'):
    assert pred_gt.shape == pred.shape
    loss = pred_gt - pred
    loss = loss ** 2
    loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def create_folders(names):
    datasets = ["ethucy", "SDD"]
    folders = {"ethucy": ["eth", "hotel", "univ", "zara1", "zara2"],
               "SDD": ["SDD"]}
    for folder_name in names:
        for dataset in datasets:
            for sub_folder in folders[dataset]:
                path = os.path.join(folder_name, dataset, sub_folder)
                if not os.path.exists(path):
                    os.makedirs(path)


def display_performance(perf_dict):
    print("==> Current Performances (ADE & FDE):")
    for a, b in perf_dict.items():
        c = copy.deepcopy(b)
        if a in ["Intention"]:
            c[0] = np.round(c[0], 4)
            print("   ", a, c)
        elif a in ["Obs_Encoder", "Pred_Encoder"]:
            c[0] = np.round(c[0], 4)
            c[1] = np.round(c[1], 4)
            print("   ", a, c)
        else:
            c[-1][0] = np.round(c[-1][0], 4)
            c[-1][1] = np.round(c[-1][1], 4)
            print("   ", a, c[-1])
