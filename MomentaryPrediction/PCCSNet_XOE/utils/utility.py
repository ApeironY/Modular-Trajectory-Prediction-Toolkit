import os
import copy
import numpy as np
import torch
from sklearn.cluster import KMeans
from PCCSNet_XOE.utils.checkings import *
from torch.optim.lr_scheduler import LambdaLR


def create_folders(names, data_dir):
    datasets = ["ethucy", "SDD"]
    folders = {"ethucy": ["eth", "hotel", "univ", "zara1", "zara2"],
               "SDD": ["SDD"]}
    for folder_name in names:
        for dataset in datasets:
            for sub_folder in folders[dataset]:
                path = os.path.join(folder_name, dataset, sub_folder)
                if not os.path.exists(path):
                    os.makedirs(path)
            path = os.path.join(data_dir, dataset, "processed_xoe")
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


def gen_memory(data, model, save_dir=None, split=10, bi_shot=False, speed_data=None, scene_data=None):
    memory_data = []
    batch = len(data) // split
    if not bi_shot:
        for i in range(split - 1):
            tmp = model.calc_memory(data[batch * i:batch * (i + 1)]).numpy()
            memory_data.append(tmp)
        tmp = model.calc_memory(data[batch * (split - 1):]).numpy()
        memory_data.append(tmp)
    else:
        for i in range(split - 1):
            tmp = model.calc_memory(data[batch * i:batch * (i + 1)], speed_data[batch * i:batch * (i + 1)],
                                    scene_data[batch * i:batch * (i + 1)]).numpy()
            memory_data.append(tmp)
        tmp = model.calc_memory(data[batch * (split - 1):], speed_data[batch * (split - 1):],
                                scene_data[batch * (split - 1):]).numpy()
        memory_data.append(tmp)

    memory_data = np.concatenate(memory_data, axis=0)
    if save_dir is not None:
        np.save(os.path.join(save_dir, "Memory.npy"), memory_data)

    return memory_data


def get_memory_data(dataset, save_dir, args):
    if os.path.exists(os.path.join(save_dir, "memory.npy")) and check_memory_consistency(save_dir, args):
        memory_data = np.load(os.path.join(save_dir, "memory.npy"))
    else:
        print("==> Generating Memory...")
        memory_data = torch.cat([dataset.obs_enc, dataset.pred_enc], dim=1)
        memory_data = memory_data.cpu().numpy()
        np.save(os.path.join(save_dir, "memory.npy"), memory_data)

    print("==> Memory Shape", memory_data.shape)

    try:
        cluster_result = np.load(os.path.join(save_dir, "cluster_result.npy"))
        size_array = np.load(os.path.join(save_dir, "size_array.npy"))
        if not check_cluster_consistency(save_dir, size_array, args):
            raise FileNotFoundError
        if args.reC:
            raise FileNotFoundError

    except FileNotFoundError:
        args.reC = False
        print("==> Clustering Memory...")
        cluster_result, size_array = cluster_data(copy.deepcopy(memory_data), n_cluster=args.n_cluster,
                                                  save_dir=save_dir, args=args)

    return memory_data, cluster_result, size_array


def get_encoded_obs(net, data, scene, bi_shot):
    if not bi_shot:
        assert scene is None
        obs_encoded = net.calc_obs_memory(data)
    else:
        assert scene is not None
        obs_encoded = net.calc_obs_memory([data, scene], bi_shot=True)
    return obs_encoded


def get_modality(memory_data, cluster_result, num_cluster):
    cluster_dict = {}
    modalities = []
    for i in range(len(cluster_result)):
        cluster_dict[cluster_result[i]] = cluster_dict.get(cluster_result[i], []) + [i]
    for i in range(num_cluster):
        modalities.append(torch.mean(memory_data[cluster_dict[i]], dim=0))
    return torch.stack(modalities, dim=0)


def cluster_data(data, n_cluster=200, save_dir=None, args=None):
    if not isinstance(data, np.ndarray):
        data = np.array(data).astype(float)

    if args is not None:
        print("pm %.4f Applied" % args.pm)
        data[:, :args.obs_feat_size] *= args.pm

    cluster_result, size_array = k_means_cluster(data, n_cluster)
    if save_dir is not None:
        np.save(os.path.join(save_dir, "cluster_result.npy"), cluster_result)
        np.save(os.path.join(save_dir, "size_array.npy"), size_array)

    return cluster_result, size_array


def k_means_cluster(data, n_clusters, init=20):
    clustering = KMeans(n_clusters=n_clusters, n_init=init).fit(data)
    n_classes = len(clustering.cluster_centers_)
    res = dict()
    size_array = []
    for i in clustering.labels_:
        res[i] = res.get(i, 0) + 1
    for i in range(n_classes):
        size_array.append(res[i])

    cluster_result = np.array(clustering.labels_)  # cluster_result[i] -> class of item i
    size_array = np.array(size_array)  # size_array[i] -> #samples in cluster i

    return cluster_result, size_array


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
