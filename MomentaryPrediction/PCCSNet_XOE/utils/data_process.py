import os
import numpy as np
import torch
import pickle
import cmath
from PCCSNet_XOE.utils.checkings import *
from torch.nn.utils.rnn import pad_sequence


def get_data(dataset_name, source, obs_len, pred_len,
             rotate=False, flip_aug=True, file_dir="../dataset"):
    if source == "ethucy":
        file_names = ["eth", "hotel", "univ-001", "univ-003",
                      "univ-example", "zara1", "zara2", "zara3"]
        ref_table = {
            "eth": {"val": [0], "train": [1, 2, 3, 4, 5, 6, 7]},
            "hotel": {"val": [1], "train": [0, 2, 3, 4, 5, 6, 7]},
            "univ": {"val": [2, 3], "train": [0, 1, 4, 5, 6, 7]},
            "zara1": {"val": [5], "train": [0, 1, 2, 3, 4, 6, 7]},
            "zara2": {"val": [6], "train": [0, 1, 2, 3, 4, 5, 7]},
        }
        start_dim = 2
        input_dim = 2

    elif source == "SDD":
        file_names = ["train", "val"]
        ref_table = {
            "SDD": {"val": [1], "train": [0]}
        }
        start_dim = 0
        input_dim = 2
    else:
        raise NotImplementedError("Dataset Not Recognized")

    if dataset_name not in ref_table.keys():
        raise NotImplementedError("Dataset Not Recognized")

    train_data, val_data, split_marks = [], [], [0]
    s = 0

    for file_idx in ref_table[dataset_name]["train"]:
        target_file_name = file_names[file_idx] + "_" + str(obs_len) + "_" + str(pred_len) + ".npy"
        path = os.path.join(file_dir, source, "processed", target_file_name)

        trajectories = np.load(path)
        assert trajectories.shape[1] == obs_len + pred_len
        train_data.append(trajectories[:, :, start_dim:start_dim+input_dim])
        s += len(trajectories)
        split_marks.append(s)

        if flip_aug:
            flipped_data = np.flip(trajectories, axis=1)
            train_data.append(flipped_data[:, :, start_dim:start_dim+input_dim])
            s += len(flipped_data)
            split_marks.append(s)

    train_data = np.concatenate(train_data, axis=0)
    assert len(train_data) == s

    for file_idx in ref_table[dataset_name]["val"]:
        path = os.path.join(file_dir, source, "processed", file_names[file_idx] + "_" + str(obs_len) + "_" + str(pred_len) + ".npy")
        trajectories = np.load(path)
        assert trajectories.shape[1] == obs_len + pred_len
        val_data.append(trajectories[:, :, start_dim:start_dim+input_dim])

    val_data = np.concatenate(val_data, axis=0)

    normalized_train_data = train_data - train_data[:, :1, :]
    normalized_val_data = val_data - val_data[:, :1, :]
    normalized_train_data[:, -pred_len:, :] = normalized_train_data[:, -pred_len:, :] - normalized_train_data[:, obs_len - 1:obs_len, :]
    normalized_val_data[:, -pred_len:, :] = normalized_val_data[:, -pred_len:, :] - normalized_val_data[:, obs_len - 1:obs_len, :]

    if rotate:
        normalized_train_data = rotate_trajectory(normalized_train_data, sep=obs_len)
        normalized_val_data = rotate_trajectory(normalized_val_data, sep=obs_len)

    return train_data, val_data, normalized_train_data, normalized_val_data, split_marks


def get_test_data(file_names, obs_len, rotate=False):
    data = []
    for file_name in file_names:
        trajectories = np.load(file_name)
        shape = trajectories.shape
        assert len(shape) == 3 and shape[1] == obs_len and shape[2] == 2, "Invalid Input Format"
        data.append(trajectories)
    data = np.concatenate(data)

    normalized_data = data - data[:, :1, :]
    if rotate:
        normalized_data = rotate_trajectory(normalized_data, sep=obs_len)

    return data, normalized_data


def rotate_trajectory(data, sep=8):
    angles = []
    tg = sep - 1
    for _, trajectory in enumerate(data):
        rho, theta = cmath.polar(complex(trajectory[tg, 0], trajectory[tg, 1]))
        angles.append(theta)

    angles = np.array(angles)
    matrix = np.array([[np.cos(-angles), -np.sin(-angles)], [np.sin(-angles), np.cos(-angles)]])
    matrix = matrix.transpose((2, 0, 1))

    for i in range(len(data)):
        tmp = np.matmul(matrix[i], data[i].transpose()).transpose()
        assert tmp[tg, 0] >= 0
        data[i] = tmp
    return data


def transform_trajectory(data):
    if len(data.shape) == 4:
        bs, _, pre_len, _ = data.shape
        for j in range(1, pre_len):
            data[:, :, j, :] = data[:, :, j, :] + data[:, :, j - 1, :]
    elif len(data.shape) == 3:
        bs, pre_len, _ = data.shape
        for j in range(1, pre_len):
            data[:, j, :] = data[:, j, :] + data[:, j - 1, :]
    else:
        raise TypeError
    return data


def get_speed_and_angle(data, t):
    data_size = len(data)
    res = np.zeros((data_size, 2)).astype(float)  # Format: (speed, angle)
    for i in range(len(data)):
        d_x, d_y = data[i, t] - data[i, 0]
        res[i] = cmath.polar(complex(d_x, d_y))
    return res


def process_scene(dataset_name, train_set, args, save_dir="PCCSNet_XOE/processed_scene"):
    obs_len, _, source, rad = args.obs_len, args.pred_len, args.data_from, args.rad
    train_data, split_marks = train_set.trj_data.numpy(), train_set.split_marks
    res_dict = {}
    for split_idx in range(1, len(split_marks)):
        raw_data = train_data[split_marks[split_idx-1]: split_marks[split_idx]]
        last_coords = raw_data[:, obs_len-1]
        speed_and_angle = get_speed_and_angle(raw_data, obs_len-1)

        for i in range(len(raw_data)):
            last_coord = last_coords[i]
            speed_max = speed_and_angle[i][0] * 1.1
            speed_min = speed_and_angle[i][0] * 0.9
            nearby_ids = []
            for j in range(len(raw_data)):
                if j == i:
                    continue
                if np.linalg.norm(last_coord - last_coords[j]) <= rad:
                    if speed_and_angle[j][0] == 0 and speed_and_angle[i][0] == 0:
                        nearby_ids.append(split_marks[split_idx - 1] + j)
                    elif speed_min < speed_and_angle[j][0] < speed_max:
                        angle_diff_tmp = np.abs(speed_and_angle[i][1] - speed_and_angle[j][1])
                        angle_diff = np.min([angle_diff_tmp, np.pi * 2 - angle_diff_tmp])
                        if angle_diff < np.pi * 0.1:
                            nearby_ids.append(split_marks[split_idx - 1] + j)

            res_dict[split_marks[split_idx - 1] + i] = nearby_ids

        print("==> Scene %d with %d Trajectories Processed" % (split_idx, len(raw_data)))

    assert len(res_dict) == len(train_data)

    file_name = "scene_" + str(rad) + "_" + str(obs_len) + ".pkl"
    with open(os.path.join(save_dir, source, dataset_name, file_name), "wb") as f:
        pickle.dump(res_dict, f, 4)
    return res_dict


def gen_prob(scene_dict, cluster_result, save_dir):
    res_class, res_prob = [], []
    for i in range(len(cluster_result)):
        neighbor_list = scene_dict[i]
        neighbor_clusters = cluster_result[neighbor_list]
        tmp_dict = {cluster_result[i]: 1}
        for cluster in neighbor_clusters:
            tmp_dict[cluster] = tmp_dict.get(cluster, 0) + 1
        classes = np.zeros(len(tmp_dict)).astype(int)
        prob = np.zeros(len(tmp_dict)).astype(float)
        for idx, item in enumerate(tmp_dict.items()):
            classes[idx] = item[0]
            prob[idx] = item[1]
        prob /= (len(neighbor_list) + 1)
        res_class.append(classes)
        res_prob.append(prob)

    with open(os.path.join(save_dir, "probabilities.pkl"), "wb") as f:
        pickle.dump((res_class, res_prob), f, 4)

    return res_class, res_prob


def get_scene_data(dataset_name, data_set, cluster_result, save_dir, args):
    file_name = "scene_" + str(args.rad) + "_" + str(args.obs_len) + ".pkl"
    file_dir = os.path.join("PCCSNet_XOE/processed_scene", args.data_from, dataset_name, file_name)
    if os.path.exists(file_dir):
        with open(file_dir, "rb") as f:
            scene_dict = pickle.load(f)
    else:
        print("==> Processing Scene...")
        scene_dict = process_scene(dataset_name, data_set, args=args)

    file_dir = os.path.join(save_dir, "probabilities.pkl")
    if os.path.exists(file_dir) and check_prob_consistency(save_dir, args):
        with open(os.path.join(save_dir, "probabilities.pkl"), "rb") as f:
            gt_classes, gt_prob = pickle.load(f)
    else:
        gt_classes, gt_prob = gen_prob(scene_dict, cluster_result, save_dir)

    gt_classes = np.array(gt_classes)
    gt_prob = np.array(gt_prob)

    data_size = len(gt_classes)
    res = np.zeros((data_size, args.n_cluster)).astype(float)
    for i in range(data_size):
        res[i][gt_classes[i]] = gt_prob[i]

    return res


def collate_helper_AOE(data):
    obs_trj = [item["obs_trj"] for item in data]
    pred = [item["pred"] for item in data]
    obs_data = [item["obs_data"] for item in data]
    data_sizes = [item["sizes"] for item in data]
    obs_trj = pad_sequence(obs_trj, batch_first=True)
    pred = pad_sequence(pred, batch_first=True)
    data_sizes = pad_sequence(data_sizes, batch_first=True)
    seq_lens = [item["obs_trj"].shape[0] for item in data]

    return {"obs_trj": obs_trj, "pred": pred, "obs_data": obs_data,
            "data_sizes": data_sizes, "seq_lens": seq_lens}


def collate_helper_MOE(data):
    obs_trj = torch.stack([item["obs_trj"] for item in data])
    pred = torch.stack([item["pred"] for item in data])
    obs_data = [item["obs_data"] for item in data]
    data_sizes = torch.stack([item["sizes"] for item in data])

    return {"obs_trj": obs_trj, "pred": pred, "obs_data": obs_data,
            "data_sizes": data_sizes}


def sgn(num):
    if num > 0.0:
        return 1.0
    elif num == 0.0:
        return 0.0
    else:
        return -1.0
