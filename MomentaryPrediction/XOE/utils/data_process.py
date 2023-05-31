import numpy as np
import pickle
import os
import copy
import cmath
import torch
from torch.nn.utils.rnn import pad_sequence
from DataPreprocess.processor import eth_ucy_Processor
from DataPreprocess.processor import SDD_Processor


def rotate_trajectory(data, obs_len):
    """
    Rotate the trajectories so that they have the same directions
    :param data: shape (num_trajectories, total_length, (x, y, v_x, v_y))
    :param obs_len:
    :param last_state_only: If True, only uses the speed of the last observed state for rotation
    :return: rotated trajectories, shape (num_trajectories, total_length, (x, y, v_x, v_y))
    """
    assert np.sum(np.abs(data[:, obs_len - 1, :2])) == 0, np.sum(np.abs(data[:, obs_len - 1, :2]))  # Ensure zero End
    angles = []
    assert obs_len >= 2

    for trajectory in data:
        cmp = complex(-trajectory[0, 0], -trajectory[0, 1])  # Bi_Shot mode is included
        rho, theta = cmath.polar(cmp)
        angles.append(theta)

    angles = np.array(angles)
    matrix = np.array([[np.cos(-angles), -np.sin(-angles)],
                       [np.sin(-angles), np.cos(-angles)]]).transpose((2, 0, 1))

    for i, trajectory in enumerate(data):
        rotated_pos = np.matmul(matrix[i], trajectory[:, :2].transpose()).transpose()
        rotated_vel = np.matmul(matrix[i], trajectory[:, 2:].transpose()).transpose()
        data[i] = np.concatenate([rotated_pos, rotated_vel], axis=-1)

    return data


def rotate_trajectory_arbi(data, pred_len):
    """
    Rotate the trajectories so that they have the same directions
    :param data: shape (num_trajectories, total_length, (x, y, v_x, v_y))
    :param obs_len:
    :param last_state_only: If True, only uses the speed of the last observed state for rotation
    :return: rotated trajectories, shape (num_trajectories, total_length, (x, y, v_x, v_y))
    """
    for i in range(len(data)):
        assert np.sum(np.abs(data[i][-pred_len - 1, :2])) == 0, np.sum(np.abs(data[i][-pred_len - 1, :2]))
    angles = []
    assert pred_len <= 12

    for trajectory in data:
        cmp = complex(-trajectory[-pred_len - 2, 0], -trajectory[-pred_len - 2, 1])  # Bi_Shot mode is included
        rho, theta = cmath.polar(cmp)
        angles.append(theta)

    angles = np.array(angles)
    matrix = np.array([[np.cos(-angles), -np.sin(-angles)],
                       [np.sin(-angles), np.cos(-angles)]]).transpose((2, 0, 1))

    for i, trajectory in enumerate(data):
        rotated_pos = np.matmul(matrix[i], trajectory[:, :2].transpose()).transpose()
        rotated_vel = np.matmul(matrix[i], trajectory[:, 2:].transpose()).transpose()
        data[i] = np.concatenate([rotated_pos, rotated_vel], axis=-1)

    return data


def get_arbitary_data(args, dataset_name, split, skip=1, req_split=False):
    obs_len = 2
    pred_len = args.pred_len
    data_source = args.data_from
    flip_aug = args.flip_aug
    grid_num = args.grid_num
    grid_size = args.grid_size

    if data_source == "ethucy":
        processor = eth_ucy_Processor(grids_per_line=grid_num, grid_size=grid_size, add_empirical=args.add_empirical)
        file_names = ["eth", "hotel", "univ-001", "univ-003",
                      "univ-example", "zara1", "zara2", "zara3"]
        ref_table = {
            "eth": {"val": [0], "train": [1, 2, 3, 4, 5, 6, 7]},
            "hotel": {"val": [1], "train": [0, 2, 3, 4, 5, 6, 7]},
            "univ": {"val": [2, 3], "train": [0, 1, 4, 5, 6, 7]},
            "zara1": {"val": [5], "train": [0, 1, 2, 3, 4, 6, 7]},
            "zara2": {"val": [6], "train": [0, 1, 2, 3, 4, 5, 7]},
        }

    elif data_source == "SDD":
        processor = SDD_Processor(grids_per_line=grid_num, grid_size=grid_size, add_empirical=args.add_empirical)
        ref_table = {
            "SDD": {"val": [], "train": []}
        }

        scene_nums = {
            "bookstore": 7,
            "coupa": 4,
            "deathCircle": 5,
            "gates": 9,
            "hyang": 15,
            "little": 4,
            "nexus": 12,
            "quad": 4
        }
        cnt = 0
        file_names = []
        for name, num in scene_nums.items():
            for idx in range(num):
                file_name = name + "_" + str(idx)
                file_names.append(file_name)
                checklist = [
                    name == "coupa" and (idx == 0 or idx == 1),
                    name == "gates" and idx == 2,
                    name == "hyang" and idx in [0, 1, 3, 8],
                    name == "little" and 0 <= idx <= 4,
                    name == "nexus" and (idx == 5 or idx == 6),
                    name == "quad" and 0 <= idx <= 4,
                ]
                if np.sum(checklist) > 0:
                    ref_table["SDD"]["val"].append(cnt)
                else:
                    ref_table["SDD"]["train"].append(cnt)
                cnt += 1

    else:
        raise NotImplementedError

    if dataset_name not in ref_table.keys():
        print(ref_table, dataset_name)
        raise NotImplementedError

    total_len = obs_len + pred_len

    all_trj_data, all_formulated_data = [], []
    split_marks = [0]
    for file_idx in ref_table[dataset_name][split]:
        if split == "train" and flip_aug:
            rev_type = [True, False]
        else:
            rev_type = [False]

        for rev in rev_type:
            file_name = "%s_g_%d_s_%d_rev_%s_emp_%s" % (file_names[file_idx], grid_num, grid_size, str(rev), str(args.add_empirical)) + ".pkl"
            print("==> Loading File", file_name)
            path = os.path.join(args.data_dir, data_source, "processed_xoe", file_name)
            if not os.path.exists(path):
                print("==> File", path, "Does Not Exist, Regenerate")
                processor.gen_data(incl_rev=rev)

            with open(path, "rb") as f:
                full_data = pickle.load(f)

            for ped_id in full_data["trj_of_ped"].keys():
                trj = full_data["trj_of_ped"][ped_id]["seq"]
                formulated_data = full_data["data_of_ped"][ped_id]

                assert len(trj) == len(formulated_data)
                if len(trj) < total_len:
                    continue

                if args.data_type == "arbi8":
                    for i in range(8, len(trj) - 11, skip):
                        trj_slice = trj[0: i + 12]
                        all_trj_data.append(trj_slice[:, :5])
                        all_formulated_data.append(formulated_data[i-1])
                elif args.data_type == "arbi":
                    for i in range(2, len(trj) - 11, skip):
                        trj_slice = trj[0: i + 12]
                        all_trj_data.append(trj_slice[:, :5])
                        all_formulated_data.append(formulated_data[i-1])

            split_marks.append(len(all_trj_data))

    normalized_trj_data = []
    for i in range(len(all_trj_data)):
        tmp_data = copy.deepcopy(all_trj_data[i][:, 1:5])
        tmp_data[..., 0:2] = tmp_data[..., 0:2] - tmp_data[-pred_len-1: -pred_len, 0:2]
        normalized_trj_data.append(tmp_data)
    normalized_trj_data = rotate_trajectory_arbi(normalized_trj_data, pred_len)

    assert len(all_trj_data) == len(all_formulated_data)

    processed_data = {
        "trj_data": np.array(all_trj_data),
        "normalized_trj_data": normalized_trj_data,
        "formulated_data": all_formulated_data,
    }

    if req_split:
        return processed_data, split_marks
    return processed_data


def get_momentary_data(args, dataset_name, split, skip=1, req_split=False):
    obs_len = 2
    pred_len = args.pred_len
    data_source = args.data_from
    flip_aug = args.flip_aug
    grid_num = args.grid_num
    grid_size = args.grid_size

    if data_source == "ethucy":
        processor = eth_ucy_Processor(grids_per_line=grid_num, grid_size=grid_size)
        file_names = ["eth", "hotel", "univ-001", "univ-003",
                      "univ-example", "zara1", "zara2", "zara3"]
        ref_table = {
            "eth": {"val": [0], "train": [1, 2, 3, 4, 5, 6, 7]},
            "hotel": {"val": [1], "train": [0, 2, 3, 4, 5, 6, 7]},
            "univ": {"val": [2, 3], "train": [0, 1, 4, 5, 6, 7]},
            "zara1": {"val": [5], "train": [0, 1, 2, 3, 4, 6, 7]},
            "zara2": {"val": [6], "train": [0, 1, 2, 3, 4, 5, 7]},
        }

    elif data_source == "SDD":
        processor = SDD_Processor(grids_per_line=grid_num, grid_size=grid_size)
        ref_table = {
            "SDD": {"val": [], "train": []}
        }

        scene_nums = {
            "bookstore": 7,
            "coupa": 4,
            "deathCircle": 5,
            "gates": 9,
            "hyang": 15,
            "little": 4,
            "nexus": 12,
            "quad": 4
        }
        cnt = 0
        file_names = []
        for name, num in scene_nums.items():
            for idx in range(num):
                file_name = name + "_" + str(idx)
                file_names.append(file_name)
                checklist = [
                    name == "coupa" and (idx == 0 or idx == 1),
                    name == "gates" and idx == 2,
                    name == "hyang" and idx in [0, 1, 3, 8],
                    name == "little" and 0 <= idx <= 4,
                    name == "nexus" and (idx == 5 or idx == 6),
                    name == "quad" and 0 <= idx <= 4,
                ]
                if np.sum(checklist) > 0:
                    ref_table["SDD"]["val"].append(cnt)
                else:
                    ref_table["SDD"]["train"].append(cnt)
                cnt += 1

    else:
        raise NotImplementedError

    if dataset_name not in ref_table.keys():
        print(ref_table, dataset_name)
        raise NotImplementedError

    total_len = obs_len + pred_len

    all_trj_data, all_formulated_data = [], []
    split_marks = [0]
    for file_idx in ref_table[dataset_name][split]:
        if split == "train" and flip_aug:
            rev_type = [True, False]
        else:
            rev_type = [False]

        for rev in rev_type:
            file_name = "%s_g_%d_s_%d_rev_%s_emp_%s" % (file_names[file_idx], grid_num, grid_size, str(rev), str(args.add_empirical)) + ".pkl"
            print("==> Loading File", file_name)
            path = os.path.join(args.data_dir, data_source, "processed_xoe", file_name)
            if not os.path.exists(path):
                print("==> File", path, "Does Not Exist, Regenerate")
                processor.gen_data(incl_rev=rev)

            with open(path, "rb") as f:
                full_data = pickle.load(f)

            for ped_id in full_data["trj_of_ped"].keys():
                trj = full_data["trj_of_ped"][ped_id]["seq"]
                formulated_data = full_data["data_of_ped"][ped_id]

                assert len(trj) == len(formulated_data)
                if len(trj) < total_len:
                    continue

                for i in range(6, len(trj) + 1 - total_len, skip):
                    trj_slice = trj[i:i + total_len]
                    all_trj_data.append(trj_slice[:, :5])
                    all_formulated_data.append(formulated_data[i+1])

            split_marks.append(len(all_trj_data))

    all_trj_data = np.stack(all_trj_data)
    normalized_trj_data = copy.deepcopy(all_trj_data)
    normalized_trj_data[..., 1:3] = normalized_trj_data[..., 1:3] - normalized_trj_data[:, obs_len - 1:obs_len, 1:3]
    normalized_trj_data[..., 1:] = rotate_trajectory(normalized_trj_data[..., 1:], obs_len=obs_len)
    assert np.sum(normalized_trj_data[:, 0, 1]) == -np.sum(np.abs(normalized_trj_data[:, 0, 1])), \
        (np.sum(normalized_trj_data[:, 0, 1]), np.sum(np.abs(normalized_trj_data[:, 0, 1])))

    assert len(all_trj_data) == len(all_formulated_data)

    processed_data = {
        "trj_data": all_trj_data,
        "normalized_trj_data": normalized_trj_data,
        "formulated_data": all_formulated_data,
    }

    if req_split:
        return processed_data, split_marks
    return processed_data


def get_arbitrary_test_data(file_names, args, skip):
    obs_len = 2
    pred_len = args.pred_len
    data_source = args.data_from

    print("==> using skip %d" % skip)
    total_len = obs_len + pred_len
    all_trj_data, all_formulated_data = [], []
    for file_name in file_names:
        print("Loading", file_name)
        path = os.path.join(args.data_dir, data_source, "processed_xoe", file_name)
        with open(path, "rb") as f:
            full_data = pickle.load(f)

        for ped_id in full_data["trj_of_ped"].keys():
            trj = full_data["trj_of_ped"][ped_id]["seq"]
            formulated_data = full_data["data_of_ped"][ped_id]

            assert len(trj) == len(formulated_data)
            if len(trj) < total_len:
                continue

            if args.data_type == "arbi8":
                for i in range(8, len(trj) - 11, skip):
                    trj_slice = trj[0: i + 12]
                    all_trj_data.append(trj_slice[:, :5])
                    all_formulated_data.append(formulated_data[i-1])
            elif args.data_type == "arbi":
                for i in range(2, len(trj) - 11, skip):
                    trj_slice = trj[0: i + 12]
                    all_trj_data.append(trj_slice[:, :5])
                    all_formulated_data.append(formulated_data[i-1])

    normalized_trj_data = []
    for i in range(len(all_trj_data)):
        tmp_data = copy.deepcopy(all_trj_data[i][:, 1:5])
        tmp_data[..., 0:2] = tmp_data[..., 0:2] - tmp_data[-pred_len-1: -pred_len, 0:2]
        normalized_trj_data.append(tmp_data)
    normalized_trj_data = rotate_trajectory_arbi(normalized_trj_data, pred_len)

    assert len(all_trj_data) == len(all_formulated_data)

    processed_data = {
        "trj_data": np.array(all_trj_data),
        "normalized_trj_data": normalized_trj_data,
        "formulated_data": all_formulated_data,
    }

    return processed_data


def get_momentary_test_data(file_names, args, skip):
    obs_len = 2
    pred_len = args.pred_len
    data_source = args.data_from

    print("==> using skip %d" % skip)
    total_len = obs_len + pred_len
    all_trj_data, all_formulated_data = [], []
    for file_name in file_names:
        print("Loading", file_name)
        path = os.path.join(args.data_dir, data_source, "processed_xoe", file_name)
        with open(path, "rb") as f:
            full_data = pickle.load(f)

        for ped_id in full_data["trj_of_ped"].keys():
            trj = full_data["trj_of_ped"][ped_id]["seq"]
            formulated_data = full_data["data_of_ped"][ped_id]

            assert len(trj) == len(formulated_data)
            if len(trj) < total_len:
                continue

            for i in range(6, len(trj) + 1 - total_len, skip):
                trj_slice = trj[i:i + total_len]
                all_trj_data.append(trj_slice[:, :5])
                all_formulated_data.append(formulated_data[i + 1])

    all_trj_data = np.stack(all_trj_data)
    normalized_trj_data = copy.deepcopy(all_trj_data)
    normalized_trj_data[..., 1:3] = normalized_trj_data[..., 1:3] - normalized_trj_data[:, obs_len - 1:obs_len, 1:3]
    normalized_trj_data[..., 1:] = rotate_trajectory(normalized_trj_data[..., 1:], obs_len=obs_len)
    assert np.sum(normalized_trj_data[:, 0, 1]) == -np.sum(np.abs(normalized_trj_data[:, 0, 1])), \
        (np.sum(normalized_trj_data[:, 0, 1]), np.sum(np.abs(normalized_trj_data[:, 0, 1])))

    assert len(all_trj_data) == len(all_formulated_data)

    processed_data = {
        "trj_data": all_trj_data,
        "normalized_trj_data": normalized_trj_data,
        "formulated_data": all_formulated_data,
    }

    return processed_data


def collate_helper_aoe(data):
    obs_trj = [item["obs_trj"] for item in data]
    pred = [item["pred"] for item in data]
    obs_data = [item["obs_data"] for item in data]
    data_sizes = [item["sizes"] for item in data]
    scene_recon_gt = [item["scene_recon_gt"] for item in data]
    ped_recon_gt = [item["ped_recon_gt"] for item in data]
    seq_lens = [item["obs_trj"].shape[0] for item in data]
    obs_trj = pad_sequence(obs_trj, batch_first=True)
    pred = pad_sequence(pred, batch_first=True)
    data_sizes = pad_sequence(data_sizes, batch_first=True)
    scene_recon_gt = pad_sequence(scene_recon_gt, batch_first=True)
    ped_recon_gt = pad_sequence(ped_recon_gt, batch_first=True)

    return {"obs_trj": obs_trj, "pred": pred, "obs_data": obs_data,
            "data_sizes": data_sizes, "scene_recon_gt": scene_recon_gt, "ped_recon_gt": ped_recon_gt, "seq_lens": seq_lens}


def collate_helper_moe(data):
    obs_trj = torch.stack([item["obs_trj"] for item in data])
    pred = torch.stack([item["pred"] for item in data])
    obs_data = [item["obs_data"] for item in data]
    data_sizes = torch.stack([item["sizes"] for item in data])
    scene_recon_gt = torch.stack([item["scene_recon_gt"] for item in data])
    ped_recon_gt = torch.stack([item["ped_recon_gt"] for item in data])

    return {"obs_trj": obs_trj, "pred": pred, "obs_data": obs_data,
            "data_sizes": data_sizes, "scene_recon_gt": scene_recon_gt, "ped_recon_gt": ped_recon_gt}


def transform_trajectory(data):
    if len(data.shape) == 4:
        _, _, pre_len, _ = data.shape
        for j in range(1, pre_len):
            data[:, :, j, :] = data[:, :, j, :] + data[:, :, j - 1, :]
    elif len(data.shape) == 3:
        _, pre_len, _ = data.shape
        for j in range(1, pre_len):
            data[:, j, :] = data[:, j, :] + data[:, j - 1, :]
    else:
        raise TypeError
    return data
