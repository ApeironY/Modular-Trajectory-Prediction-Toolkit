import numpy as np
import torch
import os
from torch.optim.lr_scheduler import LambdaLR


def preprocess_context_data(dataset_name, pad_size, split="train", data_folder=None):
    """
        map format
        -------------------------------
        |     |                       |
        |     y                       |
        |     |                       |
        |--x--*                       |
        |                             |
        -------------------------------
    """
    processed_data = []
    maps = []
    print("processing dataset", dataset_name, split)
    from dataset_info import file_names, train_val_reference, scene_range_reference

    map_id_cnt = 0
    for file_idx in train_val_reference[dataset_name][split]:
        name = file_names[file_idx]
        target_file_name = name + ".csv"
        path = os.path.join("..", "dataset", data_folder, "raw", target_file_name)
        trajectories = np.genfromtxt(path, delimiter=",").T
        step_size = 6 if name == "eth" else 10

        scene_range = scene_range_reference[name]
        scene_semantic_map = np.load(os.path.join("..", "dataset", data_folder, "semantic_maps", name + ".npy"))
        pixels_y, pixels_x = scene_semantic_map.shape
        trajectories[:, 2] = (trajectories[:, 2] - scene_range[0][0]) / (
                    scene_range[0][1] - scene_range[0][0]) * pixels_x
        trajectories[:, 3] = (trajectories[:, 3] - scene_range[1][0]) / (
                    scene_range[1][1] - scene_range[1][0]) * pixels_y
        agent_ids = np.unique(trajectories[:, 1])
        for agent_id in agent_ids:
            agent_trj = trajectories[np.where(trajectories[:, 1] == agent_id)]
            prev_cut = 0
            for j in range(len(agent_trj) - 1):
                if agent_trj[j - 1, 0] + step_size != agent_trj[j, 0]:
                    new_agent_trj = agent_trj[prev_cut:j]
                    prev_cut = j
                    if len(new_agent_trj) > 1:
                        agent_spd = np.zeros((len(new_agent_trj), 2)).astype(float)
                        agent_spd[:-1] = new_agent_trj[1:, 2:4] - new_agent_trj[:-1, 2:4]
                        agent_data = np.concatenate([new_agent_trj, agent_spd], axis=1)[:, [0, 2, 3, 4, 5]]
                        processed_data.append({"trj": agent_data, "map_id": file_idx})

            agent_trj = agent_trj[prev_cut:]
            if len(agent_trj) > 1:
                agent_spd = np.zeros((len(agent_trj), 2)).astype(float)
                agent_spd[:-1] = agent_trj[1:, 2:4] - agent_trj[:-1, 2:4]
                agent_data = np.concatenate([agent_trj, agent_spd], axis=1)[:, [0, 2, 3, 4, 5]]
                processed_data.append({"trj": agent_data, "map_id": map_id_cnt})

        scene_semantic_map = scene_semantic_map.astype(float)
        maps.append(scene_semantic_map[None, ...])
        map_id_cnt += 1

    if pad_size > 0:
        for i, data_dict in enumerate(processed_data):
            processed_data[i]["trj"][:, 1:3] = processed_data[i]["trj"][:, 1:3] + pad_size
        for i, scene_map in enumerate(maps):
            maps[i] = np.pad(scene_map, ((0, 0), (pad_size, pad_size), (pad_size, pad_size)))

    return processed_data, maps


def filter_full_social_data(social_data, thresh, obs_len):
    res = []
    print("Filtering Full Social Data ...")
    for i in range(len(social_data)):
        if i % 1000 == 0:
            print(i, '/', len(social_data))
        data = social_data[i]
        all_social_location = data['location']
        ped_seq = data['ped_seq']

        try:
            num_peds = all_social_location.shape[1]
        except AttributeError:
            num_peds = 0

        if num_peds > 0:
            all_social_speed = np.zeros_like(all_social_location)[:, :, :2]
            all_social_speed[1:] = all_social_location[1:, :, 1:] - all_social_location[:-1, :, 1:]
            all_social_speed[0] = all_social_speed[1]
            all_social_speed[np.where(np.isnan(all_social_speed))] = 0
            all_social_info = np.concatenate([all_social_location, all_social_speed], axis=-1)

        self_velocity = np.zeros_like(ped_seq)[:obs_len, :2]
        self_velocity[1:] = ped_seq[1:obs_len, 2:] - ped_seq[0:obs_len - 1, 2:]
        self_velocity[0] = self_velocity[1]
        self_location_seq = ped_seq[:obs_len, 1:]
        self_info = np.concatenate([self_location_seq, self_velocity], axis=-1)  # obs_len * 5
        filtered_social = []
        for step in range(obs_len):
            all_social_data_at_step = [self_info[step]]
            for j in range(num_peds):
                if not np.any(np.isnan(all_social_location[step, j])):
                    all_social_data_at_step = all_social_data_at_step + [all_social_info[step, j]]
            all_social_data_at_step = np.array(all_social_data_at_step)[:, 1:]  # N+1 * 4
            all_social_data_at_step[1:, :2] -= all_social_data_at_step[:1, :2]
            all_social_data_at_step[0, :2] = 0
            norm = np.linalg.norm(all_social_data_at_step[:, :2], axis=1)
            loc = np.where(norm < thresh)
            filtered_social_data_at_step = all_social_data_at_step[loc]
            assert len(filtered_social_data_at_step) > 0
            filtered_social.append(filtered_social_data_at_step)
        res.append(filtered_social)

    return res


def get_lr_scheduler(optimizer, lr_policy=None, max_iter=None):
    if lr_policy is None:
        lr_policy = {"name": "Poly", "power": 0.95}

    if lr_policy['name'] == "Poly":
        assert max_iter > 0
        num_groups = len(optimizer.param_groups)

        def lambda_f(cur_iter):
            return (1 - (cur_iter * 1.0) / max_iter) ** lr_policy['power']

        scheduler = LambdaLR(optimizer, lr_lambda=[lambda_f] * num_groups)
    else:
        raise NotImplementedError("lr policy not supported")
    return scheduler


def social_collate_helper(data):
    trj = torch.stack([item["trj"] for item in data])
    split_sizes = []
    social_info = []
    for item in data:
        for ii in range(len(item['social'])):
            split_sizes.append(len(item['social'][ii]))
            social_info.append(item['social'][ii])
    social_info = torch.cat(social_info, dim=0)
    return {"normalized_trj_data": trj, "concatenated_social_data": social_info, "split_sizes": split_sizes}

