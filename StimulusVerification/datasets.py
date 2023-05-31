import numpy as np
import pickle
import torch
from torch.utils.data.dataset import Dataset
from utils import preprocess_context_data, filter_full_social_data
import copy
import os


class Context_Stimulus_Dataset(Dataset):
    def __init__(self, dataset_name, fig_size, split="train", data_folder=None, empty_discard_prob=0.9, pred_len=12):
        super(Context_Stimulus_Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.fig_size = fig_size
        self.half_size = fig_size // 2
        self.empty_discard_prob = empty_discard_prob
        self.split = split
        self.pred_len = pred_len
        processed_data, self.maps = preprocess_context_data(dataset_name, fig_size, split, data_folder)

        self.idx_cnt = 0
        self.index_data_dict = dict()
        self.discrete_trj_idx_register = []
        if self.split == 'train':
            for i, data_dict in enumerate(processed_data):
                for j, step_data in enumerate(data_dict["trj"]):
                    self.index_data_dict[self.idx_cnt] = {"pos": step_data[1:3], "vel": step_data[3:],
                                                          "map_id": data_dict["map_id"]}
                    self.idx_cnt += 1
        else:
            discrete_trajectory_cnt = 0
            for i, data_dict in enumerate(processed_data):
                if len(data_dict['trj']) >= self.pred_len:
                    for j, step_data in enumerate(data_dict["trj"]):
                        self.index_data_dict[self.idx_cnt] = {"pos": step_data[1:3], "vel": step_data[3:],
                                                              "map_id": data_dict["map_id"]}
                        self.idx_cnt += 1
                        self.discrete_trj_idx_register.append(discrete_trajectory_cnt)
                    discrete_trajectory_cnt += 1

        self.empty_thresh = self.fig_size ** 2 / 100

    def __len__(self):
        return self.idx_cnt

    def __getitem__(self, item):
        while True:
            data_piece = self.index_data_dict[item]
            x, y = np.around(data_piece["pos"]).astype(int)  # center of cropped semantic map
            map_id = data_piece["map_id"]
            cropped_semantic_map = self.maps[map_id][:, y - self.half_size:y + self.half_size,
                                                        x - self.half_size:x + self.half_size]

            # Manually discard semantic maps that are empty (highly frequent in certain cases and could cause unsatisfactory results)
            # under a certain probability.
            if self.split == 'train':
                if np.sum(cropped_semantic_map) <= self.empty_thresh:
                    if np.random.rand() > self.empty_discard_prob:
                        return_data = {"map_data": cropped_semantic_map,
                                       "velocity_data": data_piece["vel"] * (np.random.randn(2) / 20 + 1)}
                        break
                    else:
                        item = int(np.random.rand() * self.idx_cnt)
                else:
                    return_data = {"map_data": cropped_semantic_map,
                                   "velocity_data": data_piece["vel"] * (np.random.randn(2) / 20 + 1)}
                    break
            else:
                return_data = {"map_data": cropped_semantic_map, "velocity_data": data_piece["vel"]}
                break

        if self.split == "train":
            h_flip = np.random.rand() > 0.5
            v_flip = np.random.rand() > 0.5
            trp = np.random.rand() > 0.5
            if h_flip:
                return_data["map_data"] = np.flip(return_data["map_data"], axis=1).copy()
                return_data["velocity_data"][1] = -return_data["velocity_data"][1]
            if v_flip:
                return_data["map_data"] = np.flip(return_data["map_data"], axis=2).copy()
                return_data["velocity_data"][0] = -return_data["velocity_data"][0]
            if trp:
                return_data["map_data"] = np.transpose(return_data["map_data"], (0, 2, 1)).copy()
                return_data["velocity_data"] = np.flip(return_data["velocity_data"]).copy()

        return return_data

    def evaluate_avg_prob(self, all_log_prob):
        sum_avg_log_prob = 0
        total_effective_trj_num = 0
        assert len(all_log_prob) == self.idx_cnt
        prev_trj_idx = 0
        continuous_cnt = 0
        all_log_prob_copy = []
        for i in range(self.idx_cnt):
            all_log_prob_copy.append(all_log_prob[i])
            if self.discrete_trj_idx_register[i] == prev_trj_idx:
                continuous_cnt += 1
                if continuous_cnt >= self.pred_len:
                    sum_avg_log_prob += np.mean(all_log_prob_copy[-self.pred_len:])
                    total_effective_trj_num += 1
            else:
                continuous_cnt = 0
            prev_trj_idx = self.discrete_trj_idx_register[i]
        return sum_avg_log_prob / total_effective_trj_num


class Social_Stimulus_Dataset(Dataset):
    def __init__(self, dataset_name, split="train", rotation_aug=True):
        super(Social_Stimulus_Dataset, self).__init__()
        self.obs_len = 8
        self.pred_len = 12
        self.social_inclusion_thresh = 2.5

        self.dataset_name = dataset_name
        self.rotation_aug = rotation_aug
        self.split = split

        filtered_social_file_name = dataset_name + "_filtered_social_thresh_" + str(self.social_inclusion_thresh) + "_" + split + ".pkl"
        filtered_social_file_path = os.path.join("social_data", "filtered", filtered_social_file_name)
        preprocessed_social_data_path = os.path.join("social_data", "preprocessed", dataset_name + '_' + split + '_' + "social_info.pkl")
        with open(preprocessed_social_data_path, "rb") as f:
            preprocessed_full_social_data = pickle.load(f)

        if os.path.exists(filtered_social_file_path):
            with open(filtered_social_file_path, "rb") as f:
                self.filtered_social_data = pickle.load(f)
        else:
            self.filtered_social_data = filter_full_social_data(preprocessed_full_social_data,
                                                                thresh=self.social_inclusion_thresh,
                                                                obs_len=self.obs_len)
            with open(filtered_social_file_path, "wb") as f:
                pickle.dump(self.filtered_social_data, f, protocol=4)
                print("Filtered Social Saved.")

        self.size = len(self.filtered_social_data)
        self.full_social_data = preprocessed_full_social_data
        self.divisor = torch.arange(1, self.pred_len + 1).float().unsqueeze(-1)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        if self.split == 'train':
            angle = (np.random.rand() - 0.5) * np.pi * 2 if self.rotation_aug else 0
        else:
            angle = 0

        matrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
        relative_social = copy.deepcopy(self.filtered_social_data[item])
        for i in range(self.obs_len):
            assert len(relative_social[i]) > 0
            relative_social[i][:, :2] = np.matmul(matrix, relative_social[i][:, :2].transpose()).transpose()
            relative_social[i][:, 2:] = np.matmul(matrix, relative_social[i][:, 2:].transpose()).transpose()
            relative_social[i] = torch.from_numpy(relative_social[i]).float()

        future_trj = self.full_social_data[item]["ped_seq"][-self.pred_len:, 2:] - \
                     self.full_social_data[item]["ped_seq"][self.obs_len - 1:self.obs_len, 2:]
        future_trj = np.matmul(matrix, future_trj.transpose()).transpose()

        assert len(relative_social) == self.obs_len
        return {"social": relative_social, "trj": torch.from_numpy(future_trj).float() / self.divisor}

    def evaluate_avg_prob(self, all_log_prob):
        return np.mean(all_log_prob)
