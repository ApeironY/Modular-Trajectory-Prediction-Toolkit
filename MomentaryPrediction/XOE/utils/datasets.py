import torch
from torch.utils.data import Dataset


class AOE_Trainer_Dataset(Dataset):
    def __init__(self, pred_len, trj_data, normalized_trj_data=None, formulated_data=None, grid_num=6, add_empirical=False):
        super().__init__()
        self.obs_len = 2
        self.pred_len = pred_len
        self.trj_data = trj_data
        self.normalized_trj_data = normalized_trj_data
        self.formulated_data = formulated_data
        self.grid_num = grid_num
        if add_empirical:
            self.input_dim = 6
        else:
            self.input_dim = 5

    def __len__(self):
        return len(self.trj_data)

    def __getitem__(self, item):
        obs_data = self.formulated_data[item]  # 6 * 6 * ?
        sizes = torch.zeros((self.grid_num, self.grid_num)).int()
        flattened_obs = []
        scene_recon_gt = torch.zeros(self.grid_num, self.grid_num).long()
        ped_recon_gt = torch.zeros(self.grid_num, self.grid_num).long()
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                obs_patch = torch.tensor(obs_data[i][j]).float()
                obs_patch_len = len(obs_patch)
                assert obs_patch_len % self.input_dim == 0
                if obs_patch_len > self.input_dim or torch.sum(torch.abs(obs_patch[1:5])) != 0:
                    ped_recon_gt[i, j] = 1
                scene_recon_gt[i, j] = obs_patch[0]
                sizes[i, j] = obs_patch_len / self.input_dim
                flattened_obs.append(obs_patch)

        flattened_obs = torch.cat(flattened_obs, dim=0).reshape(-1, self.input_dim)

        return {
            "obs_trj": self.normalized_trj_data[item][:-self.pred_len][..., 0:4],
            "obs_data": flattened_obs,
            "sizes": sizes.reshape(-1),
            "pred": self.normalized_trj_data[item][-self.pred_len:][..., 0:2],  # / 30
            "scene_recon_gt": scene_recon_gt.reshape(-1),
            "ped_recon_gt": ped_recon_gt.reshape(-1)
        }


class MOE_Trainer_Dataset(Dataset):
    def __init__(self, pred_len, trj_data, normalized_trj_data=None, formulated_data=None, grid_num=6):
        super().__init__()
        self.obs_len = 2
        self.pred_len = pred_len
        self.trj_data = trj_data
        self.normalized_trj_data = normalized_trj_data
        self.formulated_data = formulated_data
        self.grid_num = grid_num

    def __len__(self):
        return len(self.trj_data)

    def __getitem__(self, item):
        obs_data = self.formulated_data[item]  # 6 * 6 * ?
        sizes = torch.zeros((self.grid_num, self.grid_num)).int()
        flattened_obs = []
        scene_recon_gt = torch.zeros(self.grid_num, self.grid_num).long()
        ped_recon_gt = torch.zeros(self.grid_num, self.grid_num).long()
        for i in range(self.grid_num):
            for j in range(self.grid_num):
                obs_patch = torch.tensor(obs_data[i][j]).float()
                obs_patch_len = len(obs_patch)
                assert obs_patch_len % 5 == 0
                if obs_patch_len > 5 or torch.sum(torch.abs(obs_patch[1:5])) != 0:
                    ped_recon_gt[i, j] = 1
                scene_recon_gt[i, j] = obs_patch[0]
                sizes[i, j] = obs_patch_len / 5
                flattened_obs.append(obs_patch)

        flattened_obs = torch.cat(flattened_obs, dim=0).reshape(-1, 5)

        return {
            "obs_trj": self.normalized_trj_data[item][:self.obs_len][..., 1:5],
            "obs_data": flattened_obs,
            "sizes": sizes.reshape(-1),
            "pred": self.normalized_trj_data[item][self.obs_len:][..., 1:3],  # / 30
            "scene_recon_gt": scene_recon_gt.reshape(-1),
            "ped_recon_gt": ped_recon_gt.reshape(-1)
        }
