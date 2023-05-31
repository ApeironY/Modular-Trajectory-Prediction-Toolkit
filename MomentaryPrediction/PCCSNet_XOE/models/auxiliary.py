import cmath
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PCCSNet_XOE.models.components import EncoderLSTM, DecoderLSTM
from XOE.models.xoe import XOE


class EncoderTrainer(nn.Module):
    def __init__(self, obs_len=8, pre_len=12, hidden=48, num_layer=3):
        super(EncoderTrainer, self).__init__()

        self.obs_len = obs_len
        self.pre_len = pre_len
        self.target_len = None
        self.hidden_size = hidden // 2
        self.num_layer = num_layer
        self.encoder = EncoderLSTM(2, self.hidden_size, self.num_layer)
        self.decoder = DecoderLSTM(2 * self.hidden_size, self.num_layer)
        self.fc = nn.Linear(2 * self.hidden_size, 2)

    def forward(self, trajectory_data):
        bs, total_len, fea_len = trajectory_data.shape
        output = []
        data = trajectory_data.transpose(0, 1)
        assert len(data) == self.target_len

        encoder_hidden = self.encoder.initHidden(bs)
        trj_encoded, _ = self.encoder(data, encoder_hidden)
        outs = trj_encoded[-1].unsqueeze(0)
        decoder_hidden = self.decoder.initHidden(bs)

        for i in range(self.pre_len):
            outs, decoder_hidden = self.decoder(outs, decoder_hidden)
            output.append(outs)

        output = torch.cat(output, 0)
        output = self.fc(output)
        output = output.transpose(0, 1)

        return output


class ObsEncoderTrainer(EncoderTrainer):
    def __init__(self, obs_len=8, pre_len=12, hidden_size=48, num_layer=3):
        super(ObsEncoderTrainer, self).__init__(obs_len, pre_len, hidden_size, num_layer)
        self.target_len = obs_len


class PredEncoderTrainer(EncoderTrainer):
    def __init__(self, obs_len=8, pre_len=12, hidden_size=48, num_layer=3):
        super(PredEncoderTrainer, self).__init__(obs_len, pre_len, hidden_size, num_layer)
        self.target_len = pre_len


class XOE_Trainer(nn.Module):
    def __init__(self, args, num_labels=6, input_dim=5, decoder_layer=3, hidden_size=48, num_layer=3):
        super().__init__()
        self.hidden_size = hidden_size // 2
        self.num_layer = num_layer
        self.xoe = XOE(args, num_labels, input_dim)
        self.obs_feat_size = args.obs_feat_size
        self.grid_num = args.grid_num
        self.num_labels = num_labels
        self.pred_len = args.pred_len

        self.decoder = DecoderLSTM(self.obs_feat_size, decoder_layer)
        self.fc = nn.Linear(self.obs_feat_size, 2)

    def forward(self, obs_trj, obs_data, data_sizes, seq_lens=None):
        bs = len(obs_trj)
        output = []
        cross_embedding = self.xoe(obs_trj, obs_data, data_sizes, seq_lens)

        outs = cross_embedding.unsqueeze(0)
        decoder_hidden = self.decoder.initHidden(bs)
        for i in range(self.pred_len):
            outs, decoder_hidden = self.decoder(outs, decoder_hidden)
            output.append(outs)

        output = torch.cat(output, 0)
        output = self.fc(output)
        output = output.transpose(0, 1)

        return output


class PCCS_Dataset(Dataset):
    def __init__(self, obs_len, pred_len, trj_data,
                 normalized_trj_data=None, split_marks=None):
        super(PCCS_Dataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.trj_data = trj_data
        self.normalized_trj_data = normalized_trj_data
        self.split_marks = split_marks

        self.obs_enc = None
        self.pred_enc = None
        self.cluster_results = None
        self.classifier_gt = None
        self.classifier_weights = None
        self.current_mode = None

    def set_mode(self, mode):
        assert mode in [0, 1, 2, 3, 4, 5, 6]
        self.current_mode = mode

    def __len__(self):
        return len(self.trj_data)

    def __getitem__(self, item):
        if self.current_mode == 0:
            return {
                "input": self.normalized_trj_data[item][:self.obs_len],
                "future": self.normalized_trj_data[item][self.obs_len:]
            }

        elif self.current_mode == 1:
            return {
                "input": self.normalized_trj_data[item][self.obs_len:],
                "future": self.normalized_trj_data[item][self.obs_len:]
            }

        elif self.current_mode == 2:
            assert self.obs_enc is not None and self.pred_enc is not None
            return {
                "obs_enc": self.obs_enc[item],
                "pred_enc": self.pred_enc[item],
                "future": self.normalized_trj_data[item][self.obs_len:]
            }

        elif self.current_mode == 3:
            assert self.obs_enc is not None and self.pred_enc is not None
            assert self.cluster_results is not None
            return {
                "obs_enc": self.obs_enc[item],
                "pred_enc": self.pred_enc[item],
                "cluster_idx": self.cluster_results[item],
            }

        elif self.current_mode == 4:
            assert self.obs_enc is not None
            return {
                "obs_enc": self.obs_enc[item],
                "prob": self.classifier_gt[item],
            }

        elif self.current_mode == 5:
            assert self.obs_enc is not None
            return {
                "obs_enc": self.obs_enc[item],
                "pred": self.normalized_trj_data[item][self.obs_len:]
            }

        elif self.current_mode == 6:
            assert self.obs_enc is not None
            pos = self.trj_data[item][-1] - self.trj_data[item][0]
            _, theta = cmath.polar(complex(pos[0], pos[1]))
            matrix = torch.tensor([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]]).float()
            return {
                "obs_enc": self.obs_enc[item],
                "rotate_mat": matrix,
                "last_obs": self.trj_data[item][self.obs_len - 1]
            }


class PCCS_AOE_Dataset(Dataset):
    def __init__(self, pred_len, trj_data, formulated_obs=None, grid_num=6, input_dim=5,
                 normalized_trj_data=None, split_marks=None):
        super(PCCS_AOE_Dataset, self).__init__()
        self.pred_len = pred_len
        self.trj_data = trj_data
        self.normalized_trj_data = normalized_trj_data
        self.split_marks = split_marks

        self.formulated_obs = formulated_obs
        self.grid_num = grid_num
        self.input_dim = input_dim

        self.obs_enc = None
        self.pred_enc = None
        self.cluster_results = None
        self.classifier_gt = None
        self.classifier_weights = None
        self.current_mode = None

    def set_mode(self, mode):
        assert mode in [0, 1, 2, 3, 4, 5, 6]
        self.current_mode = mode

    def __len__(self):
        return len(self.trj_data)

    def __getitem__(self, item):
        if self.current_mode == 0:
            obs_data = self.formulated_obs[item]  # 6 * 6 * ?
            sizes = torch.zeros((self.grid_num, self.grid_num)).int()
            flattened_obs = []
            for i in range(self.grid_num):
                for j in range(self.grid_num):
                    obs_patch = torch.tensor(obs_data[i][j]).float()
                    obs_patch_len = len(obs_patch)
                    assert obs_patch_len % self.input_dim == 0
                    sizes[i, j] = obs_patch_len / self.input_dim
                    flattened_obs.append(obs_patch)

            flattened_obs = torch.cat(flattened_obs, dim=0).reshape(-1, self.input_dim)

            return {
                "obs_trj": self.normalized_trj_data[item][:-self.pred_len][..., 0:4],
                "obs_data": flattened_obs,
                "sizes": sizes.reshape(-1),
                "pred": self.normalized_trj_data[item][-self.pred_len:][..., 0:2],  # / 30
            }

        elif self.current_mode == 1:
            return {
                "input": self.normalized_trj_data[item][-self.pred_len:][..., 0:2],
                "future": self.normalized_trj_data[item][-self.pred_len:][..., 0:2]
            }

        elif self.current_mode == 2:
            assert self.obs_enc is not None and self.pred_enc is not None
            return {
                "obs_enc": self.obs_enc[item],
                "pred_enc": self.pred_enc[item],
                "future": self.normalized_trj_data[item][-self.pred_len:][..., 0:2]
            }

        elif self.current_mode == 3:
            assert self.obs_enc is not None and self.pred_enc is not None
            assert self.cluster_results is not None
            return {
                "obs_enc": self.obs_enc[item],
                "pred_enc": self.pred_enc[item],
                "cluster_idx": self.cluster_results[item],
            }

        elif self.current_mode == 4:
            assert self.obs_enc is not None
            return {
                "obs_enc": self.obs_enc[item],
                "prob": self.classifier_gt[item],
            }

        elif self.current_mode == 5:
            assert self.obs_enc is not None
            return {
                "obs_enc": self.obs_enc[item],
                "pred": self.normalized_trj_data[item][-self.pred_len:][..., 0:2]
            }

        elif self.current_mode == 6:
            assert self.obs_enc is not None
            pos = self.trj_data[item][-1] - self.trj_data[item][0]
            _, theta = cmath.polar(complex(pos[0], pos[1]))
            matrix = torch.tensor([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]]).float()
            return {
                "obs_enc": self.obs_enc[item],
                "rotate_mat": matrix,
                "last_obs": self.trj_data[item][-self.pred_len - 1]
            }


class PCCS_MOE_Dataset(Dataset):
    def __init__(self, obs_len, pred_len, trj_data, formulated_obs=None, grid_num=6, input_dim=5,
                 normalized_trj_data=None, split_marks=None):
        super(PCCS_MOE_Dataset, self).__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.trj_data = trj_data[..., 1:3]
        self.normalized_trj_data = normalized_trj_data
        self.split_marks = split_marks

        self.formulated_obs = formulated_obs
        self.grid_num = grid_num
        self.input_dim = input_dim

        self.obs_enc = None
        self.pred_enc = None
        self.cluster_results = None
        self.classifier_gt = None
        self.classifier_weights = None
        self.current_mode = None

    def set_mode(self, mode):
        assert mode in [0, 1, 2, 3, 4, 5, 6]
        self.current_mode = mode

    def __len__(self):
        return len(self.trj_data)

    def __getitem__(self, item):
        if self.current_mode == 0:
            obs_data = self.formulated_obs[item]  # 6 * 6 * ?
            sizes = torch.zeros((self.grid_num, self.grid_num)).int()
            flattened_obs = []
            for i in range(self.grid_num):
                for j in range(self.grid_num):
                    obs_patch = torch.tensor(obs_data[i][j]).float()
                    obs_patch_len = len(obs_patch)
                    assert obs_patch_len % self.input_dim == 0
                    sizes[i, j] = obs_patch_len / self.input_dim
                    flattened_obs.append(obs_patch)

            flattened_obs = torch.cat(flattened_obs, dim=0).reshape(-1, self.input_dim)

            return {
                "obs_trj": self.normalized_trj_data[item][:self.obs_len][..., 1:5],
                "obs_data": flattened_obs,
                "sizes": sizes.reshape(-1),
                "pred": self.normalized_trj_data[item][self.obs_len:][..., 1:3],  # / 30
            }

        elif self.current_mode == 1:
            return {
                "input": self.normalized_trj_data[item][self.obs_len:][..., 1:3],
                "future": self.normalized_trj_data[item][self.obs_len:][..., 1:3]
            }

        elif self.current_mode == 2:
            assert self.obs_enc is not None and self.pred_enc is not None
            return {
                "obs_enc": self.obs_enc[item],
                "pred_enc": self.pred_enc[item],
                "future": self.normalized_trj_data[item][self.obs_len:][..., 1:3]
            }

        elif self.current_mode == 3:
            assert self.obs_enc is not None and self.pred_enc is not None
            assert self.cluster_results is not None
            return {
                "obs_enc": self.obs_enc[item],
                "pred_enc": self.pred_enc[item],
                "cluster_idx": self.cluster_results[item],
            }

        elif self.current_mode == 4:
            assert self.obs_enc is not None
            return {
                "obs_enc": self.obs_enc[item],
                "prob": self.classifier_gt[item],
            }

        elif self.current_mode == 5:
            assert self.obs_enc is not None
            return {
                "obs_enc": self.obs_enc[item],
                "pred": self.normalized_trj_data[item][self.obs_len:][..., 1:3]
            }

        elif self.current_mode == 6:
            assert self.obs_enc is not None
            pos = self.trj_data[item][-1] - self.trj_data[item][0]
            _, theta = cmath.polar(complex(pos[0], pos[1]))
            matrix = torch.tensor([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]]).float()
            return {
                "obs_enc": self.obs_enc[item],
                "rotate_mat": matrix,
                "last_obs": self.trj_data[item][self.obs_len - 1]
            }
