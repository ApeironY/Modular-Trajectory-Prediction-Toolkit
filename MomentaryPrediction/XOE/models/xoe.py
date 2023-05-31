import torch
import torch.nn as nn
import os
import numpy as np
from XOE.models.auxiliary import Transformer, Mask_Trajectory_Encoder, DecoderLSTM, EncoderLSTM, ArbiEncoderLSTM


class In_Patch_Aggregator(nn.Module):
    def __init__(self, args, input_dim=5, mid_dim=16, output_dim=16, grid_num=6):
        super().__init__()
        self.grid_num = grid_num
        if args.use_AOE and args.add_empirical:
            input_dim = 6
        else:
            input_dim = 5
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, output_dim)
        )

    def forward(self, data, sizes):  # data: N * 5, sizes: sum == N
        embedded_data = self.mlp(data)
        embedded_data = torch.split(embedded_data, sizes, dim=0)
        res = []
        for i in range(len(sizes)):
            res.append(torch.max(embedded_data[i], dim=0)[0])
        res = torch.stack(res)
        return res


class Cross_Patch_Aggregator(nn.Module):
    def __init__(self, args, grid_size, patch_dim, dim, depth, heads, mlp_dim, pool='cls', dim_head=64):
        super().__init__()
        self.args = args
        self.num_patches = grid_size
        self.patch_dim = patch_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim).cuda(), requires_grad=True)
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.vel_to_embedding = nn.Linear(1, dim)
        if args.use_AOE:
            historical_dim = 1
            if args.add_behavior:
                self.behavior_to_embedding = ArbiEncoderLSTM(2, dim // 2, 3)
                historical_dim += 1
            if args.add_intention:
                self.intention_to_embedding = ArbiEncoderLSTM(2, dim // 2, 3)
                historical_dim += 1
            if historical_dim > 1:
                self.historical_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
                self.historical_pos_embedding = nn.Parameter(torch.randn(1, historical_dim, dim).cuda(), requires_grad=True)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, data, trj_data, seq_lens=None, mask=None):  # data shape: bs * grid_size * patch_dim
        bs = data.shape[0]
        data = data.reshape(bs, self.num_patches, self.patch_dim)
        data = self.patch_to_embedding(data)

        if not self.args.use_AOE:
            vel_token = self.vel_to_embedding(trj_data[:, -1, 2:3]).unsqueeze(1)
            historical_embedding = vel_token
        else:
            vel_token = []
            for i in range(bs):
                vel_token.append(self.vel_to_embedding(trj_data[i, seq_lens[i] - 1: seq_lens[i], 2:3]))
            vel_token = torch.cat(vel_token).unsqueeze(1)
            trj_data = trj_data.transpose(0, 1)

            if not self.args.add_behavior and not self.args.add_intention:
                historical_embedding = vel_token
            else:
                historical_data = vel_token
                if self.args.add_behavior:
                    behavior_hidden = self.behavior_to_embedding.initHidden(bs)
                    behavior_embedding = self.behavior_to_embedding(trj_data[..., :2], seq_lens, behavior_hidden).transpose(0, 1)
                    historical_data = torch.cat((historical_data, behavior_embedding), dim=1)
                if self.args.add_intention:
                    intention_hidden = self.intention_to_embedding.initHidden(bs)
                    intention_embedding = self.intention_to_embedding(trj_data[..., :2], seq_lens, intention_hidden).transpose(0, 1)
                    historical_data = torch.cat((historical_data, intention_embedding), dim=1)

                historical_data += self.historical_pos_embedding

                historical_embedding = self.historical_transformer(historical_data, mask)
                historical_embedding = historical_embedding[:, 0:1]

        data = torch.cat((historical_embedding, data), dim=1)
        data += self.pos_embedding

        x = self.transformer(data, mask)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x


class XOE(nn.Module):
    def __init__(self, args, num_labels=6, input_dim=5):
        super().__init__()
        self.embedding_size = args.embedding_size
        self.obs_feat_size = args.obs_feat_size
        self.grid_num = args.grid_num

        self.input_dim = input_dim

        self.in_patch = In_Patch_Aggregator(args, output_dim=self.embedding_size)
        self.cross_patch = Cross_Patch_Aggregator(args, grid_size=self.grid_num ** 2, patch_dim=self.embedding_size, dim=self.obs_feat_size, depth=2, heads=8, mlp_dim=16)

    def forward(self, obs_trj, obs_data, data_sizes, seq_lens=None):
        bs = len(obs_trj)
        in_patch_aggregated_data = []
        for i in range(bs):
            agg = self.in_patch(obs_data[i].cuda(), sizes=data_sizes[i].tolist())
            in_patch_aggregated_data.append(agg.reshape(self.grid_num, self.grid_num, self.embedding_size))
        in_patch_aggregated_data = torch.stack(in_patch_aggregated_data)  # bs * N * N * emb_dim
        cross_embedding = self.cross_patch(in_patch_aggregated_data, obs_trj, seq_lens)

        return cross_embedding


class XOE_Trainer(nn.Module):
    def __init__(self, args, num_labels=6, input_dim=5, decoder_layer=3):
        super().__init__()
        self.xoe = XOE(args, num_labels, input_dim)
        self.mask = args.mask_size
        self.choice_size = args.pred_len - self.mask
        self.obs_feat_size = args.obs_feat_size
        self.grid_num = args.grid_num
        self.num_labels = num_labels
        self.recon = args.reconstruct
        self.pred_len = args.pred_len

        if self.choice_size > 0:
            self.masked_gt_encoder = \
                Mask_Trajectory_Encoder(mask_size=self.choice_size, patch_dim=3, dim=self.obs_feat_size, depth=2, heads=4, mlp_dim=8)  # input_3: time-step, x, y ==> feature
            self.decoder = DecoderLSTM(self.obs_feat_size * 2, decoder_layer)
            self.fc = nn.Linear(self.obs_feat_size * 2, 2)

        if self.recon:
            self.scene_recon = nn.Sequential(
                nn.Linear(self.obs_feat_size, self.grid_num ** 2 * num_labels),
                nn.ReLU(),
                nn.Linear(self.grid_num ** 2 * num_labels, self.grid_num ** 2 * num_labels),
            )
            self.ped_recon = nn.Sequential(
                nn.Linear(self.obs_feat_size, self.grid_num ** 2 * 2),
                nn.ReLU(),
                nn.Linear(self.grid_num ** 2 * 2, self.grid_num ** 2 * 2),
            )

    def forward(self, obs_trj, obs_data, data_sizes, seq_lens=None, gt_data=None):
        bs = len(obs_trj)
        cross_embedding = self.xoe(obs_trj, obs_data, data_sizes, seq_lens)

        if self.recon:
            recon_input = cross_embedding
            scene_logits = self.scene_recon(recon_input).reshape((bs, self.grid_num ** 2, self.num_labels))
            ped_logits = self.ped_recon(recon_input).reshape((bs, self.grid_num ** 2, 2))
        else:
            scene_logits = None
            ped_logits = None

        if self.choice_size > 0:
            output = []
            assert gt_data is not None
            idx = np.sort(np.random.choice(range(self.pred_len), self.choice_size, replace=False))
            idx = torch.from_numpy(idx).cuda()
            gt_data = gt_data[:, idx, :]
            idx = idx.unsqueeze(0).unsqueeze(-1)
            gt_data = torch.cat((torch.repeat_interleave(idx, bs, dim=0).float(), gt_data), dim=2)
            masked_gt_encoding = self.masked_gt_encoder(gt_data)
            outs = torch.cat([cross_embedding, masked_gt_encoding], dim=-1).unsqueeze(0)

            decoder_hidden = self.decoder.initHidden(bs)
            for i in range(self.pred_len):
                outs, decoder_hidden = self.decoder(outs, decoder_hidden)
                output.append(outs)

            output = torch.cat(output, 0)
            output = self.fc(output)
            output = output.transpose(0, 1)

        else:
            output = None

        return output, scene_logits, ped_logits

    def loadencoder(self, save_dir):
        checkpoint = torch.load(os.path.join(save_dir, "behaviorextractor.pth"))
        self.aoe.cross_patch.behavior_to_embedding.load_state_dict(checkpoint)
        checkpoint = torch.load(os.path.join(save_dir, "intentionextractor.pth"))
        self.aoe.cross_patch.intention_to_embedding.load_state_dict(checkpoint)


class BehaviorExtractorTrainer(nn.Module):
    def __init__(self, hidden_size=48, num_layer=3):
        super(BehaviorExtractorTrainer, self).__init__()
        self.hidden_size = hidden_size // 2
        self.num_layer = num_layer
        self.encoder_obs = ArbiEncoderLSTM(2, self.hidden_size, self.num_layer)
        self.encoder_pred = EncoderLSTM(2, self.hidden_size, self.num_layer)
        self.t = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self, obs_trj, pred_trj, seq_lens):
        bs = len(obs_trj)
        pred_trj = pred_trj.transpose(0, 1)
        obs_trj = obs_trj.transpose(0, 1)

        obs_hidden = self.encoder_obs.initHidden(bs)
        obs_encoded = self.encoder_obs(obs_trj, seq_lens, obs_hidden).squeeze(0)

        pred_hidden = self.encoder_pred.initHidden(bs)
        pred_encoded, _ = self.encoder_pred(pred_trj, pred_hidden)
        pred_encoded = pred_encoded[-1]

        obs_encoded_e = torch.nn.functional.normalize(obs_encoded, p=2, dim=1)
        pred_encoded_e = torch.nn.functional.normalize(pred_encoded, p=2, dim=1)

        logits = torch.mm(obs_encoded_e, pred_encoded_e.T) * torch.exp(self.t)

        return logits


class IntentionExtractorTrainer(nn.Module):
    def __init__(self, pred_len=12, hidden_size=48, num_layer=3):
        super(IntentionExtractorTrainer, self).__init__()
        self.pred_len = pred_len
        self.hidden_size = hidden_size // 2
        self.num_layer = num_layer
        self.encoder = ArbiEncoderLSTM(2, self.hidden_size, self.num_layer)
        self.generator = nn.Linear(2 * self.hidden_size, 2)

    def forward(self, trajectory_data, seq_lens):
        bs, _, _ = trajectory_data.shape
        data = trajectory_data.transpose(0, 1)

        encoder_hidden = self.encoder.initHidden(bs)
        final_encoded = self.encoder(data, seq_lens, encoder_hidden)

        intention = self.generator(final_encoded)

        return intention
