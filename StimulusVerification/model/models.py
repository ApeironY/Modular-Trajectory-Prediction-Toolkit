import torch
import torch.nn as nn
from model.models_aux import GMM2D, SubgraphNet


class Social_Stimulus_Verifier(nn.Module):
    def __init__(self, feat_dim, obs_len, pred_len, num_mixtures, trj_scale=1):
        super(Social_Stimulus_Verifier, self).__init__()
        self.feat_size = feat_dim
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_mixtures = num_mixtures
        self.scale = trj_scale

        self.social_encoder = SubgraphNet(4, feat_size=feat_dim)
        self.seq_encoder = nn.LSTM(feat_dim, feat_dim, num_layers=2, batch_first=True)
        self.distribution_param_generator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * pred_len // 2),
            nn.ReLU(),
            nn.Linear(feat_dim * pred_len // 2, pred_len * num_mixtures * 6)
        )

    def initHidden(self, batch_size):
        return (torch.zeros(2, batch_size, self.feat_size).cuda(),
                torch.zeros(2, batch_size, self.feat_size).cuda())

    def forward(self, concatenated_social_data, split_sizes, normalized_trj_data, test_mode=False):
        concatenated_social_data = concatenated_social_data.float().cuda()
        normalized_trj_data = normalized_trj_data.float().cuda()

        num_trj = len(normalized_trj_data)
        encoded_data = self.social_encoder(concatenated_social_data, split_sizes)  # Test time operations

        if not test_mode:
            encoded_data = encoded_data.reshape(num_trj, self.obs_len, encoded_data.shape[-1])
        else:
            encoded_data = encoded_data[None].repeat(num_trj, 1, 1)

        hidden = self.initHidden(num_trj)
        output, hidden = self.seq_encoder(encoded_data, hidden)
        encoded_data = output[:, -1]
        dist_params = self.distribution_param_generator(encoded_data)

        dist_params = dist_params.reshape(num_trj, self.pred_len, self.num_mixtures * 6)
        log_pis, mus, log_sigmas, corr = torch.split(
            dist_params, [self.num_mixtures, self.num_mixtures * 2, self.num_mixtures * 2, self.num_mixtures], dim=-1
        )
        corr = torch.tanh(corr)
        gmm = GMM2D(log_pis, mus, log_sigmas, corr)
        log_prob = gmm.log_prob(normalized_trj_data * self.scale)

        return log_prob.mean(-1)


class Context_Stimulus_Verifier(nn.Module):
    def __init__(self, input_channel, feat_dim, num_mixtures, scale=1):
        super(Context_Stimulus_Verifier, self).__init__()
        self.input_size = 2
        self.input_channel = input_channel
        self.cond_feat_dim = feat_dim
        self.num_mixtures = num_mixtures
        self.velocity_scale = scale

        self.context_encoder = nn.Sequential(  # A CNN encoder tailored for 50 * 50 image inputs.
            nn.Conv2d(self.input_channel, 64, kernel_size=3, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, feat_dim, kernel_size=5, padding=0),
        )
        self.distribution_param_generator = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim * 2),
            nn.ReLU(),
            nn.Linear(feat_dim * 2, self.num_mixtures * 6, bias=False)
        )
        self.mul = self.num_mixtures * self.input_size

    def forward(self, velocity_data, map_data):
        velocity_data = velocity_data.float().cuda()
        map_data = map_data.float().cuda()
        bs = len(map_data)
        map_encoding = self.context_encoder(map_data).squeeze()
        if bs == 1:
            map_encoding = map_encoding.unsqueeze(0)
        dist_params = self.distribution_param_generator(map_encoding)
        log_pis, mus, log_sigmas, corr = torch.split(dist_params, [self.num_mixtures, self.mul, self.mul, self.num_mixtures], dim=-1)
        corr = torch.tanh(corr)
        gmm = GMM2D(log_pis, mus, log_sigmas, corr)
        log_prob = gmm.log_prob(velocity_data * self.velocity_scale)
        return log_prob

