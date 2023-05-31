# Auxiliary Components of the Stimulus Verifier. Codes in this file are borrowed from
# various open-source projects such as:
# https://github.com/StanfordASL/Trajectron-plus-plus
# and
# https://github.com/Liang-ZX/VectorNet

import torch
import torch.distributions as td
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]


class GMM2D(td.Distribution):
    r"""
    Gaussian Mixture Model using 2D Multivariate Gaussians each of as N components:
    Cholesky decompesition and affine transformation for sampling:

    .. math:: Z \sim N(0, I)

    .. math:: S = \mu + LZ

    .. math:: S \sim N(\mu, \Sigma) \rightarrow N(\mu, LL^T)

    where :math:`L = chol(\Sigma)` and

    .. math:: \Sigma = \left[ {\begin{array}{cc} \sigma^2_x & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma^2_y \\ \end{array} } \right]

    such that

    .. math:: L = chol(\Sigma) = \left[ {\begin{array}{cc} \sigma_x & 0 \\ \rho \sigma_y & \sigma_y \sqrt{1-\rho^2} \\ \end{array} } \right]

    :param log_pis: Log Mixing Proportions :math:`log(\pi)`. [..., N]
    :param mus: Mixture Components mean :math:`\mu`. [..., N * 2]
    :param log_sigmas: Log Standard Deviations :math:`log(\sigma_d)`. [..., N * 2]
    :param corrs: Cholesky factor of correlation :math:`\rho`. [..., N]
    :param clip_lo: Clips the lower end of the standard deviation.
    :param clip_hi: Clips the upper end of the standard deviation.
    """
    def __init__(self, log_pis, mus, log_sigmas, corrs):
        super(GMM2D, self).__init__(batch_shape=log_pis.shape[0], event_shape=log_pis.shape[1:])
        self.components = log_pis.shape[-1]
        self.dimensions = 2
        self.device = log_pis.device

        log_pis = torch.clamp(log_pis, min=-1e5)
        self.log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)  # [..., N]
        self.mus = self.reshape_to_components(mus)         # [..., N, 2]
        self.log_sigmas = self.reshape_to_components(log_sigmas)  # [..., N, 2]

        # 2022/08/19 ADDED CLAMP to sigma
        # self.log_sigmas = torch.clamp(self.log_sigmas, min=-40, max=42)
        self.log_sigmas = torch.clamp(self.log_sigmas, min=-30, max=30)

        self.sigmas = torch.exp(self.log_sigmas)                       # [..., N, 2]
        self.one_minus_rho2 = 1 - corrs**2                        # [..., N]
        self.one_minus_rho2 = torch.clamp(self.one_minus_rho2, min=1e-5, max=1)  # otherwise log can be nan
        self.corrs = corrs  # [..., N]

        self.L = torch.stack([torch.stack([self.sigmas[..., 0], torch.zeros_like(self.log_pis)], dim=-1),
                              torch.stack([self.sigmas[..., 1] * self.corrs,
                                           self.sigmas[..., 1] * torch.sqrt(self.one_minus_rho2)],
                                          dim=-1)],
                             dim=-2)

        self.pis_cat_dist = td.Categorical(logits=log_pis)

    @classmethod
    def from_log_pis_mus_cov_mats(cls, log_pis, mus, cov_mats):
        corrs_sigma12 = cov_mats[..., 0, 1]
        sigma_1 = torch.clamp(cov_mats[..., 0, 0], min=1e-8)
        sigma_2 = torch.clamp(cov_mats[..., 1, 1], min=1e-8)
        sigmas = torch.stack([torch.sqrt(sigma_1), torch.sqrt(sigma_2)], dim=-1)
        log_sigmas = torch.log(sigmas)
        corrs = corrs_sigma12 / (torch.prod(sigmas, dim=-1))
        return cls(log_pis, mus, log_sigmas, corrs)

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched.

        :param sample_shape: Shape of the samples
        :return: Samples from the GMM.
        """
        mvn_samples = (self.mus + torch.squeeze(torch.matmul(self.L, torch.unsqueeze(
                        torch.randn(size=sample_shape + self.mus.shape, device=self.device), dim=-1)
                      ), dim=-1))
        component_cat_samples = self.pis_cat_dist.sample(sample_shape)
        selector = torch.unsqueeze(to_one_hot(component_cat_samples, self.components), dim=-1)
        return torch.sum(mvn_samples*selector, dim=-2)

    def log_prob(self, value):
        r"""
        Calculates the log probability of a value using the PDF for bivariate normal distributions:

        .. math::
            f(x | \mu, \sigma, \rho)={\frac {1}{2\pi \sigma _{x}\sigma _{y}{\sqrt {1-\rho ^{2}}}}}\exp
            \left(-{\frac {1}{2(1-\rho ^{2})}}\left[{\frac {(x-\mu _{x})^{2}}{\sigma _{x}^{2}}}+
            {\frac {(y-\mu _{y})^{2}}{\sigma _{y}^{2}}}-{\frac {2\rho (x-\mu _{x})(y-\mu _{y})}
            {\sigma _{x}\sigma _{y}}}\right]\right)

        :param value: The log probability density function is evaluated at those values.
        :return: Log probability
        """
        # x: [..., 2]
        value = torch.unsqueeze(value, dim=-2)       # [..., 1, 2]
        dx = value - self.mus                       # [..., N, 2]

        p1 = torch.sum((dx/self.sigmas)**2, dim=-1)
        p2 = 2*self.corrs*torch.prod(dx, dim=-1)/torch.prod(self.sigmas, dim=-1)
        exp_nominator = (p1 - p2)    # [..., N]

        component_log_p = -(2*np.log(2*np.pi)
                            + torch.log(self.one_minus_rho2)
                            + 2*torch.sum(self.log_sigmas, dim=-1)
                            + exp_nominator/self.one_minus_rho2) / 2
        res = torch.logsumexp(self.log_pis + component_log_p, dim=-1)
        return res

    def get_for_node_at_time(self, n, t):
        return self.__class__(self.log_pis[:, n:n+1, t:t+1], self.mus[:, n:n+1, t:t+1],
                              self.log_sigmas[:, n:n+1, t:t+1], self.corrs[:, n:n+1, t:t+1])

    def mode(self):
        """
        Calculates the mode of the GMM by calculating probabilities of a 2D mesh grid

        :param required_accuracy: Accuracy of the meshgrid
        :return: Mode of the GMM
        """
        if self.mus.shape[-2] > 1:
            samp, bs, time, comp, _ = self.mus.shape
            assert samp == 1, "For taking the mode only one sample makes sense."
            mode_node_list = []
            for n in range(bs):
                mode_t_list = []
                for t in range(time):
                    nt_gmm = self.get_for_node_at_time(n, t)
                    x_min = self.mus[:, n, t, :, 0].min()
                    x_max = self.mus[:, n, t, :, 0].max()
                    y_min = self.mus[:, n, t, :, 1].min()
                    y_max = self.mus[:, n, t, :, 1].max()
                    search_grid = torch.stack(torch.meshgrid([torch.arange(x_min, x_max, 0.01),
                                                              torch.arange(y_min, y_max, 0.01)]), dim=2
                                              ).view(-1, 2).float().to(self.device)

                    ll_score = nt_gmm.log_prob(search_grid)
                    argmax = torch.argmax(ll_score.squeeze(), dim=0)
                    mode_t_list.append(search_grid[argmax])
                mode_node_list.append(torch.stack(mode_t_list, dim=0))
            return torch.stack(mode_node_list, dim=0).unsqueeze(dim=0)
        return torch.squeeze(self.mus, dim=-2)

    def reshape_to_components(self, tensor):
        if len(tensor.shape) == 5:
            return tensor
        return torch.reshape(tensor, list(tensor.shape[:-1]) + [self.components, self.dimensions])

    def get_covariance_matrix(self):
        cov = self.corrs * torch.prod(self.sigmas, dim=-1)
        E = torch.stack([torch.stack([self.sigmas[..., 0]**2, cov], dim=-1),
                         torch.stack([cov, self.sigmas[..., 1]**2], dim=-1)],
                        dim=-2)
        return E

    @property
    def arg_constraints(self):
        return dict()


class SubgraphNet_Layer(nn.Module):
    def __init__(self, input_channels=32, hidden_channels=16):
        super().__init__()
        self.fc = nn.Linear(input_channels, hidden_channels)

    def forward(self, input_data):
        hidden = self.fc(input_data)
        encode_data = torch.relu(hidden)
        polyline_feature = torch.max(encode_data, dim=0)[0].unsqueeze(0)
        polyline_feature = polyline_feature.repeat(len(encode_data), 1)
        output = torch.cat([encode_data, polyline_feature], 1)
        return output


class SubgraphNet(nn.Module):
    def __init__(self, input_channels, feat_size):
        super().__init__()
        self.sublayer1 = SubgraphNet_Layer(input_channels, hidden_channels=feat_size // 2)
        self.sublayer2 = SubgraphNet_Layer(input_channels=feat_size, hidden_channels=feat_size // 2)
        self.sublayer3 = SubgraphNet_Layer(input_channels=feat_size, hidden_channels=feat_size // 2)

    def forward(self, input_data, split_sizes):  # input.shape -> torch.Size([30, 6]) torch.Size([18, 8]),
        size_cnt = 0
        polyline_feature = []
        for i in range(len(split_sizes)):
            out1 = self.sublayer1(input_data[size_cnt:size_cnt+split_sizes[i]])
            out2 = self.sublayer2(out1)
            out3 = self.sublayer3(out2)
            polyline_feature.append(torch.max(out3, dim=0)[0])
            size_cnt += split_sizes[i]
        polyline_feature = torch.stack(polyline_feature)

        return polyline_feature


class GraphAttentionNet(nn.Module):
    def __init__(self, in_dim=128, key_dim=64, value_dim=64):
        super().__init__()
        self.queryFC = nn.Linear(in_dim, key_dim)
        nn.init.kaiming_normal_(self.queryFC.weight)
        self.keyFC = nn.Linear(in_dim, key_dim)
        nn.init.kaiming_normal_(self.keyFC.weight)
        self.valueFC = nn.Linear(in_dim, value_dim)
        nn.init.kaiming_normal_(self.valueFC.weight)

    def forward(self, polyline_feature):
        p_query = F.relu(self.queryFC(polyline_feature))
        p_key = F.relu(self.keyFC(polyline_feature))
        p_value = F.relu(self.valueFC(polyline_feature))
        query_result = p_query.mm(p_key.t())
        query_result = query_result / (p_key.shape[1] ** 0.5)
        attention = F.softmax(query_result, dim=1)
        output = attention.mm(p_value)
        return torch.max(output, dim=0)[0]

