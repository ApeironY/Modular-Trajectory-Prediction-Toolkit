import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape((b, n, self.heads, self.dim_head)).permute(0, 2, 1, 3), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            mask = nn.functional.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 2, 1, 3).reshape((b, n, self.heads * self.dim_head))
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Mask_Trajectory_Encoder(nn.Module):
    def __init__(self, mask_size, patch_dim, dim, depth, heads, mlp_dim, pool='cls', dim_head=64):
        super().__init__()
        self.num_patches = mask_size
        self.patch_dim = patch_dim
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim).cuda(), requires_grad=True)
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.token = nn.Parameter(torch.randn(1, 1, dim).cuda(), requires_grad=True)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, data, mask=None):  # data shape: bs * grid_size * patch_dim
        bs = data.shape[0]
        data = data.reshape(bs, self.num_patches, self.patch_dim)
        data = self.patch_to_embedding(data)
        data = torch.cat((self.token.repeat((bs, 1, 1)), data), dim=1)
        data += self.pos_embedding
        x = self.transformer(data, mask)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer=3):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layer, bidirectional=True)

    def forward(self, input_data, hidden):
        output, hidden = self.lstm(input_data, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return (torch.zeros(self.num_layer * 2, batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layer * 2, batch_size, self.hidden_size).cuda())


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, num_layer=1):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layer)

    def forward(self, input_data, hidden):
        output, hidden = self.lstm(input_data, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return (torch.zeros(self.num_layer, batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layer, batch_size, self.hidden_size).cuda())


class ArbiEncoderLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=48, num_layer=3):
        super(ArbiEncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.input_size = input_size
        self.encoder = nn.LSTM(self.input_size, self.hidden_size, self.num_layer, bidirectional=True)

    def forward(self, obs_data, seq_lens, hidden):
        obs_data_padded = pack_padded_sequence(obs_data, seq_lens, batch_first=False, enforce_sorted=False)
        trj_encoded, _ = self.encoder(obs_data_padded, hidden)
        trj_padded, _ = pad_packed_sequence(trj_encoded, batch_first=False)
        final_encoded = []
        trj_padded = trj_padded.transpose(0, 1)
        for i in range(len(seq_lens)):
            final_encoded.append(trj_padded[i, seq_lens[i] - 1: seq_lens[i]])
        final_encoded = torch.cat(final_encoded, dim=0).unsqueeze(0).cuda()

        return final_encoded

    def initHidden(self, batch_size):
        return (torch.zeros(self.num_layer * 2, batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layer * 2, batch_size, self.hidden_size).cuda())
