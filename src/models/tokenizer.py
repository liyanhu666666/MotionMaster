import torch
import torch.nn as nn
import torch.nn.functional as F

from quantizers import FSQ
from models.resnet import Resnet1D


class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()

        blocks = []
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())

        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.permute(0, 2, 1).float()
        x = self.model(x)
        x = x.permute(0, 2, 1)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width=3,
                 output_emb_width=512,
                 down_t=2,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []

        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.model(x)
        return x.permute(0, 2, 1)

class FSQAE(nn.Module):
    def __init__(self, input_dim: int = 85, hidden_dim: int = 256, latent_dim: int = 5):
        super().__init__()

        self.encoder = Encoder(input_emb_width=input_dim,
                               output_emb_width=latent_dim,
                               down_t=2,
                               stride_t=2,
                               width=hidden_dim,
                               depth=3,
                               dilation_growth_rate=3,
                               activation='relu',
                               norm=None)
        
        self.decoder = Decoder(input_emb_width=input_dim,
                               output_emb_width=latent_dim,
                               down_t=2,
                               stride_t=2,
                               width=hidden_dim,
                               depth=3,
                               dilation_growth_rate=3,
                               activation='relu',
                               norm=None)

        levels = [8, 8, 8, 6, 5]

        self.quantize = FSQ(levels=levels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, return_id=False) -> torch.Tensor:
        z = self.encode(x)
        z_q, id_t = self.quantize(z)
        x_recon = self.decode(z_q)

        if return_id:
            return x_recon, id_t
        else:
            return x_recon

