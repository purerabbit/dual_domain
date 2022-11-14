import itertools
import torch
import torch.nn as nn
from typing import List, Dict, Sequence, Tuple

from torch import Tensor

DEFAULT_ACT = nn.LeakyReLU


# ========================================================================================
# Basic components
# ========================================================================================
class MultiConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=1, stride=1, *, n_conv=1,
                 mid_channels=None, p_dropout=0.5, NormLayer=None, ActLayer=DEFAULT_ACT):   #p_dropout=0.0 former
        super().__init__()
        mid_channels = mid_channels or out_channels

        def _make_conv(in_channels, out_channels):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)]
            if p_dropout >= 0.0:
                layers.append(nn.Dropout2d(p_dropout))
            if NormLayer is not None:
                layers.append(NormLayer(out_channels))
            layers.append(ActLayer())
            return layers

        assert n_conv >= 1, f"At least has 1 convolutional layers, detecting `n_conv`={n_conv}"
        conv_channels = [in_channels] + [mid_channels for _ in range(n_conv - 1)] + [out_channels]
        conv_block_list = [_make_conv(conv_channels[i], conv_channels[i + 1]) for i in range(len(conv_channels) - 1)]
        layers = list(itertools.chain(*conv_block_list))
        self.multi_conv = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: [B, C_in, H, W]
        :return: [B, C_out, H, W]
        """
        y = self.multi_conv(x)
        return y



# ========================================================================================
# Abstract blocks for building encoder or decoder
# ========================================================================================
class ConcateUpBlock(nn.Module):
    """
res──────┐
         ▼
      ┌──────┐C_cat┌──┐C_cat┌─────┐
C_in─►│Concat├────►│Up├────►│Block├─►C_out
      └──────┘     └──┘     └─────┘
    """

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.block = block

    def forward(self, x: torch.Tensor, *res: Sequence[torch.Tensor]):
        """
        :param x: [B, C_x, ...]
        :param res: [B, C_res, ...]
        :return: [B, C_out, ...]
        """
        x_cat = torch.cat([x, *res], dim=1)  # [B, C_cat=C_in+C_res, ...]
        x_up = self.up(x_cat)
        y = self.block(x_up)  # [B, C_out, ...]
        return y


class UpConcateBlock(nn.Module):
    """
C_res_list────────┐
                  ▼
      ┌──┐C_up ┌──────┐C_cat┌─────┐
C_in─►│Up├────►│Concat├────►│Block├─►C_out
      └──┘     └──────┘     └─────┘
    """

    def __init__(self, upsampler: nn.Module, block: nn.Module) -> None:
        super().__init__()
        self.up = upsampler
        self.block = block

    def forward(self, x: torch.Tensor, *res: Sequence[torch.Tensor]):
        """
        :param x: [B, C_in, ...]
        :param res: [[B, C_up_1, ...], [B, C_up_2, ...], ...]
        :return: [B, C_out, ...]
        """
        x_up = self.up(x)
        x_cat = torch.cat([x_up, *res], dim=1)  # [B, C_cat=C_in+C_res, ...]
        y = self.block(x_cat)  # [B, C_out, ...]
        return y


class DownBlock(nn.Module):
    """
         ┌────┐   ┌─────┐
C_in────►│Down├──►│Block├──►C_out
         └────┘   └─────┘
    """

    def __init__(self, downsampler: nn.Module, block: nn.Module) -> None:
        super().__init__()
        self.down = downsampler
        self.block = block

    def forward(self, x):
        """
        :param x: [B, C_in, ...]
        :return: x[B, C_out, ...]
        """
        x_down = self.down(x)
        y = self.block(x_down)
        return y


# ==================== UNet Architecture ==================================
#                   EXAMPLE CLASSES
#
#            ┌─────────────res1───────────────┐
#            │                                │
#            │     ┌───────res2─────────┐     │
#            │     │                    │     │
#            │     │     ┌─res3───┐     │     │
#            │     │     │        ▼     ▼     ▼
# x_0─────IN─┴─►D1─┴─►D2─┴─►D3───►U1───►U2───►U3───►OUT───►y
#           x_1   x_2   x_3   x_4   z_1   z_2   z_3    z_4
#           C     2C    4C    8C    4C    2C    C
#                             (z_0)
# IN: in_conv, OUT: out_conv
# Dx: Down-stage_x, Ux: Up-stage_x
# =============================================================================
class UNetEncoder2d(nn.Module):
    """
    Convention: any stage-variables-list outside the module are ordered in ascending number of channels
    """

    def __init__(self, in_channels=2, base_channels=32, num_downs=3, downsample_size=(2, 2), *,  # *表示后面的内容一定要加上名称输入
                 is_batch_norm=True, n_conv_per_blk=2):
        super().__init__()

        channels_stages_out = [base_channels * 2 ** i for i in range(num_downs + 1)]  # eg. (32, 64, 128, 256)

        def _build_MultiConv(in_channels_stage, out_channels_stage):
            block = MultiConv2d(in_channels_stage, out_channels_stage,
                                kernel_size=(3, 3),
                                n_conv=n_conv_per_blk,
                                NormLayer=nn.BatchNorm2d if is_batch_norm else None,
                                ActLayer=nn.LeakyReLU)
            return block

        in_conv = _build_MultiConv(in_channels, channels_stages_out[0])
        downs = [DownBlock(block=_build_MultiConv(channels_stages_out[i], channels_stages_out[i + 1]),
                           downsampler=nn.MaxPool2d(kernel_size=downsample_size))
                 for i in range(num_downs)]
        layers = [in_conv] + downs

        self.layers = nn.ModuleList(layers)

    def forward(self, x_0: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        x: (B, C, H, W)
        """
        x = x_0
        xs = []
        for layer in self.layers:
            x = layer(x)
            xs.append(x)

        return xs



class UNetDecoder2d(nn.Module):
    """
    Convention: any stage-variables-list outside the module are ordered in ascending number of channels
    """

    def __init__(self, out_channels: int = 2, base_channels=32, num_ups=3, upsample_size=(2, 2), *,
                 is_batch_norm=True, n_conv_per_blk=2):
        super().__init__()

        channels_stages_in = [base_channels * 2 ** i for i in range(num_ups + 1)]  # eg. (32, 64, 128, 256)
        channels_stages_in = list(reversed(channels_stages_in))  # eg. (256, 128, 64, 32)

        def _build_MultiConv(in_channels_stage, out_channels_stage):
            block = MultiConv2d(in_channels_stage, out_channels_stage,
                                kernel_size=(3, 3),
                                n_conv=n_conv_per_blk,
                                NormLayer=nn.BatchNorm2d if is_batch_norm else None,
                                ActLayer=nn.LeakyReLU)
            return block

        layers = []
        for i in range(num_ups):
            stage_upsampled_channels = channels_stages_in[i]
            stage_concate_channels = stage_upsampled_channels + channels_stages_in[i + 1]
            stage_out_channels = channels_stages_in[i] // 2
            up = UpConcateBlock(upsampler=nn.Upsample(scale_factor=upsample_size, mode='nearest'),
                                block=_build_MultiConv(stage_concate_channels, stage_out_channels))
            layers.append(up)

        self.layers = nn.ModuleList(layers)  # from UNet bottom to top
        self.out_conv = _build_MultiConv(base_channels, out_channels)

    def forward(self, xs: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        :param xs: [[B, C_1, H_1, W_1], ..., [B, C_n, H_n, W_n]]
            NOTE: item 1...{n-1} are skip-connections, item n is the final feature maps from encoder
        :return: [B, C_out, H, W]
        """
        assert len(xs) == len(self.layers) + 1, \
            f"Number of residual is not matched with UNet Decoder"
        z = xs[-1]  # z_0
        res_list_reversed = list(reversed(xs[:-1]))  # [x_{n-1}, x_{n-2}, ..., x_1]

        for layer, res in zip(self.layers, res_list_reversed):
            z = layer(z, res)

        y = self.out_conv(z)
        return y


class UNet(nn.Module):
    """
    Abstract UNet framework, assume:
        encoder:
            input: x
            outputs: x_1, x_2, ..., x_n
        decoder:
            input: x_n, ..., x_2, x_1
            outputs: z_n = y
    return z_n = y
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, x):
        x_list = self.encoder(x)
        y = self.decoder(x_list)
        return y

# ========================================================================================
