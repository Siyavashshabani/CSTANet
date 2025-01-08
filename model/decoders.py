from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.utils.deprecate_utils import deprecated_arg

import torch.nn as nn
from collections.abc import Sequence

class Decoder1(nn.Module):
    def __init__(
        self,
        feature_size: int = 48,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        # Initialize decoder blocks
        self.dec1 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=feature_size*16,
        out_channels=feature_size*16,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )

        self.dec2 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=feature_size*16,
        out_channels=feature_size*8,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )

        self.dec3 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=feature_size*8,
        out_channels=feature_size*4,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )
        self.dec4 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=feature_size*4,
        out_channels=feature_size*2,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )
    def forward(self, x):
        # Apply the decoder blocks
        ## dwon, up
        x1 = self.dec1(x[4], x[3])

        x2 = self.dec2(x1  ,x[2] )

        x3 = self.dec3(x2  ,x[1] )

        x4 = self.dec4(x3  ,x[0] )

        return x4



# Assuming UnetrUpBlock is defined elsewhere and correctly imported


class Decoder2(nn.Module):
    def __init__(
        self,
        feature_size: int = 48,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        # Initialize decoder blocks

        self.dec3 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=feature_size*16,
        out_channels=feature_size*8,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )

        self.dec2 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=feature_size*8,
        out_channels=feature_size*4,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )
        self.dec1 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=feature_size*4,
        out_channels=feature_size*2,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )
    def forward(self, x):
        # Apply the decoder blocks
        ## dwon, up
        x1 = self.dec3(x[3], x[2])

        x2 = self.dec2(x1  ,x[1])

        x3 = self.dec1(x2  ,x[0])

        return x3



class Decoder3(nn.Module):
    def __init__(
        self,
        feature_size: int = 48,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        # Initialize decoder blocks

        self.dec2 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=feature_size*8,
        out_channels=feature_size*4,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )
        self.dec1 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=feature_size*4,
        out_channels=feature_size*2,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )
    def forward(self, x):
        # Apply the decoder blocks
        ## dwon, up
        x1 = self.dec2(x[2], x[1])

        x2 = self.dec1(x1  ,x[0])

        return x2


class Decoder4(nn.Module):
    def __init__(
        self,
        feature_size: int = 48,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        # Initialize decoder blocks

        self.dec1 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=feature_size*4,
        out_channels=feature_size*2,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )
    def forward(self, x):
        x1 = self.dec1(x[1], x[0])

        return x1


class DecoderF(nn.Module):
    def __init__(
        self,
        feature_size: int = 48,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        # Initialize decoder blocks
        self.dec1 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=2*feature_size,
        out_channels=2*feature_size,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )

        self.dec2 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=2*feature_size,
        out_channels=2*feature_size,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )

        self.dec3 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=2*feature_size,
        out_channels=2*feature_size,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )
        self.dec4 = UnetrUpBlock(
        spatial_dims=spatial_dims,
        in_channels=2*feature_size,
        out_channels=2*feature_size,
        kernel_size=3,
        upsample_kernel_size=2,
        norm_name=norm_name,
        res_block=True,
    )
    def forward(self, x):
        # Apply the decoder blocks
        ## dwon, up
        x1 = self.dec1(x[4],x[3] )
        
        x2 = self.dec2(x1  ,x[2] )

        x3 = self.dec3(x2  ,x[1] )

        x4 = self.dec4(x3  ,x[0] )

        return x4
