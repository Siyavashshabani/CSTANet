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



class Encoder1(nn.Module):
    def __init__(
        self,
        feature_size: int = 48,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        # Initialize decoder blocks
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=2*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2*feature_size,
            out_channels=4*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4*feature_size,
            out_channels=8*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8*feature_size,
            out_channels=16*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16*feature_size,
            out_channels=16*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

    def forward(self, x):
        # Apply the encoder blocks
        x[0] = self.encoder0(x[0])
        x[1] = self.encoder1(x[1])
        x[2] = self.encoder2(x[2])
        x[3] = self.encoder3(x[3])
        x[4] = self.encoder4(x[4])
        return x


class Encoder2(nn.Module):
    def __init__(
        self,
        feature_size: int = 48,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        # Initialize decoder blocks
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=2*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2*feature_size,
            out_channels=4*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4*feature_size,
            out_channels=8*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8*feature_size,
            out_channels=16*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )


    def forward(self, x):
        # Apply the encoder blocks
        x[0] = self.encoder0(x[0])
        x[1] = self.encoder1(x[1])
        x[2] = self.encoder2(x[2])
        x[3] = self.encoder3(x[3])
        return x


class Encoder3(nn.Module):
    def __init__(
        self,
        feature_size: int = 48,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        # Initialize decoder blocks
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=2*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2*feature_size,
            out_channels=4*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4*feature_size,
            out_channels=8*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )


    def forward(self, x):
        # Apply the encoder blocks
        x[0] = self.encoder0(x[0])
        x[1] = self.encoder1(x[1])
        x[2] = self.encoder2(x[2])
        return x

class Encoder4(nn.Module):
    def __init__(
        self,
        feature_size: int = 48,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        # Initialize decoder blocks
        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=2*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2*feature_size,
            out_channels=4*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

    def forward(self, x):
        # Apply the encoder blocks
        x[0] = self.encoder0(x[0])
        x[1] = self.encoder1(x[1])
        return x
