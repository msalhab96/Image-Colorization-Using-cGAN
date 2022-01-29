#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import relu


class ConvBlock(nn.Module):
    """A ConvBlock consists of Conv layer followed by relu and batch norm

    Args:
        in_channels (int): the number of channels in the input tensor.
        out_channels (int): the number of channels out of the conv layer.
        kernel_size (int): the conv layer kernel size.
        stride (int): the conv layer stride size.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int
            ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.b_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return relu(self.b_norm(self.conv(x)))


class TransposeConvBlock(nn.Module):
    """A TransposeConvBlock consists of TransposConv layer followed by
    relu and batch norm

    Args:
        in_channels (int): the number of channels in the input tensor.
        out_channels (int): the number of channels out of the conv layer.
        kernel_size (int): the conv layer kernel size.
        stride (int): the conv layer stride size.
        padd_out (int): the number of padding to be introduced
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int
            ) -> None:
        super().__init__()
        self.tr_conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride
            )
        self.b_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return relu(self.b_norm(self.tr_conv(x)))


class Generator(nn.Module):
    """The Generator takes the input as a grayscale image
    and convert it into colorized image

    Args:
        in_channels (int): the number of channels in the input tensor.
        out_channels (int): the number of channels in the target/output image.
        scaling_factor (int): the channel scaling factor, for the
        ith layer will have n * scaling_factor * i channels
        base_n_channels (int, optional): the base number of channels.
        Defaults to 32.
        device (str, optional): the device to store the model on.
        Defaults to 'cuda'
        """
    def __init__(
                self,
                in_channels: int,
                out_channels: int,
                scaling_factor: int,
                base_n_channels=32,
                device='cuda'
                ) -> None:
        super().__init__()
        self.enc_layers = nn.ModuleList([ConvBlock(
            in_channels,
            base_n_channels * scaling_factor,
            3,
            1).to(device)
        ])
        self.enc_layers.extend([
            ConvBlock(
                base_n_channels * scaling_factor * i,
                base_n_channels * scaling_factor * (i + 1),
                kernel_size=4,
                stride=2
                ).to(device)
            for i in range(1, 5)
        ])
        self.dec_layers = nn.ModuleList([
            TransposeConvBlock(
                base_n_channels * scaling_factor * i,
                base_n_channels * scaling_factor * (i - 1),
                kernel_size=4,
                stride=2
            ).to(device)
            for i in range(5, 1, -1)
            ])
        self.last_deconv = nn.ConvTranspose2d(
            in_channels=base_n_channels * scaling_factor,
            out_channels=out_channels,
            kernel_size=3,
            stride=1
            ).to(device)

    def forward(self, x: Tensor) -> Tensor:
        out = x
        result = []
        for layer in self.enc_layers:
            out = layer(out)
            result.append(out)
        result = list(reversed(result))
        for i, layer in enumerate(self.dec_layers):
            if i >= 1 and i <= 4:
                out += result[i]
            out = layer(out)
        return torch.sigmoid(self.last_deconv(out))


class Discriminator(nn.Module):
    """The Discriminator classify whether the given image
    is a real or fake image

    Args:
        in_channels (int): the number of channels in the input tensor.
        scaling_factor (int): the channel scaling factor, for the
        ith layer will have n * scaling_factor * i channels
        base_n_channels (int, optional): the base number of channels.
        Defaults to 32.
    """

    def __init__(
            self,
            in_channels: int,
            base_n_channels: int,
            scaling_factor: int
            ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=base_n_channels * scaling_factor,
                kernel_size=4,
                stride=1
                )
        ])
        self.layers.extend([
            nn.Conv2d(
                in_channels=base_n_channels * i * scaling_factor,
                out_channels=base_n_channels * (i + 1) * scaling_factor,
                kernel_size=4,
                stride=2
            )
            for i in range(1, 4)
        ])
        self.fc = nn.Linear(in_features=16 * 25, out_features=1)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        out = out.reshape(-1, 16 * 25)
        out = self.fc(out)
        return torch.sigmoid(out)


class Loss:
    def __init__(self, device: str) -> None:
        self.device = device
        self.bce_criterion = nn.BCELoss().to(self.device)
        self.mae_criterion = nn.L1Loss().to(self.device)

    def calc_bce_target_1(self, result: Tensor, beta=0.9):
        """Used to calculate the BCE loss with 1 as a ground truth

        Args:
            result (Tensor): A tensor of probabilites
            beta (float, optional): smoothing variable. Defaults to 0.9.
        """
        bce_criterion = nn.BCELoss().to(self.device)
        target = beta * torch.ones(result.shape[0])
        target = target.to(self.device)
        return bce_criterion(result, target)

    def calc_bce_target_0(self, result: Tensor):
        """Used to calculate the BCE loss with 0 as a ground truth

        Args:
            result (Tensor): A tensor of probabilites
        """
        bce_criterion = nn.BCELoss().to(self.device)
        target = torch.zeros(result.shape[0])
        target = target.to(self.device)
        return bce_criterion(result, target)

    def calc_mae(self, result: Tensor, target: Tensor, lamb=100):
        """Used to calculate the the mean absolute error
        between result and target

        Args:
            result (Tensor): the result out of the model
            target (Tensor): the target/ground truth
            lamb (int, optional): weighting factor. Defaults to 100.
        """
        assert result.shape == target.shape, \
            'the result and target shape does not match!'
        mae_criterion = nn.L1Loss().to(self.device)
        target = target.to(self.device)
        return lamb * mae_criterion(result, target)
