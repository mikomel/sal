import math

import torch

from torch import nn


class StructureAwareLayer(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 1,
        num_rows: int = 6,
        num_cols: int = 420,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.weights = torch.nn.Parameter(
            torch.randn(num_rows, num_cols, out_channels, kernel_size)
        )
        self.biases = torch.nn.Parameter(torch.randn(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.biases, -bound, bound)

    def forward(
        self, x: torch.Tensor, num_rows: int = 3, num_cols: int = 3
    ) -> torch.Tensor:
        """
        :param x: a tensor of shape (batch_size, in_channels, dim)
        :return: a tensor of shape (batch_size, out_channels, dim / kernel_size)
        """
        w = self.weights.unfold(
            0, num_rows, num_rows
        )  # (self.num_rows / num_rows, num_cols, out_dim, in_dim, num_rows)
        w = w.unfold(
            1, num_cols, num_cols
        )  # (self.num_rows / num_rows, self.num_cols / num_cols, out_dim, in_dim, num_rows, num_cols)
        w = w.mean((0, 1))  # (out_dim, in_dim, num_rows, num_cols)
        w = w.flatten(start_dim=1)  # (out_dim, in_dim * num_rows * num_cols)
        w = w.transpose(0, 1)  # (in_dim * num_rows * num_cols, out_dim)

        batch_size, in_channels, in_dim = x.shape
        num_groups = in_dim // self.kernel_size
        x = x.view(
            batch_size, in_channels, num_groups, self.kernel_size
        )  # n in_c ng gs
        x = x.transpose(1, 2).contiguous()  # b ng in_c gs
        x = x.flatten(0, 1)  # (b ng) in_c gs
        x = x.flatten(1, 2)  # (b ng) (in_c gs)

        x = torch.einsum("bd,dc->bc", x, w) + self.biases  # (b ng) out_c
        x = x.view(batch_size, num_groups, self.out_channels)  # b ng out_c
        return x.transpose(1, 2)  # b out_c ng
