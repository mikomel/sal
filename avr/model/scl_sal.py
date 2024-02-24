from typing import Optional

import torch
from einops.layers.torch import Rearrange
from torch import nn

from avr.model.neural_blocks import (
    ResidualPreNormFeedForward,
    ResidualFeedForward,
    FeedForwardResidualBlock,
    Scattering,
    ConvBnRelu,
)
from avr.model.structure_aware_layer import StructureAwareLayer


class SCLSAL(nn.Module):
    def __init__(
        self,
        num_hidden_channels: int = 32,
        embedding_size: int = 128,
        ff_dim: int = 80,
        image_size: int = 160,
        num_local_scattering_groups: int = 10,
        num_global_scattering_groups: int = 80,
        sal_num_rows: int = 6,
        sal_num_cols: int = 420,
        ffblock: str = "pre-norm-residual",
    ):
        super(SCLSAL, self).__init__()
        assert ff_dim % num_local_scattering_groups == 0
        assert ff_dim % num_global_scattering_groups == 0
        local_scattering_group_size = ff_dim // num_local_scattering_groups
        global_scattering_group_size = (
            num_local_scattering_groups * 8
        ) // num_global_scattering_groups
        c = num_hidden_channels
        d = embedding_size
        conv_dimension = (40 * (image_size // 80)) ** 2

        if ffblock == "pre-norm-residual":
            FeedForward = ResidualPreNormFeedForward
        elif ffblock == "residual-without-norm":
            FeedForward = ResidualFeedForward
        elif ffblock == "residual-with-norm":
            FeedForward = FeedForwardResidualBlock
        else:
            raise ValueError(f"Incorrect value for ffblock: {ffblock}")

        self.model_local = nn.Sequential(
            ConvBnRelu(1, c // 2, kernel_size=3, stride=2, padding=1),
            ConvBnRelu(c // 2, c // 2, kernel_size=3, padding=1),
            ConvBnRelu(c // 2, c, kernel_size=3, padding=1),
            ConvBnRelu(c, c, kernel_size=3, padding=1),
            # Panel projection
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(conv_dimension, ff_dim),
            nn.ReLU(inplace=True),
            FeedForward(ff_dim),
            # Attribute scattering
            Scattering(num_groups=num_local_scattering_groups),
            nn.Linear(c * local_scattering_group_size, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, 8),
            # Merge
            nn.Flatten(start_dim=-2, end_dim=-1),
            FeedForward(num_local_scattering_groups * 8),
        )

        self.sal = StructureAwareLayer(
            out_channels=64,
            kernel_size=global_scattering_group_size,
            num_rows=sal_num_rows,
            num_cols=sal_num_cols,
        )
        self.model_global = nn.Sequential(
            Rearrange("b d c -> (b c) d"),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 5),
            Rearrange("(b c) d -> b (c d)", c=num_global_scattering_groups),
            FeedForward(5 * num_global_scattering_groups),
            nn.Linear(5 * num_global_scattering_groups, d),
        )

    def forward(
        self,
        context: torch.Tensor,
        answers: torch.Tensor,
        num_rows: Optional[int] = -1,
        num_cols: Optional[int] = 3,
    ) -> torch.Tensor:
        x = torch.cat([context, answers], dim=1)
        num_context_panels = context.size(1)
        num_answer_panels = answers.size(1)
        batch_size, num_panels, num_channels, height, width = x.shape
        num_rows = (num_context_panels + 1) // 3 if num_rows == -1 else num_rows

        x = x.view((batch_size * num_panels), num_channels, height, width)
        x = self.model_local(x)
        x = x.view(batch_size, num_panels, -1)

        x = torch.cat(
            [
                x[:, :num_context_panels, :]
                .unsqueeze(dim=1)
                .repeat(1, num_answer_panels, 1, 1),
                x[:, num_context_panels:, :].unsqueeze(dim=2),
            ],
            dim=2,
        )
        x = x.view((batch_size * num_answer_panels), (num_context_panels + 1), -1)

        x = self.sal.forward(
            x, num_rows=num_rows, num_cols=num_cols
        )
        x = self.model_global(x)
        x = x.view(batch_size, num_answer_panels, -1)
        return x


class OddOneOutSCLSAL(SCLSAL):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_panels, num_channels, height, width = x.shape
        x = x.view((batch_size * num_panels), num_channels, height, width)
        x = self.model_local(x)
        x = x.view(batch_size, num_panels, -1)

        embedding_dim = x.shape[-1]
        mask = (
            ~torch.eye(num_panels, device=x.device, dtype=torch.bool)
            .unsqueeze(-1)
            .repeat(1, 1, embedding_dim)
        )
        x = torch.stack(
            [
                x.masked_select(m.repeat(batch_size, 1, 1)).view(
                    batch_size, num_panels - 1, embedding_dim
                )
                for m in mask
            ],
            dim=1,
        )  # b p p-1 d
        x = x.view((batch_size * num_panels), (num_panels - 1), -1)

        x = self.sal.forward(
            x, num_rows=1, num_cols=num_panels - 1
        )
        x = self.model_global(x)
        x = x.view(batch_size, num_panels, -1)
        return x
