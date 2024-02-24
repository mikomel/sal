from typing import Optional

import torch
from einops.layers.torch import Rearrange
from torch import nn

from avr.model.neural_blocks import (
    ConvBnRelu,
    ResidualPreNormFeedForward,
    ResidualFeedForward,
    FeedForwardResidualBlock,
    GroupObjectsIntoPairs,
    DeepLinearBNReLU,
    Sum,
)
from avr.model.structure_aware_layer import StructureAwareLayer
from avr.model.wild_relation_network import TagPanelEmbeddings


class SCAR(nn.Module):
    def __init__(
        self,
        num_hidden_channels: int = 32,
        embedding_size: int = 128,
        ff_dim: int = 80,
        image_size: int = 160,
        local_kernel_size: int = 10,
        global_kernel_size: int = 10,
        sal_num_rows: int = 6,
        sal_num_cols: int = 420,
        ffblock: str = "pre-norm-residual",
    ):
        super(SCAR, self).__init__()
        assert ff_dim % local_kernel_size == 0
        assert ff_dim % global_kernel_size == 0
        local_group_size = ff_dim // local_kernel_size
        global_group_size = (local_kernel_size * 8) // global_kernel_size
        c = num_hidden_channels
        conv_dimension = (40 * (image_size // 80)) ** 2

        if ffblock == "pre-norm-residual":
            FeedForward = ResidualPreNormFeedForward
        elif ffblock == "residual-without-norm":
            FeedForward = ResidualFeedForward
        elif ffblock == "residual-with-norm":
            FeedForward = FeedForwardResidualBlock
        else:
            raise ValueError(f"Incorrect value for ffblock: {ffblock}")

        self.global_kernel_size = global_kernel_size
        self.global_group_size = global_group_size

        self.model_local = nn.Sequential(
            ConvBnRelu(1, c // 2, kernel_size=3, stride=2, padding=1),
            ConvBnRelu(c // 2, c // 2, kernel_size=3, padding=1),
            ConvBnRelu(c // 2, c, kernel_size=3, padding=1),
            ConvBnRelu(c, c, kernel_size=3, padding=1),
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(conv_dimension, ff_dim),
            nn.ReLU(inplace=True),
            FeedForward(ff_dim, activation=nn.GELU),
            nn.Conv1d(
                c, 128, kernel_size=(local_group_size,), stride=(local_group_size,)
            ),
            nn.GELU(),
            nn.Conv1d(128, 8, kernel_size=(1,), stride=(1,)),
            nn.Flatten(start_dim=-2, end_dim=-1),
            FeedForward(local_kernel_size * 8, activation=nn.GELU),
        )

        self.sal = StructureAwareLayer(
            out_channels=64,
            kernel_size=global_group_size,
            num_rows=sal_num_rows,
            num_cols=sal_num_cols,
        )
        self.model_global = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=(1,)),
            nn.GELU(),
            nn.Conv1d(32, 5, kernel_size=(1,)),
            nn.Flatten(start_dim=-2, end_dim=-1),
            FeedForward(5 * global_kernel_size, activation=nn.GELU),
            nn.Linear(5 * global_kernel_size, embedding_size),
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

        x = self.sal.forward(x, num_rows=num_rows, num_cols=num_cols)
        x = self.model_global(x)
        x = x.view(batch_size, num_answer_panels, -1)
        return x


class RelationNetworkSAL(nn.Module):
    def __init__(
        self, in_dim: int, out_channels: int, out_dim: int, embedding_size: int = 128
    ):
        super().__init__()
        self.tag_panel_embeddings = TagPanelEmbeddings()
        self.relation_network = nn.Sequential(
            GroupObjectsIntoPairs(),
            DeepLinearBNReLU(
                2, 2 * (in_dim + 9), embedding_size, change_dim_first=True
            ),
            Sum(dim=1),
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, out_channels * out_dim),
            Rearrange("b (c d) -> b c d", c=out_channels, d=out_dim),
        )

    def forward(
        self, x: torch.Tensor, num_rows: int = 3, num_cols: int = 3
    ) -> torch.Tensor:
        num_context_panels = num_rows * num_cols
        x = self.tag_panel_embeddings(x, num_context_panels)
        x = self.relation_network(x)
        return x


class RelationNetworkSCAR(SCAR):
    def __init__(
        self,
        num_hidden_channels: int = 32,
        embedding_size: int = 128,
        ff_dim: int = 80,
        image_size: int = 160,
        local_kernel_size: int = 10,
        global_kernel_size: int = 10,
        sal_num_rows: int = 6,
        sal_num_cols: int = 420,
        ffblock: str = "pre-norm-residual",
    ):
        super().__init__(
            num_hidden_channels,
            embedding_size,
            ff_dim,
            image_size,
            local_kernel_size,
            global_kernel_size,
            sal_num_rows,
            sal_num_cols,
            ffblock,
        )
        in_dim = local_kernel_size * 8
        self.sal = RelationNetworkSAL(
            in_dim=in_dim, out_channels=64, out_dim=self.global_kernel_size
        )


class LstmSAL(nn.Module):
    def __init__(
        self, in_dim: int, out_channels: int, out_dim: int, embedding_size: int = 128
    ):
        super().__init__()
        self.tag_panel_embeddings = TagPanelEmbeddings()
        self.lstm = nn.Sequential(
            nn.Linear(in_dim + 9, embedding_size),
            nn.GELU(),
            nn.LayerNorm(embedding_size),
            nn.LSTM(
                batch_first=True,
                input_size=embedding_size,
                hidden_size=embedding_size,
                num_layers=1,
            ),
        )
        self.projection = nn.Sequential(
            nn.Linear(embedding_size, out_channels * out_dim),
            Rearrange("b (c d) -> b c d", c=out_channels, d=out_dim),
        )

    def forward(
        self, x: torch.Tensor, num_rows: int = 3, num_cols: int = 3
    ) -> torch.Tensor:
        num_context_panels = num_rows * num_cols
        x = self.tag_panel_embeddings(x, num_context_panels)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.projection(x)
        return x


class LstmSCAR(SCAR):
    def __init__(
        self,
        num_hidden_channels: int = 32,
        embedding_size: int = 128,
        ff_dim: int = 80,
        image_size: int = 160,
        local_kernel_size: int = 10,
        global_kernel_size: int = 10,
        sal_num_rows: int = 6,
        sal_num_cols: int = 420,
        ffblock: str = "pre-norm-residual",
    ):
        super().__init__(
            num_hidden_channels,
            embedding_size,
            ff_dim,
            image_size,
            local_kernel_size,
            global_kernel_size,
            sal_num_rows,
            sal_num_cols,
            ffblock,
        )
        in_dim = local_kernel_size * 8
        self.sal = LstmSAL(
            in_dim=in_dim, out_channels=64, out_dim=self.global_kernel_size
        )


class OddOneOutSCAR(SCAR):
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

        x = self.sal.forward(x, num_rows=1, num_cols=num_panels - 1)
        x = self.model_global(x)
        x = x.view(batch_size, num_panels, -1)
        return x


class OddOneOutRelationNetworkSCAR(RelationNetworkSCAR):
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

        x = self.sal.forward(x, num_rows=1, num_cols=num_panels - 1)
        x = self.model_global(x)
        x = x.view(batch_size, num_panels, -1)
        return x


class OddOneOutLstmSCAR(LstmSCAR):
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

        x = self.sal.forward(x, num_rows=1, num_cols=num_panels - 1)
        x = self.model_global(x)
        x = x.view(batch_size, num_panels, -1)
        return x
