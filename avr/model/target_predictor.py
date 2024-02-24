from enum import Enum

from torch import nn


class TargetPredictor(Enum):
    LINEAR = "linear"
    MLP = "mlp"

    def create(self, cfg) -> nn.Module:
        d = cfg.avr.model.embedding_size

        if self == TargetPredictor.LINEAR:
            return nn.Linear(d, 1)

        elif self == TargetPredictor.MLP:
            d = cfg.avr.model.embedding_size
            return nn.Sequential(
                nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1), nn.Flatten(-2, -1)
            )

        else:
            raise ValueError()
