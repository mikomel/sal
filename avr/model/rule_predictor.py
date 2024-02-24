from enum import Enum

from torch import nn

from avr.model.neural_blocks import Sum


class RulePredictor(Enum):
    FLAT_LINEAR = 'flat-linear'
    FLAT_ACT = 'flat-act'
    ACT_FLAT_LINEAR = 'act-flat-linear'
    ACT_FLAT_ACT = 'act-flat-act'
    SUM_LINEAR = 'sum-linear'
    SUM_ACT = 'sum-act'
    ACT_SUM_LINEAR = 'act-sum-linear'
    ACT_SUM_ACT = 'act-sum-act'

    def create(self, cfg, num_rules: int, num_answers: int) -> nn.Module:
        d = cfg.avr.model.embedding_size
        p = num_answers
        r = num_rules

        if self == RulePredictor.FLAT_LINEAR:
            return nn.Sequential(
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(p * d, r))

        if self == RulePredictor.FLAT_ACT:
            return nn.Sequential(
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(p * d, d),
                nn.GELU(),
                nn.Linear(d, r))

        if self == RulePredictor.ACT_FLAT_LINEAR:
            return nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(p * d, r))

        if self == RulePredictor.ACT_FLAT_ACT:
            return nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(p * d, d),
                nn.GELU(),
                nn.Linear(d, r))

        elif self == RulePredictor.SUM_LINEAR:
            return nn.Sequential(
                Sum(dim=-2),
                nn.Linear(d, r))

        elif self == RulePredictor.SUM_ACT:
            return nn.Sequential(
                Sum(dim=-2),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, r))

        elif self == RulePredictor.ACT_SUM_LINEAR:
            return nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                Sum(dim=-2),
                nn.Linear(d, r))

        elif self == RulePredictor.ACT_SUM_ACT:
            return nn.Sequential(
                nn.Linear(d, d),
                nn.GELU(),
                Sum(dim=-2),
                nn.Linear(d, d),
                nn.GELU(),
                nn.Linear(d, r))

        else:
            raise ValueError()
