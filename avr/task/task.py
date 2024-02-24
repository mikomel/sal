from typing import Optional

from avr.model.rule_predictor import RulePredictor
from avr.model.target_predictor import TargetPredictor


class Task:
    def __init__(
        self,
        problem: str,
        dataset: str,
        num_answers: int,
        target_predictor: TargetPredictor,
        rule_predictor: Optional[RulePredictor] = None,
        num_rows: int = 3,
        num_cols: int = 3,
        target_loss_ratio: float = 1.0,
        num_rules: int = 0,
        rules_loss_ratio: float = 0.0,
        name: Optional[str] = None,
    ):
        self.name = name if name else dataset
        self.problem = problem
        self.dataset = dataset
        self.num_answers = num_answers
        self.target_predictor = target_predictor
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.target_loss_ratio = target_loss_ratio
        self.num_rules = num_rules
        self.rule_predictor = rule_predictor
        self.rules_loss_ratio = rules_loss_ratio

    def has_rules(self) -> bool:
        return self.num_rules > 0
