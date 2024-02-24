from typing import Optional, Dict

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy, MultilabelAccuracy

from avr.model.rule_predictor import RulePredictor
from avr.model.target_predictor import TargetPredictor
from avr.task.avr_module import AVRModule


class PgmModule(AVRModule):
    def __init__(
        self,
        cfg: DictConfig,
        target_predictor: TargetPredictor,
        rule_predictor: RulePredictor,
    ):
        super().__init__(cfg)
        self.target_pred_head = target_predictor.create(cfg)
        self.rule_pred_head = rule_predictor.create(cfg, cfg.num_rules, cfg.num_answers)
        self.target_loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict(
            {
                "tr": nn.ModuleDict(
                    {
                        "acc": nn.ModuleDict(
                            {
                                "target": MulticlassAccuracy(
                                    num_classes=cfg.num_answers
                                ),
                                "rules": MultilabelAccuracy(num_labels=cfg.num_rules),
                            }
                        )
                    }
                ),
                "val": nn.ModuleDict(
                    {
                        "acc": nn.ModuleDict(
                            {
                                "target": MulticlassAccuracy(
                                    num_classes=cfg.num_answers
                                ),
                                "rules": MultilabelAccuracy(num_labels=cfg.num_rules),
                            }
                        )
                    }
                ),
                "test": nn.ModuleDict(
                    {
                        "acc": nn.ModuleDict(
                            {
                                "target": MulticlassAccuracy(
                                    num_classes=cfg.num_answers
                                ),
                                "rules": MultilabelAccuracy(num_labels=cfg.num_rules),
                            }
                        )
                    }
                ),
            }
        )

    def _step(self, split: str, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        rules_loss_scaling = 10.0
        context, answers, y, rules = batch
        embedding = self.model(context, answers)

        y_hat = self.target_pred_head(embedding)
        target_loss = self.target_loss(y_hat, y)
        acc = self.log_target_metrics(split, y, y_hat, target_loss)

        rules_hat = self.rule_pred_head(embedding)
        rules_loss = F.binary_cross_entropy_with_logits(rules_hat, rules)
        self.log_auxiliary_metrics(split, rules, rules_hat, rules_loss)

        loss = target_loss + rules_loss_scaling * rules_loss
        self.logm(loss, "loss", split)
        return {"loss": loss, "acc": acc}

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        return self._step("tr", batch, batch_idx)

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        return self._step("val", batch, batch_idx)

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        return self._step("test", batch, batch_idx)

    def log_target_metrics(
        self, split: str, y: torch.Tensor, y_hat: torch.Tensor, loss: torch.Tensor
    ) -> torch.Tensor:
        self.logm_type(loss, "loss", split, "target")
        acc = self.metrics[split]["acc"]["target"](y_hat, y)
        self.logm_type(acc, "acc", split, "target")
        return acc

    def log_auxiliary_metrics(
        self,
        split: str,
        rules: torch.Tensor,
        rules_hat: torch.Tensor,
        loss: torch.Tensor,
    ):
        self.logm_type(loss, "loss", split, "rules")
        # acc_rule = self.metrics[split]['acc']['rules'](rules_hat.sigmoid(), rules.int())
        # self.logm_type(acc_rule, 'acc', split, 'rules')
