from collections import defaultdict
from typing import Optional, Dict

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy, MultilabelAccuracy

from avr.data.rpm.raven.configuration import RavenConfiguration
from avr.model.rule_predictor import RulePredictor
from avr.model.target_predictor import TargetPredictor
from avr.task.avr_module import AVRModule


class RavenModule(AVRModule):
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
        self.outputs = {
            "val": defaultdict(lambda: defaultdict(list)),
            "test": defaultdict(lambda: defaultdict(list)),
        }
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
                        ),
                        "accs": nn.ModuleList(
                            [
                                nn.ModuleDict(
                                    {
                                        "target": MulticlassAccuracy(
                                            num_classes=cfg.num_answers
                                        ),
                                        "rules": MultilabelAccuracy(
                                            num_labels=cfg.num_rules
                                        ),
                                    }
                                )
                                for _ in range(self.get_num_val_configurations())
                            ]
                        ),
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
                        ),
                        "accs": nn.ModuleList(
                            [
                                nn.ModuleDict(
                                    {
                                        "target": MulticlassAccuracy(
                                            num_classes=cfg.num_answers
                                        ),
                                        "rules": MultilabelAccuracy(
                                            num_labels=cfg.num_rules
                                        ),
                                    }
                                )
                                for _ in range(self.get_num_test_configurations())
                            ]
                        ),
                    }
                ),
            }
        )

    def _step(
        self, split: str, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        rules_loss_scaling = 10.0
        context, answers, y, rules = batch
        embedding = self.model(context, answers)

        y_hat = self.target_pred_head(embedding)
        target_loss = self.target_loss(y_hat, y)
        acc = self.log_target_metrics(split, y, y_hat, target_loss, dataloader_idx)

        rules_hat = self.rule_pred_head(embedding)
        rules_loss = F.binary_cross_entropy_with_logits(rules_hat, rules)
        self.log_auxiliary_metrics(split, rules, rules_hat, rules_loss, dataloader_idx)

        loss = target_loss + rules_loss_scaling * rules_loss
        if dataloader_idx is not None:
            configuration = self.get_test_configuration(dataloader_idx).short_name()
            self.logm_configuration(loss, "loss", split, configuration)
        else:
            self.logm(loss, "loss", split)
        return {"loss": loss, "acc": acc}

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        return self._step("tr", batch, batch_idx)

    def _eval_step(
        self, split: str, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        metrics = self._step(split, batch, batch_idx, dataloader_idx)
        for k, v in metrics.items():
            self.outputs[split][k][dataloader_idx].append(v)
        return metrics

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        return self._eval_step("val", batch, batch_idx, dataloader_idx)

    def _on_eval_epoch_end(self, split: str) -> None:
        for metric, dataloader_index_to_values in self.outputs[split].items():
            mean_value = torch.stack(
                [
                    value
                    for values in dataloader_index_to_values.values()
                    for value in values
                ]
            ).mean()
            self.logm_type(mean_value, metric, split, "target")
        self.outputs[split].clear()

    def on_validation_epoch_end(self) -> None:
        self._on_eval_epoch_end("val")

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        return self._eval_step("test", batch, batch_idx, dataloader_idx)

    def on_test_epoch_end(self) -> None:
        self._on_eval_epoch_end("test")

    def log_target_metrics(
        self,
        split: str,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        loss: torch.Tensor,
        dataloader_idx: Optional[int] = None,
    ) -> torch.Tensor:
        if dataloader_idx is not None:
            configuration = self.get_configuration(split, dataloader_idx).short_name()
            self.logm_configuration_type(loss, "loss", split, configuration, "target")
            acc_configuration = self.metrics[split]["accs"][dataloader_idx]["target"](
                y_hat, y
            )
            self.logm_configuration_type(
                acc_configuration, "acc", split, configuration, "target"
            )
            return acc_configuration
        else:
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
        dataloader_idx: Optional[int] = None,
    ):
        if dataloader_idx is not None:
            configuration = self.get_configuration(split, dataloader_idx).short_name()
            self.logm_configuration_type(loss, "loss", split, configuration, "rules")
            # acc_rule_configuration = self.metrics[split]['accs'][dataloader_idx]['rules'](rules_hat.sigmoid(), rules.int())
            # self.logm_configuration_type(acc_rule_configuration, 'acc', split, configuration, 'rules')
        else:
            self.logm_type(loss, "loss", split, "rules")
            # acc_rule = self.metrics[split]['acc']['rules'](rules_hat.sigmoid(), rules.int())
            # self.logm_type(acc_rule, 'acc', split, 'rules')

    def get_configuration(self, split: str, dataloader_idx: int) -> RavenConfiguration:
        if split == "val":
            return self.cfg.avr.data.rpm.raven.val.configurations[dataloader_idx]
        elif split == "test":
            return self.cfg.avr.data.rpm.raven.test.configurations[dataloader_idx]
        else:
            raise ValueError(f"No configuration for split: {split}")

    def get_val_configuration(self, dataloader_idx: int) -> RavenConfiguration:
        return self.cfg.avr.data.rpm.raven.val.configurations.get(
            dataloader_idx, dataloader_idx
        )

    def get_test_configuration(self, dataloader_idx: int) -> RavenConfiguration:
        return self.cfg.avr.data.rpm.raven.test.configurations.get(
            dataloader_idx, dataloader_idx
        )

    def get_num_val_configurations(self) -> int:
        return len(self.cfg.avr.data.rpm.raven.val.configurations)

    def get_num_test_configurations(self) -> int:
        return len(self.cfg.avr.data.rpm.raven.test.configurations)
