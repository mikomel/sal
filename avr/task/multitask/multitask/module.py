from abc import ABC
from typing import Dict
from typing import List

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy, MultilabelAccuracy

from avr.task.avr_module import AVRModule
from avr.task.task import Task


class MultiTaskModule(AVRModule, ABC):
    def __init__(self, cfg: DictConfig, use_single_target_pred_head: bool = True):
        super().__init__(cfg, use_single_target_pred_head)
        self.val_losses = []

        self.tasks: List[Task] = instantiate(cfg.avr.tasks)

        if use_single_target_pred_head:
            target_pred_head = self.tasks[0].target_predictor.create(cfg)
            self.target_pred_heads = nn.ModuleDict(
                {task.name: target_pred_head for task in self.tasks}
            )
        else:
            self.target_pred_heads = nn.ModuleDict(
                {task.name: task.target_predictor.create(cfg) for task in self.tasks}
            )

        self.rule_pred_heads = nn.ModuleDict(
            {
                task.name: task.rule_predictor.create(
                    cfg, task.num_rules, num_answers=task.num_answers
                )
                for task in self.tasks
                if task.has_rules()
            }
        )

        self.target_loss = nn.CrossEntropyLoss()

        self.metrics = nn.ModuleDict(
            {
                split: nn.ModuleDict(
                    {
                        "acc": nn.ModuleDict(
                            {
                                "target": nn.ModuleDict(
                                    {
                                        **{
                                            "all": MulticlassAccuracy(
                                                num_classes=cfg.num_answers
                                            )
                                        },
                                        **{
                                            task.name: MulticlassAccuracy(
                                                num_classes=task.num_answers
                                            )
                                            for task in self.tasks
                                        },
                                    }
                                ),
                                "rules": nn.ModuleDict(
                                    {
                                        task.name: MultilabelAccuracy(
                                            num_labels=task.num_rules
                                        )
                                        for task in self.tasks
                                        if task.has_rules()
                                    }
                                ),
                            }
                        )
                    }
                )
                for split in ["tr", "val", "test"]
            }
        )

    def training_step(
        self, batch: Dict[str, List[List[torch.Tensor]]], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss = torch.tensor(0.0, device=self.device)
        for task in self.tasks:
            (context, answers), y = (
                batch[task.name][0][0],
                batch[task.name][0][1],
            ), batch[task.name][0][2]
            embedding = self.model(
                context, answers, num_rows=task.num_rows, num_cols=task.num_cols
            )
            y_hat = self.target_pred_heads[task.name](embedding)

            acc = self.metrics["tr"]["acc"]["target"][task.name](y_hat, y)
            self.log(
                f"tr/{task.name}/acc/target",
                acc,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
            )

            loss_target = self.target_loss(y_hat, y)
            self.log(
                f"tr/{task.name}/loss/target",
                loss_target,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            loss += task.target_loss_ratio * loss_target

            if task.has_rules():
                rules = batch[task.name][0][3]
                rules_hat = self.rule_pred_heads[task.name](embedding)
                loss_rules = F.binary_cross_entropy_with_logits(rules_hat, rules)
                self.log(
                    f"tr/{task.name}/loss/rules",
                    loss_rules,
                    on_epoch=True,
                    add_dataloader_idx=False,
                )
                loss += task.rules_loss_ratio * loss_rules

        self.log(
            "tr/loss", loss, on_epoch=True, prog_bar=True, add_dataloader_idx=False
        )

        return {"loss": loss}

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: int
    ) -> Dict[str, torch.Tensor]:
        task = self.tasks[dataloader_idx]
        (context, answers), y = (batch[0], batch[1]), batch[2]
        embedding = self.model(
            context, answers, num_rows=task.num_rows, num_cols=task.num_cols
        )

        y_hat = self.target_pred_heads[task.name](embedding)

        acc = self.metrics["val"]["acc"]["target"][task.name](y_hat, y)
        self.log(
            f"val/{task.name}/acc/target",
            acc,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        loss_target = self.target_loss(y_hat, y)
        self.log(
            f"val/{task.name}/loss/target",
            loss_target,
            on_epoch=True,
            add_dataloader_idx=False,
        )

        loss = task.target_loss_ratio * loss_target
        if task.has_rules():
            rules = batch[3]
            rules_hat = self.rule_pred_heads[task.name](embedding)
            loss_rules = F.binary_cross_entropy_with_logits(rules_hat, rules)
            self.log(
                f"val/{task.name}/loss/rules",
                loss_rules,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            loss += task.rules_loss_ratio * loss_rules

        self.val_losses.append(loss)

        return {"loss": loss}

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: int
    ) -> Dict[str, torch.Tensor]:
        task = self.tasks[dataloader_idx]
        (context, answers), y = (batch[0], batch[1]), batch[2]
        embedding = self.model(
            context, answers, num_rows=task.num_rows, num_cols=task.num_cols
        )

        y_hat = self.target_pred_heads[task.name](embedding)

        acc = self.metrics["test"]["acc"]["target"][task.name](y_hat, y)
        self.log(
            f"test/{task.name}/acc/target",
            acc,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )

        loss_target = self.target_loss(y_hat, y)
        self.log(
            f"test/{task.name}/loss/target",
            loss_target,
            on_epoch=True,
            add_dataloader_idx=False,
        )

        loss = task.target_loss_ratio * loss_target
        if task.has_rules():
            rules = batch[3]
            rules_hat = self.rule_pred_heads[task.name](embedding)
            loss_rules = F.binary_cross_entropy_with_logits(rules_hat, rules)
            self.log(
                f"test/{task.name}/loss/rules",
                loss_rules,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            loss += task.rules_loss_ratio * loss_rules

        return {"loss": loss}

    def on_validation_epoch_end(self) -> None:
        val_losses = torch.tensor(self.val_losses)
        val_loss = val_losses.mean()
        self.log(
            "val/loss",
            val_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )
        self.val_losses.clear()
