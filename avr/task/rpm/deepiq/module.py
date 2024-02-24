from typing import Dict

import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from avr.model.target_predictor import TargetPredictor
from avr.task.avr_module import AVRModule


class DeepiqModule(AVRModule):
    def __init__(self, cfg: DictConfig, target_predictor: TargetPredictor):
        super(DeepiqModule, self).__init__(cfg)
        self.target_pred_head = target_predictor.create(cfg)
        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict(
            {
                "tr": nn.ModuleDict(
                    {"acc": MulticlassAccuracy(num_classes=cfg.num_answers)}
                ),
                "val": nn.ModuleDict(
                    {"acc": MulticlassAccuracy(num_classes=cfg.num_answers)}
                ),
                "test": nn.ModuleDict(
                    {"acc": MulticlassAccuracy(num_classes=cfg.num_answers)}
                ),
            }
        )

    def _step(self, split: str, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        context, answers, y = batch
        embedding = self.model(context, answers)

        y_hat = self.target_pred_head(embedding)
        loss = self.loss(y_hat, y)
        acc = self.log_target_metrics(split, y, y_hat, loss)

        return {"loss": loss, "acc": acc}

    def training_step(self, batch, batch_idx):
        return self._step("tr", batch, batch_idx)

    def validation_step(self, batch, batch_idx: int):
        self._step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx: int):
        self._step("test", batch, batch_idx)

    def log_target_metrics(
        self, split: str, y: torch.Tensor, y_hat: torch.Tensor, loss: torch.Tensor
    ) -> torch.Tensor:
        self.log(
            f"{split}/loss/target",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )
        acc = self.metrics[split]["acc"](y_hat, y)
        self.log(
            f"{split}/acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )
        return acc
