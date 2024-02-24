from typing import Dict, Optional

import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from avr.model.target_predictor import TargetPredictor
from avr.task.avr_module import AVRModule


class DeepIQModule(AVRModule):
    def __init__(self, cfg: DictConfig, target_predictor: TargetPredictor):
        super(DeepIQModule, self).__init__(cfg)
        self.classifier = target_predictor.create(cfg)
        self.loss = nn.CrossEntropyLoss()
        self.accs = {"val": [], "test": []}
        self.losses = {"val": [], "test": []}
        self.metrics = nn.ModuleDict(
            {
                "tr": nn.ModuleDict(
                    {
                        "acc": nn.ModuleDict(
                            {str(n): MulticlassAccuracy(num_classes=n) for n in [4, 5]}
                        )
                    }
                ),
                "val": nn.ModuleDict(
                    {
                        "acc": nn.ModuleDict(
                            {str(n): MulticlassAccuracy(num_classes=n) for n in [4, 5]}
                        )
                    }
                ),
                "test": nn.ModuleDict(
                    {
                        "acc": nn.ModuleDict(
                            {str(n): MulticlassAccuracy(num_classes=n) for n in [4, 5]}
                        )
                    }
                ),
            }
        )

    def training_step(
        self, batch, batch_idx, dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        (x, y) = batch[4]
        embedding = self.model(x)
        y_hat = self.classifier(embedding).squeeze()
        loss_4 = self.loss(y_hat, y)
        self.log(
            "tr/4/loss", loss_4, on_epoch=True, logger=True, add_dataloader_idx=False
        )
        acc_4 = self.metrics["tr"]["acc"]["4"](y_hat, y)
        self.log(
            "tr/4/acc", acc_4, on_epoch=True, logger=True, add_dataloader_idx=False
        )

        (x, y) = batch[5]
        embedding = self.model(x)
        y_hat = self.classifier(embedding).squeeze()
        loss_5 = self.loss(y_hat, y)
        self.log(
            "tr/5/loss", loss_5, on_epoch=True, logger=True, add_dataloader_idx=False
        )
        acc_5 = self.metrics["tr"]["acc"]["5"](y_hat, y)
        self.log(
            "tr/5/acc", acc_5, on_epoch=True, logger=True, add_dataloader_idx=False
        )

        return {"loss": loss_4 + loss_5, "acc_4": acc_4, "acc_5": acc_5}

    def _test_step(
        self, split: str, batch, dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        num_panels = dataloader_idx + 4
        x, y = batch
        embedding = self.model(x)
        y_hat = self.classifier(embedding).squeeze()

        loss = self.loss(y_hat, y)
        self.log(
            f"{split}/{num_panels}/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )

        acc = self.metrics[split]["acc"][str(num_panels)](y_hat, y)
        self.log(
            f"{split}/{num_panels}/acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )

        self.losses[split].append(loss)
        self.accs[split].append(acc)

        return {"loss": loss, "acc": acc}

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        return self._test_step("val", batch, dataloader_idx)

    def test_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        return self._test_step("test", batch, dataloader_idx)

    def on_validation_epoch_end(self) -> None:
        self._on_test_epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._on_test_epoch_end("test")

    def _on_test_epoch_end(self, split: str) -> None:
        losses = torch.tensor(self.losses[split])
        loss = losses.mean()
        self.log(
            f"{split}/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )
        self.losses[split].clear()

        accs = torch.tensor(self.accs[split])
        acc = accs.mean()
        self.log(
            f"{split}/acc",
            acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )
        self.accs[split].clear()
