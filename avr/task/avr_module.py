from abc import ABC

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer


class AVRModule(pl.LightningModule, ABC):
    def __init__(self, cfg: DictConfig):
        super(AVRModule, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = instantiate(cfg.avr.model)

    def configure_optimizers(self):
        optimizer: Optimizer = instantiate(
            self.cfg.torch.optimizer, params=self.parameters()
        )
        scheduler: object = instantiate(self.cfg.torch.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.cfg.monitor},
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def logm(self, value: torch.Tensor, metric: str, split: str):
        self.log(
            f"{split}/{metric}",
            value,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )

    def logm_type(self, value: torch.Tensor, metric: str, split: str, type: str):
        self.log(
            f"{split}/{metric}/{type}",
            value,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )

    def logm_configuration(
        self, value: torch.Tensor, metric: str, split: str, configuration: str
    ):
        self.log(
            f"{split}/{configuration}/{metric}",
            value,
            on_epoch=True,
            logger=True,
            add_dataloader_idx=False,
        )

    def logm_configuration_type(
        self,
        value: torch.Tensor,
        metric: str,
        split: str,
        configuration: str,
        type: str,
    ):
        self.log(
            f"{split}/{configuration}/{metric}/{type}",
            value,
            on_epoch=True,
            logger=True,
            add_dataloader_idx=False,
        )
