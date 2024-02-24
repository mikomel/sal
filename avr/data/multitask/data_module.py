from typing import List, Dict, Optional

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from avr.task.task import Task


class MultiTaskDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(MultiTaskDataModule, self).__init__()
        self.cfg: DictConfig = cfg
        self.tasks: List[Task] = instantiate(cfg.avr.tasks)
        self.train_datasets: Dict[str, List[Dataset]] = {}
        self.val_datasets: Dict[str, List[Dataset]] = {}
        self.test_datasets: Dict[str, List[Dataset]] = {}

    def setup(self, stage: Optional[str] = None):
        self.train_datasets = {
            task.name: [
                instantiate(
                    self.cfg["avr"]["data"][task.problem][task.dataset]["train"]
                )
            ]
            for task in self.tasks
        }
        self.val_datasets = {
            task.name: [
                instantiate(self.cfg["avr"]["data"][task.problem][task.dataset]["val"])
            ]
            for task in self.tasks
        }
        self.test_datasets = {
            task.name: [
                instantiate(self.cfg["avr"]["data"][task.problem][task.dataset]["test"])
            ]
            for task in self.tasks
        }

    def train_dataloader(self) -> Dict[str, List[DataLoader]]:
        return {
            task: [
                instantiate(self.cfg.torch.data_loader.train, dataset=ds, shuffle=True)
                for ds in datasets
            ]
            for task, datasets in self.train_datasets.items()
        }

    def val_dataloader(self) -> List[DataLoader]:
        return [
            instantiate(self.cfg.torch.data_loader.val, dataset=ds, shuffle=False)
            for _, datasets in self.val_datasets.items()
            for ds in datasets
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return [
            instantiate(self.cfg.torch.data_loader.test, dataset=ds, shuffle=False)
            for _, datasets in self.test_datasets.items()
            for ds in datasets
        ]
