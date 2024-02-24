from typing import Dict, List, Optional

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from avr.data.ooo.deepiq.dataset import DeepiqDataset


class DeepiqDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(DeepiqDataModule, self).__init__()
        self.cfg: DictConfig = cfg
        self.train_datasets: Dict[DeepiqDataset] = {}
        self.val_datasets: Dict[DeepiqDataset] = {}
        self.test_datasets: Dict[DeepiqDataset] = {}

    def setup(self, stage: Optional[str] = None):
        self.train_datasets = {
            n: instantiate(self.cfg.avr.data.ooo.deepiq.train, num_panels=[n])
            for n in [4, 5]
        }
        self.val_datasets = {
            n: instantiate(self.cfg.avr.data.ooo.deepiq.val, num_panels=[n])
            for n in [4, 5]
        }
        self.test_datasets = {
            n: instantiate(self.cfg.avr.data.ooo.deepiq.test, num_panels=[n])
            for n in [4, 5]
        }

    def train_dataloader(self) -> Dict[int, DataLoader]:
        return {
            n: instantiate(self.cfg.torch.data_loader.train, dataset=ds, shuffle=True)
            for n, ds in self.train_datasets.items()
        }

    def val_dataloader(self) -> List[DataLoader]:
        return [
            instantiate(self.cfg.torch.data_loader.val, dataset=ds)
            for _, ds in self.val_datasets.items()
        ]

    def test_dataloader(self) -> List[DataLoader]:
        return [
            instantiate(self.cfg.torch.data_loader.test, dataset=ds)
            for _, ds in self.test_datasets.items()
        ]
