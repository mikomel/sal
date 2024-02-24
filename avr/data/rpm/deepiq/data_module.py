from typing import List, Optional

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from avr.data.rpm.deepiq.dataset import DeepiqDataset


class DeepiqDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(DeepiqDataModule, self).__init__()
        self.cfg: DictConfig = cfg
        self.train_dataset: DeepiqDataset = None
        self.val_dataset: DeepiqDataset = None
        self.test_dataset: DeepiqDataset = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = instantiate(self.cfg.avr.data.rpm.deepiq.train)
        self.val_dataset = instantiate(self.cfg.avr.data.rpm.deepiq.val)
        self.test_dataset = instantiate(self.cfg.avr.data.rpm.deepiq.test)

    def train_dataloader(self) -> DataLoader:
        return instantiate(
            self.cfg.torch.data_loader.train, dataset=self.train_dataset, shuffle=True
        )

    def val_dataloader(self) -> List[DataLoader]:
        return instantiate(self.cfg.torch.data_loader.val, dataset=self.val_dataset)

    def test_dataloader(self) -> List[DataLoader]:
        return instantiate(self.cfg.torch.data_loader.test, dataset=self.test_dataset)
