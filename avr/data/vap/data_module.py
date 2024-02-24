from typing import Optional

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from avr.data.vap.dataset import VisualAnalogyDataset


class VAPDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super(VAPDataModule, self).__init__()
        self.cfg: DictConfig = cfg
        self.train_dataset: VisualAnalogyDataset = None
        self.val_dataset: VisualAnalogyDataset = None
        self.test_dataset: VisualAnalogyDataset = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = instantiate(self.cfg.avr.data.vap.hill2019learning.train)
        self.val_dataset = instantiate(self.cfg.avr.data.vap.hill2019learning.val)
        self.test_dataset = instantiate(self.cfg.avr.data.vap.hill2019learning.test)

    def train_dataloader(self) -> DataLoader:
        return instantiate(
            self.cfg.torch.data_loader.train, dataset=self.train_dataset, shuffle=True
        )

    def val_dataloader(self) -> DataLoader:
        return instantiate(self.cfg.torch.data_loader.val, dataset=self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return instantiate(self.cfg.torch.data_loader.test, dataset=self.test_dataset)
