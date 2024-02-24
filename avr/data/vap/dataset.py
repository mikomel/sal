import glob
import os
import re
from abc import ABC
from typing import List
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from avr.data.augmentor import AugmentorFactory
from avr.data.dataset import DatasetSplit, DEFAULT_DATASET_SPLITS
from avr.data.transform import shuffle_answers, to_tensor
from avr.data.vap.regime import VisualAnalogyRegime, AnswerGenerationStrategy
from avr.data.vap.rule_encoder import VAPRuleEncoder, DenseVAPRuleEncoder
from avr.graphic.transform import resize


class VisualAnalogyDataset(Dataset, ABC):
    FILEPATH_PATTERN = re.compile(r".*/analogy_([\w\.]+)_(\w+)_(\w+)_(\d+).npz")

    def __init__(
        self,
        dataset_root_dir: str = ".",
        regime: VisualAnalogyRegime = VisualAnalogyRegime.NOVEL_DOMAIN_TRANSFER,
        answer_generation_strategy: AnswerGenerationStrategy = AnswerGenerationStrategy.NORMAL,
        splits: List[DatasetSplit] = DEFAULT_DATASET_SPLITS,
        rule_encoder: VAPRuleEncoder = DenseVAPRuleEncoder(),
        do_shuffle_answers: bool = False,
        image_size: int = 160,
        augmentor_factory: AugmentorFactory = AugmentorFactory.IDENTITY,
    ):
        self.dataset_root_dir = dataset_root_dir
        self.regime = regime
        self.answer_generation_strategy = answer_generation_strategy
        self.split_names = [s.value for s in splits]
        self.filenames = self._get_filenames()
        self.rule_encoder = rule_encoder
        self.do_shuffle_answers = do_shuffle_answers
        self.image_size = image_size
        self.augmentor = augmentor_factory.create(image_size=image_size, num_panels=9)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        with np.load(self.filenames[idx], allow_pickle=True) as data:
            images = data["image"]
            h, w, c = images.shape
            images = np.ascontiguousarray(images.reshape(c, h, w))
            images = np.stack(
                [resize(image, self.image_size, self.image_size) for image in images]
            )
            images = self.augmentor.augment(images)
            context, answers = images[:5], images[5:]
            target = data["target"]
            if self.do_shuffle_answers:
                answers, target = shuffle_answers(answers, target)

            context = to_tensor(context).unsqueeze(dim=1)
            answers = to_tensor(answers).unsqueeze(dim=1)
            rules = self.rule_encoder.encode(data)
            return context, answers, target, rules

    def _get_filenames(self):
        filenames = []
        filename_pattern = os.path.join(
            self.dataset_root_dir, self.regime.value, "*.npz"
        )
        filename_pattern = os.path.expanduser(filename_pattern)
        for f in glob.glob(filename_pattern):
            if self._should_contain_filename(f):
                filenames.append(f)
        return filenames

    def _should_contain_filename(self, filename: str):
        filename = self._split_filename(filename)
        return (
            filename["regime"] == self.regime.value
            and filename["split"] in self.split_names
            and (
                AnswerGenerationStrategy.MIX == self.answer_generation_strategy
                or filename["answer_generation_strategy"]
                == self.answer_generation_strategy.value
            )
        )

    def _split_filename(self, filename: str):
        match = re.match(self.FILEPATH_PATTERN, filename)
        return {
            "regime": match.group(1),
            "split": match.group(2),
            "answer_generation_strategy": match.group(3),
            "id": match.group(4),
        }
