import os
import re
from typing import List, Tuple

import numpy as np
import torch
from skimage import color, io, transform
from torch.utils.data import Dataset

from avr.data.augmentor import AugmentorFactory
from avr.data.dataset import DatasetSplit, DEFAULT_DATASET_SPLITS
from avr.data.transform import shuffle_answers, to_tensor
from avr.data.transform import train_test_split


class DeepiqDataset(Dataset):
    FILEPATH_PATTERN = re.compile(r"(\d+).png")

    def __init__(
        self,
        dataset_root_dir: str = ".",
        splits: List[DatasetSplit] = DEFAULT_DATASET_SPLITS,
        image_size: int = 160,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        do_shuffle_panels: bool = False,
        augmentor_factory: AugmentorFactory = AugmentorFactory.IDENTITY,
        num_panels: List[int] = (4, 5),
    ):
        assert train_ratio + val_ratio + test_ratio == 1.0
        for n in num_panels:
            assert n in (4, 5)
        self.dataset_root_dir = dataset_root_dir
        self.problem_ids, self.correct_answers = self._get_problems(
            splits, train_ratio, val_ratio, num_panels
        )
        self.image_size = image_size
        self.do_shuffle_panels = do_shuffle_panels
        self.augmentor = augmentor_factory.create(
            image_size, num_panels=max(num_panels)
        )

    def __len__(self) -> int:
        return len(self.problem_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = io.imread(f"{self.dataset_root_dir}/{self.problem_ids[idx]}.png")
        num_panels = image.shape[1] // 100
        image = color.rgba2rgb(image)
        image = color.rgb2gray(image)
        image *= 255
        image = transform.resize(
            image,
            (self.image_size, self.image_size * num_panels),
            anti_aliasing=True,
        )
        context = np.array(
            [
                image[:, i * self.image_size : (i + 1) * self.image_size]
                for i in range(num_panels)
            ]
        )
        context = self.augmentor.augment(context)

        correct_answer = self.correct_answers[idx]
        if self.do_shuffle_panels:
            context, correct_answer = shuffle_answers(context, correct_answer)
        context = to_tensor(context)

        return context.unsqueeze(dim=1), correct_answer

    def _get_problems(
        self,
        splits: List[DatasetSplit],
        train_ratio: float,
        val_ratio: float,
        num_panels: List[int],
    ) -> Tuple[List[str], List[int]]:
        with open(f"{self.dataset_root_dir}/answers.csv", "r") as f:
            all_correct_answers = [int(answer) for answer in f.readlines()]
        all_problem_ids = list(
            sorted(
                [f[:4] for f in os.listdir(self.dataset_root_dir) if f.endswith(".png")]
            )
        )
        if 4 in num_panels and not 5 in num_panels:
            all_problem_ids, all_correct_answers = (
                all_problem_ids[:500],
                all_correct_answers[:500],
            )
        elif 4 not in num_panels and 5 in num_panels:
            all_problem_ids, all_correct_answers = (
                all_problem_ids[500:],
                all_correct_answers[500:],
            )
        ids_train, ids_val, answers_train, answers_val = train_test_split(
            all_problem_ids,
            all_correct_answers,
            train_size=train_ratio,
            stratify=all_correct_answers,
            random_state=42,
        )
        ids_val, ids_test, answers_val, answers_test = train_test_split(
            ids_val,
            answers_val,
            train_size=val_ratio,
            stratify=answers_val,
            random_state=42,
        )
        problem_ids = []
        correct_answers = []
        if DatasetSplit.TRAIN in splits:
            problem_ids.extend(ids_train)
            correct_answers.extend(answers_train)
        if DatasetSplit.VAL in splits:
            problem_ids.extend(ids_val)
            correct_answers.extend(answers_val)
        if DatasetSplit.TEST in splits:
            problem_ids.extend(ids_test)
            correct_answers.extend(answers_test)
        return problem_ids, correct_answers
