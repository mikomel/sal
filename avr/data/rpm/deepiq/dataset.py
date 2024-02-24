import os
import re
from abc import ABC
from typing import List, Tuple, Optional

import numpy as np
import torch
from skimage import color, io, transform
from torch.utils.data import Dataset

from avr.data.augmentor import AugmentorFactory
from avr.data.dataset import DatasetSplit, DEFAULT_DATASET_SPLITS
from avr.data.transform import shuffle_answers, to_tensor
from avr.data.transform import train_test_split
from avr.graphic.transform import resize


class DeepiqDataset(Dataset, ABC):
    FILEPATH_PATTERN = re.compile(r"(\d+)/_\w+.png")
    WIDTH = 160
    HEIGHT = 160

    def __init__(
        self,
        dataset_root_dir: str = ".",
        splits: List[DatasetSplit] = DEFAULT_DATASET_SPLITS,
        image_size: int = 80,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        do_shuffle_answers: bool = False,
        augmentor_factory: AugmentorFactory = AugmentorFactory.IDENTITY,
    ):
        assert train_ratio + val_ratio + test_ratio == 1.0
        self.dataset_root_dir = dataset_root_dir
        self.problem_ids, self.correct_answers = self._get_problems(
            splits, train_ratio, val_ratio
        )
        self.image_size = image_size
        self.do_shuffle_answers = do_shuffle_answers
        self.augmentor = augmentor_factory.create(image_size=image_size, num_panels=13)

    def __len__(self) -> int:
        return len(self.problem_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        context, answers = self.load(
            self.problem_ids[idx], dataset_root_dir=self.dataset_root_dir
        )
        context = np.array(
            [resize(p, self.image_size, self.image_size) for p in context]
        )
        answers = np.array(
            [resize(p, self.image_size, self.image_size) for p in answers]
        )

        images = np.concatenate([context, answers], axis=0)
        images = self.augmentor.augment(images)
        context, answers = images[:8], images[8:]

        correct_answer = self.correct_answers[idx]
        if self.do_shuffle_answers:
            answers, correct_answer = shuffle_answers(answers, correct_answer)

        context = to_tensor(context).unsqueeze(dim=1)
        answers = to_tensor(answers).unsqueeze(dim=1)
        return context, answers, correct_answer

    @classmethod
    def load(
        cls,
        problem_id: str,
        greyscale: bool = True,
        dataset_root_dir: str = ".",
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> Tuple[np.array, np.array]:
        context_path = f"{dataset_root_dir}/{problem_id}_test.png"
        context_image = io.imread(context_path)
        context_image = color.rgba2rgb(context_image)
        if greyscale:
            context_image = color.rgb2gray(context_image)
        context_image *= 255
        original_height, original_width = (
            context_image.shape[0] // 3,
            context_image.shape[1] // 3,
        )
        height = height if height else original_height
        width = width if width else original_width
        if height != original_height or width != original_width:
            context_image = transform.resize(
                context_image, (height * 3, width * 3), anti_aliasing=True
            )
        context = np.array(
            [
                context_image[
                    i * height : (i + 1) * height, j * width : (j + 1) * width
                ]
                for i in range(3)
                for j in range(3)
            ][:8]
        )

        answers_path = f"{dataset_root_dir}/{problem_id}_answers.png"
        answers_image = io.imread(answers_path)
        answers_image = color.rgba2rgb(answers_image)
        if greyscale:
            answers_image = color.rgb2gray(answers_image)
        answers_image *= 255

        original_height, original_width = (
            answers_image.shape[0],
            answers_image.shape[1] // 5,
        )
        height = height if height else original_height
        width = width if width else original_width
        if height != original_height or width != original_width:
            answers_image = transform.resize(
                answers_image, (height * 1, width * 5), anti_aliasing=True
            )
        answers = np.array(
            [answers_image[:, j * width : (j + 1) * width] for j in range(5)]
        )

        return context, answers

    def _get_problems(
        self, splits: List[DatasetSplit], train_ratio: float, val_ratio: float
    ) -> Tuple[List[str], List[int]]:
        with open(f"{self.dataset_root_dir}/answers.csv", "r") as f:
            all_correct_answers = [int(answer) for answer in f.readlines()]
        all_problem_ids = list(
            sorted(
                [
                    f[:4]
                    for f in os.listdir(self.dataset_root_dir)
                    if f.endswith("answers.png")
                ]
            )
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
