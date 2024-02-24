import itertools
from typing import Any, Tuple

import numpy as np
import torch
from sklearn import model_selection


def to_tensor(image: np.array) -> torch.Tensor:
    image = image.astype("float32") / 255.0
    return torch.tensor(image)


def train_test_split(
    *arrays,
    test_size: float = None,
    train_size: float = None,
    random_state: int = None,
    shuffle: bool = True,
    stratify: Any = None
) -> Any:
    """
    Wrapper for sklearn.model_selection.train_test_split
    that additionally supports train_size and test_size equal to 0.0.
    """
    if (train_size is not None and 0 < train_size < 1) or (
        test_size is not None and 0 < test_size < 1
    ):
        return model_selection.train_test_split(
            *arrays,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify
        )
    elif train_size == 0:
        return list(itertools.chain.from_iterable(zip([[]] * len(arrays), arrays)))
    else:
        return list(itertools.chain.from_iterable(zip(arrays, [[]] * len(arrays))))


def shuffle_answers(panels: np.array, target: int) -> Tuple[np.array, int]:
    indices = list(range(len(panels)))
    np.random.shuffle(indices)
    return panels[indices], indices.index(target)


def shuffle_panels(images: np.array) -> np.array:
    indices = list(range(len(images)))
    np.random.shuffle(indices)
    return images[indices]
