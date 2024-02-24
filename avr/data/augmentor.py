from enum import Enum
from typing import List

import albumentations
import numpy as np
from albumentations import (
    VerticalFlip,
    HorizontalFlip,
    RandomRotate90,
    Rotate,
    Transpose,
    OneOf,
    RandomGridShuffle,
)

from avr.data.augmentation import Roll, HorizontalRoll, VerticalRoll


class Augmentor:
    def __init__(self, num_panels: int, p: float = 0.0, transforms: List = ()):
        self.num_panels = num_panels
        self.transform = albumentations.Compose(
            transforms,
            additional_targets={
                f"image{i + 1}": "image" for i in range(num_panels - 1)
            },
            p=p,
        )

    def augment(self, images: np.array) -> np.array:
        """
        Applies the same Albumentation transform to all images from the matrix.
        :param images: numpy array representing images of matrix with shape (num_panels, width, height)
        :return: numpy array with augmented RPM images with shape (num_panels, width, height)
        """
        num_panels = images.shape[0]
        kwargs = {f"image{i + 1}": images[i + 1, :, :] for i in range(num_panels - 1)}
        augmented_images = self.transform(image=images[0, :, :], **kwargs)
        return np.stack(
            [
                augmented_images[f"image{i}"] if i > 0 else augmented_images["image"]
                for i in range(num_panels)
            ]
        )


class IdentityAugmentor(Augmentor):
    def augment(self, images: np.array) -> np.array:
        return images


IDENTITY_AUGMENTOR = IdentityAugmentor(num_panels=-1)


class AugmentorFactory(Enum):
    IDENTITY = "identity"
    SIMPLE = "simple"
    MLCL = "mlcl"

    def create(self, image_size: int, num_panels: int) -> Augmentor:
        if self == AugmentorFactory.IDENTITY:
            return IdentityAugmentor(num_panels=num_panels)
        elif self == AugmentorFactory.SIMPLE:
            return Augmentor(
                num_panels=num_panels,
                p=0.5,
                transforms=[
                    VerticalFlip(p=0.25),
                    HorizontalFlip(p=0.25),
                    RandomRotate90(p=0.25),
                    Rotate(p=0.25),
                    Transpose(p=0.25),
                ],
            )
        elif self == AugmentorFactory.MLCL:
            return Augmentor(
                num_panels=num_panels,
                p=0.5,
                transforms=[
                    VerticalFlip(p=0.25),
                    HorizontalFlip(p=0.25),
                    RandomRotate90(p=0.25),
                    Rotate(p=0.25),
                    Transpose(p=0.25),
                    OneOf(
                        [
                            RandomGridShuffle(grid=(2, 2)),
                            RandomGridShuffle(grid=(3, 3)),
                        ],
                        p=0.25,
                    ),
                    OneOf(
                        [
                            Roll(
                                p=0.4,
                                max_horizontal_shift=image_size,
                                max_vertical_shift=image_size,
                            ),
                            HorizontalRoll(p=0.3, max_shift=image_size),
                            VerticalRoll(p=0.3, max_shift=image_size),
                        ],
                        p=0.5,
                    ),
                ],
            )
        else:
            raise ValueError()
