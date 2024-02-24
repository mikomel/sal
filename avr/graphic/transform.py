import cv2 as cv
import numpy as np


def resize(image: np.array, height: int, width: int) -> np.array:
    return cv.resize(
        np.asarray(image, dtype=float),
        dsize=(width, height),
        interpolation=cv.INTER_AREA,
    )
