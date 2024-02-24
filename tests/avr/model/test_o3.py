from typing import Callable

import pytest
import torch

from avr.model.scar import (
    OddOneOutSCAR,
    OddOneOutRelationNetworkSCAR,
    OddOneOutLstmSCAR,
)
from avr.model.scl_sal import OddOneOutSCLSAL

BATCH_SIZE = 4
NUM_PANELS_1 = 5
NUM_PANELS_2 = 6
NUM_CHANNELS = 1
IMAGE_SIZE = 80
EMBEDDING_SIZE = 128


@pytest.fixture
def context_1():
    return torch.rand(BATCH_SIZE, NUM_PANELS_1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def context_2():
    return torch.rand(BATCH_SIZE, NUM_PANELS_2, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


@pytest.mark.parametrize(
    "model_fn",
    [OddOneOutSCAR, OddOneOutRelationNetworkSCAR, OddOneOutLstmSCAR, OddOneOutSCLSAL],
)
def test(model_fn: Callable, context_1: torch.tensor, context_2: torch.tensor):
    model = model_fn(image_size=IMAGE_SIZE)

    y = model(context_1)
    assert y.shape == (BATCH_SIZE, NUM_PANELS_1, EMBEDDING_SIZE)

    y = model(context_2)
    assert y.shape == (BATCH_SIZE, NUM_PANELS_2, EMBEDDING_SIZE)
