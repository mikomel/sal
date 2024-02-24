from functools import partial
from typing import Callable

import pytest
import torch

from avr.model.copinet import CoPINet
from avr.model.lstm import CnnLstm
from avr.model.relbase import RelBase
from avr.model.scar import SCAR
from avr.model.scl import SCL
from avr.model.scl_sal import SCLSAL
from avr.model.sran import SRAN
from avr.model.wild_relation_network import WildRelationNetwork

BATCH_SIZE = 4
NUM_CONTEXT_PANELS = 8
NUM_ANSWER_PANELS = 5
NUM_CHANNELS = 1
IMAGE_SIZE = 80
EMBEDDING_SIZE = 128


@pytest.fixture
def context():
    return torch.rand(
        BATCH_SIZE, NUM_CONTEXT_PANELS, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE
    )


@pytest.fixture
def answers():
    return torch.rand(
        BATCH_SIZE, NUM_ANSWER_PANELS, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE
    )


@pytest.mark.parametrize(
    "model_fn",
    [
        partial(WildRelationNetwork, image_size=IMAGE_SIZE),
        partial(CnnLstm, image_size=IMAGE_SIZE),
        partial(SCAR, image_size=IMAGE_SIZE),
        partial(SCLSAL, image_size=IMAGE_SIZE),
        partial(SCL, image_size=IMAGE_SIZE),
        CoPINet,
        SRAN,
        RelBase,
    ],
)
def test(model_fn: Callable, context: torch.tensor, answers: torch.tensor):
    model = model_fn()
    y = model(context, answers)
    assert y.shape == (BATCH_SIZE, NUM_ANSWER_PANELS, EMBEDDING_SIZE)
