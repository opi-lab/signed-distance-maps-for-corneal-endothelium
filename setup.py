import sys
import itertools
import tensorflow as tf
import numpy as np
import guttae

from tensorflow import keras
from guttae import deeptrack as dt
from tensorflow.keras import layers, backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

TEST_VARIABLES = {
    "depth": [6],
    "breadth": [16],
    "batch_size": [8],
    "min_data_size": [2048],
    "max_data_size": [2049],
    "path_to_dataset": ["datasets"],
    "augmentation_dict": [
        {
            "FlipLR": {},
            "FlipUD": {},
            "FlipDiagonal": {},
            "Affine": {
                "rotate": "lambda: np.random.rand() * 2 * np.pi",
            },
        }
    ],
}


def model_initializer(
    depth,
    breadth,
    **kwargs,
):
    model = guttae.models.get_model(breadth, depth)

    model.compile(
        loss="mae",
        metrics=["mae"],
        optimizer=Adam(learning_rate=0.0001),
    )
    return model


# Populate models
_models = []
_generators = []


def append_model(**arguments):
    _models.append((arguments, lambda: model_initializer(**arguments)))


def append_generator(**arguments):

    _generators.append(
        (
            arguments,
            lambda: guttae.DataGenerator(**arguments),
        )
    )


for prod in itertools.product(*TEST_VARIABLES.values()):

    arguments = dict(zip(TEST_VARIABLES.keys(), prod))
    append_model(**arguments)
    append_generator(**arguments)


def get_model(i):
    try:
        i = int(i)
    except ValueError:
        pass

    args, model = _models[i]
    return args, model()


def get_generator(i):
    try:
        i = int(i)
    except ValueError:
        pass

    args, generator = _generators[i]
    return args, generator()
