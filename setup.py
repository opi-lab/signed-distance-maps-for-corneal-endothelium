import sys
import itertools
import tensorflow
import numpy as np
import guttae

from tensorflow import keras
from guttae import deeptrack as dt
from tensorflow.keras import layers, backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

TEST_VARIABLES = {
    "generator_depth": [5],
    "generator_breadth": [16],
    "discriminator_depth": [4],
    "batch_size": [8],
    "min_data_size": [2048],
    "max_data_size": [2049],
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
    "mae_loss_weight": [0.8],
    "path_to_dataset": [r"C:/GU/GUTTA/datasets"],
}


def model_initializer(
    generator_depth,
    generator_breadth,
    discriminator_depth,
    mae_loss_weight=1,
    **kwargs,
):
    generator = guttae.generator(generator_breadth, generator_depth)
    discriminator = guttae.discriminator(discriminator_depth)

    generator.summary()
    discriminator.summary()

    return dt.models.cgan(
        generator=generator,
        discriminator=discriminator,
        discriminator_loss="mse",
        discriminator_optimizer=Adam(lr=0.0002, beta_1=0.5),
        assemble_loss=["mse", "mae"],
        assemble_optimizer=Adam(lr=0.0002, beta_1=0.5),
        assemble_loss_weights=[1, mae_loss_weight],
    )


# Populate models
_models = []
_generators = []


def append_model(**arguments):
    _models.append((arguments, lambda: model_initializer(**arguments)))


def append_generator(**arguments):

    _generators.append(
        (
            arguments,
            lambda: guttae.DataGenerator(
                **arguments,
            ),
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