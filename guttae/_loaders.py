from . import deeptrack as dt
from typing import List
import numpy as np
import itertools
import random
import glob
import os


def Augmentation(
    image: dt.Feature,
    augmentation_list=None,
    default_value=lambda x: x,
    **kwargs
):
    augmented_image = image
    for augmentation in augmentation_list:
        print("With ", augmentation)

        args = augmentation_list[augmentation].copy()
        for key, val in args.items():
            if isinstance(val, str):
                args[key] = eval(val)

        augmented_image += getattr(dt, augmentation, default_value)(
            **args, **kwargs
        )

    return augmented_image


def DataLoader(path_to_dataset=None, augmentation=None, **kwargs):
    # Define path to the dataset
    DATASET_PATH = os.path.join(".", path_to_dataset)

    TRAINING_PATH = os.path.join(DATASET_PATH, "training")
    VALIDATION_PATH = os.path.join(DATASET_PATH, "validation")

    training_set = glob.glob(os.path.join(TRAINING_PATH, "*cor*"))
    validation_set = glob.glob(os.path.join(VALIDATION_PATH, "*cor*"))

    print("Loading images from: \t", DATASET_PATH)

    # random.seed(1)
    random.shuffle(training_set)
    random.shuffle(validation_set)

    print("Training on {0} images".format(len(training_set)))
    print("Validating on {0} images".format(len(validation_set)))

    training_iterator = itertools.cycle(training_set)
    validation_iterator = itertools.cycle(validation_set)

    def get_filename(validation):
        if validation:
            return next(validation_iterator)
        else:
            return next(training_iterator)

    root = dt.DummyFeature(filename=get_filename)

    cor = (
        root
        + dt.LoadImage(path=lambda filename: filename, **root.properties)
        + dt.Lambda(lambda: lambda i: i * 1.0)
    ) + dt.NormalizeMinMax(min=-1, max=1)

    cell = (
        root
        + dt.LoadImage(
            path=lambda filename: filename.replace("cor", "cell"),
            **root.properties,
        )
        + dt.Lambda(lambda: lambda image: image / 255)
    )

    guttae = (
        root
        + dt.LoadImage(
            path=lambda filename: filename.replace("cor", "guttae"),
            **root.properties,
        )
        + dt.Lambda(lambda: lambda image: -1.0 * image / 255)
    )

    seg = dt.Combine([cell, guttae]) + dt.Merge(
        lambda: lambda image: np.sum(image, axis=0)
    )

    dataset = dt.Combine([cor, seg])

    if augmentation:
        augmented_dataset = Augmentation(
            dataset, augmentation_list=augmentation
        )
    else:
        augmented_dataset = dataset

    return (
        dt.ConditionalSetFeature(
            on_true=dataset,
            on_false=augmented_dataset,
            condition="is_validation",
            is_validation=lambda validation: validation,
        )
        + dt.AsType("float64")
    )


def batch_function(image: dt.Image or List[dt.Image]) -> dt.Image:
    return image[0]


def label_function(image: dt.Image or List[dt.Image]) -> dt.Image:
    return image[1]


_VALIDATION_SET_SIZE = 16

conf = {}


def DataGenerator(
    min_data_size=1000, max_data_size=2000, augmentation_dict={}, **kwargs
):

    feature = DataLoader(augmentation=augmentation_dict, **kwargs)

    conf["feature"] = feature

    args = {
        "feature": feature,
        "label_function": label_function,
        "batch_function": batch_function,
        "min_data_size": min_data_size,
        "max_data_size": max_data_size,
        **kwargs,
    }
    return dt.utils.safe_call(dt.generators.ContinuousGenerator, **args)


def get_validation_set(size=_VALIDATION_SET_SIZE):
    data_loader = conf["feature"]

    data = []
    labels = []
    for _ in range(size):
        data_loader.update(validation=True, is_validation=True)
        output = data_loader.resolve()
        data.append(batch_function(output))
        labels.append(label_function(output))

    return data, labels
