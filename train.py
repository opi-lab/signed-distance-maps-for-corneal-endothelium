from os import error
import tensorflow.keras as keras
import tensorflow as tf
import sys
import getopt
import importlib
import numpy as np

import guttae

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Grab passed arguments
opts, args = getopt.getopt(sys.argv[2:], "i:e:p:n:r:")

script = sys.argv[1]
# Defaults
args = {"epochs": 100000, "patience": 100, "repet": 1}

username = guttae.get_user_name()

index = None
for opt, arg in opts:
    if opt == "-i":
        index = arg
    elif opt == "-e":
        args["epochs"] = int(arg)
    elif opt == "-p":
        args["patience"] = int(arg)
    elif opt == "-r":
        args["repet"] = int(arg)
    elif opt == "-n":
        username = arg

if index is None:
    error("option -i not set")
    sys.exit(0)

indices = guttae.parse_index(index)

user_models = importlib.import_module(script)

for index in indices:
    try:
        m_header_dict, model = user_models.get_model(index)
        d_header_dict, generator = user_models.get_generator(index)
    except KeyError as e:
        print(e)
        print(
            "The obove error likely occured because the index range was"
            " larger than the number of defined models. If so, you can"
            " safely disregard the error."
        )
        break

    headers = {**m_header_dict, **d_header_dict, **args}

    print("")

    print("=" * 50, "START", "=" * 50)

    print(
        "Running trial on model {0}, with patience of {1}".format(
            index, args["patience"]
        )
    )

    # Define callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args["patience"],
        restore_best_weights=True,
    )

    # Define data generators
    print("Grabbing validation data...")
    validation_data = guttae.get_validation_set(2)

    for i in range(args["repet"]):
        validation_data = guttae.get_validation_set(2)
        with generator:

            h = model.fit(
                generator,
                epochs=args["epochs"],
                callbacks=[early_stopping],
                validation_data=validation_data,
                validation_batch_size=2,
            )
        predictions = model.model.predict(validation_data[0][:2], batch_size=2)

        guttae.save_training_results(
            index=index,
            name=username,
            history=h.history,
            model=model.model,
            headers=headers,
            inputs=validation_data[0],
            predictions=predictions,
            targets=validation_data[1],
        )