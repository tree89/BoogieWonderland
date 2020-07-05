# PYTHONPATH=. python main.py --config=configs/classification_test.json

import os
import json
import argparse

import tensorflow as tf

from core.backbones import *
from core.utils.data_loader import load_data
from core.utils.modeler import Modeler
from core.utils.configurator import Configurator

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, help="Path of configs file")

args = vars(ap.parse_args())


def main():
    config_path = args["config"]
    configs = Configurator(config_path)
    configs.load_configs()
    configs.check_configs()
    modeler = Modeler(configs)
    train_ds, val_ds = load_data(configs.dataset, configs.batch_size)
    configs.load_model = False

    if configs.load_model:
        model = tf.keras.models.load_model(configs.load_model)
        modeler.modeler_model = model
        print(modeler.modeler_model.summary())
    else:
        modeler.make_model()

    if configs.task_type == "classification":
        modeler.compile_model()
        modeler.train_model(train_ds, batch_size=configs.batch_size, epochs=configs.epoch)
    else:
        pass


if __name__ == "__main__":
    main()
