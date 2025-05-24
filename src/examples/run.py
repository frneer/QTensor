#!/usr/bin/env python3

import argparse
import importlib
import pickle
from pathlib import Path

from data_collection.callbacks import CaptureWeightCallback
from tensorflow.keras.optimizers import Adam

from configs.qmodel import apply_quantization


def main(args):
    module_name = f"models.{args.model}"
    model_module = importlib.import_module(module_name)
    model = model_module.model

    print(f"Loaded model from {module_name}: {model}")

    qconfig = model_module.qconfigs[args.qconfig]

    print(f"Applying quantization configuration {args.qconfig}: {qconfig}")

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    dataset_module_name = f"datasets.{args.dataset}"
    dataset_module = importlib.import_module(dataset_module_name)
    train_dataset, val_dataset, test_dataset = dataset_module.generate_dataset(
        args.batch_size
    )

    # Pretrain the model to get a baseline
    train_dataset.batch(args.pre_training_batch_size)
    val_dataset.batch(args.pre_training_batch_size)
    model.fit(
        train_dataset,
        epochs=args.pre_training_epochs,
        batch_size=args.pre_training_batch_size,
        validation_data=val_dataset,
    )

    # Apply quantization
    qmodel = apply_quantization(model, qconfig)
    qmodel.compile(
        optimizer=Adam(learning_rate=0.001 / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(qmodel.summary())
    print(f"qweights: {[w.name for w in qmodel.layers[1].weights]}")
    # print(f"qactivations: {[w.name for w in qmodel.layers[1].weights]}")

    callback_tuples = [
        (CaptureWeightCallback(qlayer), qconfig[layer.name])
        for layer, qlayer in zip(model.layers, qmodel.layers)
        if layer.name in qconfig
    ]

    train_dataset.batch(args.batch_size)
    val_dataset.batch(args.batch_size)
    hist = qmodel.fit(
        train_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=val_dataset,
        callbacks=[callback for callback, _ in callback_tuples],
    )

    output_dict = {}
    output_dict["global"] = hist.history
    for callback, qconfig in callback_tuples:
        output_dict[callback.layer.name] = {}
        output_dict[callback.layer.name]["history"] = callback.get_history()
        output_dict[callback.layer.name]["qconfig"] = qconfig
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(
        args.output_path / f"{args.model}_{args.qconfig}.pkl", "wb"
    ) as f:
        pickle.dump(output_dict, f)

    test_dataset.batch(args.batch_size)
    qmodel.evaluate(test_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, help="Model to train", required=True
    )
    parser.add_argument(
        "--qconfig", type=str, help="Quantization configuration", required=True
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset to use", required=True
    )

    parser.add_argument(
        "--pre-training-epochs",
        type=int,
        default=2,
        help="Number of epochs for pre-training",
    )
    parser.add_argument(
        "--pre-training-batch-size",
        type=int,
        default=1024 // 4,
        help="Batch size for pre-training",
    )

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024 // 4,
        help="Batch size for training",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("snapshots"),
        help="Path to save snapshots",
    )

    args = parser.parse_args()
    main(args)
