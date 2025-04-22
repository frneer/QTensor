#!/usr/bin/env python3

#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

import argparse
#import importlib
import pickle
import hashlib
import os
import json

from pathlib import Path

from configs.qmodel import apply_quantization
from data_collection.callbacks import CaptureWeightCallback
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image_dataset_from_directory

from tensorflow.keras import backend as K
import gc

import numpy as np
import time


from quantizers.flex_quantizer import FlexQuantizer
from quantizers.uniform_quantizer import UniformQuantizer

from functions import apply_bn_folding, compute_alpha_dict, apply_alpha_dict, print_model_weighs, report_memory_usage


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


qconfig_uniform = {
        "block1_conv1": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block1_conv2": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block2_conv1": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block2_conv2": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block3_conv1": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block3_conv2": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block3_conv3": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block4_conv1": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block4_conv2": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block4_conv3": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block5_conv1": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block5_conv2": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block5_conv3": { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "fc1"         : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "fc2"         : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        #"predictions" : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)} },
        }
qconfig_flex = {
        "block1_conv1": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block1_conv2": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block2_conv1": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block2_conv2": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block3_conv1": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block3_conv2": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block3_conv3": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block4_conv1": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block4_conv2": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block4_conv3": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block5_conv1": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block5_conv2": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "block5_conv3": { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "fc1"         : { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        "fc2"         : { "weights": {"kernel": FlexQuantizer(bits=8, n_levels=4, signed=True)}, "activations": {"activation": UniformQuantizer(bits=8 , signed=False)}, },
        #"predictions" : { "weights": {"kernel": UniformQuantizer(bits=8, signed=True)} },
        }
qconfig = qconfig_flex

output_path = Path("snapshots")

# Pre-training parameters
pre_training_epochs = 3
pre_training_batch_size = 64
#pre_training_learning_rate = 0.001 * (pre_training_batch_size/256)
pre_training_learning_rate = 0.001

# BN parameters
post_bnf_epochs = 0
post_bnf_batch_size = 32
#post_bnf_learning_rate = 0.0005 * (post_bnf_batch_size/256)
post_bnf_learning_rate = 0.0005

# QAT parameters
qat_epochs = 10
qat_batch_size = 32
#qat_learning_rate = 0.0001 * (qat_batch_size/256)
qat_learning_rate = 0.0001


if __name__ == "__main__":

    # Load Imagenet-1k dataset
    #(x_train, y_train), (x_test, y_test) = cifar100.load_data()
    #input_shape = (None,) + x_train.shape[1:]
    #image_shape = input_shape[1:]
    #categories = 100

    #x_train = x_train.astype("float32") / 255.0
    #x_test = x_test.astype("float32") / 255.0

    #y_train = to_categorical(y_train, 100)
    #y_test = to_categorical(y_test, 100)

    #image_dir = './datasets/download/imagenet-1k'
    #img_height, img_width = 224, 224
    #batch_size = 256

    #train_ds = image_dataset_from_directory(
    #    image_dir + '/train',
    #    validation_split=0.2,
    #    subset='training',
    #    seed=123,
    #    image_size=(img_height, img_width),
    #    batch_size=batch_size)

    #val_ds = image_dataset_from_directory(
    #    image_dir + '/train',
    #    validation_split=0.2,
    #    subset='validation',
    #    seed=123,
    #    image_size=(img_height, img_width),
    #    batch_size=batch_size)

    #test_ds = image_dataset_from_directory(
    #    image_dir + '/test',
    #    image_size=(img_height, img_width),
    #    batch_size=batch_size,
    #    shuffle=False)

    #train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    #test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    image_dir = './datasets/download/imagenet-1k'
    img_height, img_width = 224, 224
    image_shape = (img_height, img_width, 3)
    input_shape = (None,) + image_shape
    categories = 1000

    dataset = image_dataset_from_directory(
        image_dir,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=pre_training_batch_size)

    total_count = dataset.cardinality().numpy()
    train_size = int(0.8 * total_count)
    val_size = int(0.1 * total_count)
    test_size = total_count - train_size - val_size

    train_ds = dataset.take(train_size)
    val_test_ds = dataset.skip(train_size)
    val_ds = val_test_ds.take(val_size)
    test_ds = val_test_ds.skip(val_size)

    train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #def conv_block(x, filters, convs):
    #    for _ in range(convs):
    #        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    #    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    #    return x

    #inputs = tf.keras.Input(shape=image_shape)
    #x = conv_block(inputs, 64, 2)
    #x = conv_block(x, 128, 2)
    #x = conv_block(x, 256, 3)
    #x = conv_block(x, 512, 3)
    ##x = conv_block(x, 512, 3)
    #x = tf.keras.layers.Flatten()(x)
    #x = tf.keras.layers.Dense(512, activation='relu')(x)
    #x = tf.keras.layers.Dropout(0.5)(x)
    #outputs = tf.keras.layers.Dense(categories, activation='softmax')(x)
    #model = tf.keras.Model(inputs, outputs)

    #base_model = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
    #base_model.trainable = False

    #inputs = tf.keras.Input(shape=image_shape)
    #x = base_model(inputs, training=False)
    #x = layers.Flatten()(x)
    #x = layers.Dense(512, activation='relu')(x)
    #x = layers.Dropout(0.5)(x)
    #outputs = layers.Dense(categories, activation='softmax')(x)

    #model = tf.keras.Model(inputs, outputs)

    #pre_training_config = {
    #        "model"                     : "vgg16;top=yes;weights=imagenet",
    #        "dataset"                   : "imagenet-1k",
    #        "pre_training_epochs"       : pre_training_epochs,
    #        "pre_training_batch_size"   : pre_training_batch_size,
    #        "pre_training_learning_rate": pre_training_learning_rate,
    #        }
    #models_dir = './models/pretrained_models/'

    #model = VGG16(weights='imagenet')

    #model.build(input_shape=input_shape)


    pre_training_config = {
        "model": "vgg16;top=yes;weights=imagenet",
        "dataset": "imagenet-1k",
        "pre_training_epochs": pre_training_epochs,
        "pre_training_batch_size": pre_training_batch_size,
        "pre_training_learning_rate": pre_training_learning_rate,
    }

    #models_dir = './models/pretrained_models/'
    #os.makedirs(models_dir, exist_ok=True)

    #config_str = json.dumps(pre_training_config, sort_keys=True)
    #config_hash = hashlib.md5(config_str.encode()).hexdigest()
    #model_path = os.path.join(models_dir, f'{config_hash}.keras')
    #pickle_path = os.path.join(models_dir, f'{config_hash}.pkl')
    #json_path = os.path.join(models_dir, f'{config_hash}.json')

    # Define directory using Path
    models_dir = Path('./models/pretrained_models/')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate file hash from config
    config_str = json.dumps(pre_training_config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    # Define paths using Path
    model_path = models_dir / f'{config_hash}.keras'
    pickle_path = models_dir / f'{config_hash}.pkl'
    json_path = models_dir / f'{config_hash}.json'

    print("#####################################################")
    start_time = time.time()
    if model_path.exists():
        print(f'Loading pre-trained model from "{model_path}" (hash={config_hash}).')
        model = tf.keras.models.load_model(model_path)
    else:
        print(f'Pre-trained model NOT FOUND, building a new one (hash={config_hash}).')
        model = VGG16(weights='imagenet')
        model.build(input_shape=input_shape)
    print(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    print("#####################################################\n")

    print(f"#####################################################")
    print(f"Model summary")
    model.summary(line_length=100)
    print(f"#####################################################\n")

    print(f"#####################################################")
    print(f"Pre-Training")
    report_memory_usage()
    if not model_path.exists():
        print(f'Pre-training NEEDED (hash={config_hash}).')
        model.compile(
            optimizer=Adam(learning_rate=pre_training_learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top-1'),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top-5'),
                ]
        )

        if pre_training_epochs > 0:
            model.fit(train_ds,
                      batch_size=pre_training_batch_size,
                      epochs=pre_training_epochs,
                      #validation_split=0.1,
                      validation_data=val_ds,
                      )
        model.save(model_path)

        with open(pickle_path, 'wb') as f:
            pickle.dump(pre_training_config, f)

        with open(json_path, 'w') as f:
            json.dump(pre_training_config, f, indent=2)

    else:
        print(f'Pre-training NOT NEEDED, model loaded from "{model_path}" (hash={config_hash}).')
    report_memory_usage()
    print(f"#####################################################\n")

    print(f"#####################################################")
    print(f"POST FIT Evaluation")
    report_memory_usage()
    #scores = model.evaluate(test_ds)
    report_memory_usage()
    print(f"#####################################################\n")

    #fmodel = apply_bn_folding(model, merge_activation=True)
    #fmodel.build(input_shape=input_shape)
    #fmodel.compile(
    #    optimizer=Adam(learning_rate=post_bnf_learning_rate),
    #    loss="sparse_categorical_crossentropy",
    #    metrics=[
    #        'accuracy',
    #        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top-1'),
    #        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top-5'),
    #        ]
    #)

    ## Free memory
    #del model
    #K.clear_session()
    #gc.collect()

    #print(f"#####################################################")
    #print(f"Post-BNF-Training")
    #if post_bnf_epochs > 0:
    #    fmodel.fit(train_ds,
    #               batch_size=post_bnf_batch_size,
    #               epochs=post_bnf_epochs,
    #               #validation_split=0.1,
    #               validation_data=val_ds,
    #               )
    #print(f"#####################################################\n")

    #print(f"#####################################################")
    #print(f"Folded model summary")
    #fmodel.summary(line_length=100)
    #print(f"#####################################################\n")

    #print(f"#####################################################")
    #print(f"POST BN FOLDING Evaluation")
    #report_memory_usage()
    ##scores = fmodel.evaluate(test_ds)
    #report_memory_usage()
    #print(f"#####################################################\n")
    fmodel = model #TODO(Colo): Problemas de memoria, lo quito por ahora, no es necesario para VGG16.

    # Apply quantization
    print(f"#####################################################")
    print(f"Apply quantization")
    report_memory_usage()
    for i, qc in enumerate(qconfig.values()):
        print(f"{i:04d}: {qc}")
    qmodel = apply_quantization(fmodel, qconfig)
    qmodel.build(input_shape=input_shape)
    qmodel.compile(
        optimizer=Adam(learning_rate=qat_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            'accuracy',
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, name='top-1'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top-5'),
            ]
    )
    report_memory_usage()
    print(f"#####################################################\n")

    # Initialize alpha values
    print("#####################################################")
    print("Alpha Dictionary (Weights and Activations)")
    report_memory_usage()
    with tf.device('/CPU:0'):
        alpha_dict = compute_alpha_dict(fmodel, train_ds.take(3), batch_size=1)
        for i, (layer_name, weights_dict) in enumerate(alpha_dict.items()):
            print(f"{i:04d}: {layer_name}")
            for key, alpha_value in weights_dict.items():
                print(f"    {key}: {alpha_value}")
        print("---------------------------")
        print("Apply alpha values")
        qmodel = apply_alpha_dict(qmodel, alpha_dict)
    report_memory_usage()
    print("#####################################################\n")

    # TODO(Colo): this information can be save now, before deleting the fmodel
    #layername_qconfig_vector = [(layer.name, qconfig[layer.name]) for layer in fmodel.layers if layer.name in qconfig]
    ## Free memory
    #del fmodel
    #K.clear_session()
    #gc.collect()

    #print("#####################################################")
    #print("DEBUG: Model weights")
    #print_model_weighs(model)
    #print("--------------------------------")
    #print_model_weighs(fmodel)
    #print("--------------------------------")
    #print_model_weighs(qmodel)
    #print("#####################################################\n")

    print(f"#####################################################")
    print(f"POST QUANTIZATION Evaluation")
    report_memory_usage()
    scores = qmodel.evaluate(test_ds)
    report_memory_usage()
    print(f"#####################################################\n")

    callback_tuples = [(CaptureWeightCallback(qlayer), qconfig[layer.name]) for layer, qlayer in zip(fmodel.layers, qmodel.layers) if layer.name in qconfig]
    # TODO(Colo): replace this code, to use the `layername_qconfig_vector`, as the fmodel/model no longer exists.

    # Free memory
    del fmodel
    del alpha_dict
    K.clear_session()
    gc.collect()

    print(f"#####################################################")
    print(f"QAT")
    report_memory_usage()
    if qat_epochs > 0:
        hist = qmodel.fit(train_ds,
                          epochs=qat_epochs,
                          batch_size=qat_batch_size,
                          #validation_split=0.1,
                          validation_data=val_ds,
                          callbacks=[callback for callback, _ in callback_tuples],
                          )
    report_memory_usage()
    print(f"#####################################################\n")

    print(f"#####################################################")
    print(f"POST QAT Evaluation")
    report_memory_usage()
    scores = qmodel.evaluate(test_ds)
    report_memory_usage()
    print(f"#####################################################\n")

    output_dict = {}
    for callback, qconfig in callback_tuples:
        output_dict[callback.layer.name] = {}
        output_dict[callback.layer.name]["history"] = callback.get_history()
        output_dict[callback.layer.name]["qconfig"] = qconfig
    with open(output_path / "output_dict.pkl", "wb") as f:
        pickle.dump(output_dict, f)



