#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.models import clone_model


from pathlib import Path
import json
import hashlib
import time
import pickle

from functions import run_stage, load_data


output_path = Path("snapshots")
output_path.mkdir(parents=True, exist_ok=True)

checkpoints_output_path = Path("checkpoints")
checkpoints_output_path.mkdir(parents=True, exist_ok=True)




# Stage 0: Model creation
stage0_hyperparams = {
        "hyperparams" : {
            "stage_name"       : "model_creation",
            "stage_type"       : "model_creation",
            "stage_seed"       : 12345,
            "stage_function"   : "model_create",
            "stage_parameters" : {
                "dataset"     : "mnist",
                "input_shape" : [None, 28, 28, 1],
                "categories"  : 10,
                "model_name"  : "lenet5_custom",
                },
            "previous_hash"   : None, # Does not need to be defined as is the first stage
            },
        "other"      : {
            'print_summary'  : False,
            'model_evaluate' : False,
            'model_save'     : True,
            },
        }

# Stage 1: Initial training
stage1_hyperparams = {
        "hyperparams" : {
            "stage_name"       : "initial_training",
            "stage_type"       : "training",
            "stage_seed"       : 12345,
            "stage_function"   : "model_train",
            "stage_parameters" : {
                "dataset"          : "mnist",
                "input_shape"      : [None, 28, 28, 1],
                "categories"       : 10,
                "epochs"           : 2,
                "batch_size"       : 1024,
                "learning_rate"    : 0.001, #Another option is learning_rate = 0.0001 * (batch_size/256),
                "validation_split" : 0.1,
                },
            "previous_hash"   : None, # Needs to be defined
            },
        "other"      : {
            'print_summary'  : False,
            'model_evaluate' : False,
            'model_save'     : True,
            },
        }

# Stage 2: BN folding
stage2_hyperparams = {
        "hyperparams" : {
            "stage_name"       : "bnf",
            "stage_type"       : "model_transformation",
            "stage_seed"       : 12345,
            "stage_function"   : "model_transform_bnf",
            "stage_parameters" : {
                "merge_activation" : True,
                },
            "previous_hash"   : None, # Needs to be defined
            },
        "other"      : {
            'print_summary'  : False,
            'model_evaluate' : False,
            'model_save'     : False, #TODO(Colo): This is set to false, as loading this model using tf.keras.models.load_model(...) fails!
            },
        }

# Stage 3: Post BN folding training
stage3_hyperparams = {
        "hyperparams" : {
            "stage_name"       : "pbnf_training",
            "stage_type"       : "training",
            "stage_seed"       : 12345,
            "stage_function"   : "model_train",
            "stage_parameters" : {
                "dataset"          : "mnist",
                "input_shape"      : [None, 28, 28, 1],
                "categories"       : 10,
                "epochs"           : 1,
                "batch_size"       : 32,
                "learning_rate"    : 0.0005, #Another option is learning_rate = 0.0001 * (batch_size/256),
                "validation_split" : 0.1,
                },
            "previous_hash"   : None, # Needs to be defined
            },
        "other"      : {
            'print_summary'  : False,
            'model_evaluate' : False,
            'model_save'     : True,
            },
        }

# Stage 4: Model quantization
stage4_hyperparams = {
        "hyperparams" : {
            "stage_name"       : "quantization",
            "stage_type"       : "model_transformation",
            "stage_seed"       : 12345,
            "stage_function"   : "model_quantize",
            "stage_parameters" : {
                "input_shape" : [None, 28, 28, 1],
                'kernel'     : [
                    {'type': "flexible", 'bits': 8, 'n_levels':5},
                    {'type': "flexible", 'bits': 8, 'n_levels':5},
                    {'type': "flexible", 'bits': 8, 'n_levels':5},
                    {'type': "flexible", 'bits': 8, 'n_levels':5},
                    {'type': "flexible", 'bits': 8, 'n_levels':5},
                    ],
                'bias'      : [
                    {'type': None},
                    {'type': None},
                    {'type': None},
                    {'type': None},
                    {'type': None},
                    ],
                'activations' : [
                    {'type': "uniform", 'bits': 8},
                    {'type': "uniform", 'bits': 8},
                    {'type': "uniform", 'bits': 8},
                    {'type': "uniform", 'bits': 8},
                    {'type': "uniform", 'bits': 8},
                    ],
                },
            },
            "previous_hash"   : None, # Needs to be defined
        "other"      : {
            'print_summary'  : False,
            'model_evaluate' : False,
            'model_save'     : False, #TODO(Colo): This is set to false, as loading quantize models using tf.keras.models.load_model(...) fails!
            },
        }

# Stage 5: Alpha initialization
stage5_hyperparams = {
        "hyperparams" : {
            "stage_name"       : "alpha_initialiation",
            "stage_type"       : "parameter_initialization",
            "stage_seed"       : 12345,
            "stage_function"   : "model_initialize_parameters",
            "stage_parameters" : {
                "dataset"     : "mnist",
                "input_shape" : [None, 28, 28, 1],
                "categories"  : 10,
                "type"        : "alpha",
                },
            "previous_hash"   : None, # Needs to be defined
            },
        "other"      : {
            'print_summary'  : False,
            'model_evaluate' : False,
            'model_save'     : False, #TODO(Colo): This is set to false, as loading quantize models using tf.keras.models.load_model(...) fails!
            },
        }

# Stage 6: QAT
stage6_hyperparams = {
        "hyperparams" : {
            "stage_name"       : "qat",
            "stage_type"       : "training",
            "stage_seed"       : 12345,
            "stage_function"   : "model_train",
            "stage_parameters" : {
                "dataset"          : "mnist",
                "input_shape"      : [None, 28, 28, 1],
                "categories"       : 10,
                "epochs"           : 2,
                "batch_size"       : 32,
                "learning_rate"    : 0.0001, #Another option is learning_rate = 0.0001 * (batch_size/256),
                "validation_split" : 0.1,
                },
            "previous_hash"   : None, # Needs to be defined
            },
        "other"      : {
            'print_summary'  : True,
            'model_evaluate' : True,
            'model_save'     : True,
            },
        }


stages_hyperparams = [
        stage0_hyperparams,
        stage1_hyperparams,
        stage2_hyperparams,
        stage3_hyperparams,
        stage4_hyperparams,
        stage5_hyperparams,
        stage6_hyperparams,
        ]




if __name__ == "__main__":
    model = None
    ref_model = None
    previous_hash = None

    for i, v in enumerate(stages_hyperparams):
        hyperparams = v['hyperparams']
        other       = v['other']

        # Opening message
        print(f"Stage {i}({hyperparams['stage_name']}): Start")

        # Set previous hash
        hyperparams['previous_hash'] = previous_hash

        # Generate file hash from hyperparams
        stage_str = json.dumps(hyperparams, sort_keys=True)
        stage_hash = hashlib.md5(stage_str.encode()).hexdigest()
        previous_hash = stage_hash

        # Define paths using Path
        model_path = checkpoints_output_path / f'{stage_hash}.keras'
        pickle_path = checkpoints_output_path / f'{stage_hash}.pkl'
        json_path = checkpoints_output_path / f'{stage_hash}.json'

        # Run stage
        save_flag = False
        start_time = time.time()
        if model_path.exists():
            print(f'Loading checkpoint from "{model_path}"')
            model = tf.keras.models.load_model(model_path)
        else:
            print(f'Checkpoint NOT FOUND (hash={stage_hash}).')
            save_flag = True
            model = run_stage(model, hyperparams, other, ref_model)

        # If this is not a quantization stage, then save this model as reference model
        if hyperparams['stage_type'] != 'model_quantize':
            ref_model = model
            # TODO(Colo): Here we should make a copy of the model, but the code below fails, so for now I’m leaving it like this to avoid raising an error.
            #ref_model = clone_model(model)       # clones architecture
            #ref_model.set_weights(model.get_weights())  # copies weights

        # Model summary
        if other.get('print_summary', False):
            model.summary(line_length=100)

        # Model evaluation
        if other.get('model_evaluate', False):
            dataset = hyperparams['stage_parameters']['dataset']
            data = load_data(dataset)
            print(f"Stage {i}({hyperparams['stage_name']}): Evaluation")
            loss, accuracy = model.evaluate(x=data['x_test'], y=data['y_test'])
            print(f"  → loss={loss:.4f}, acc={accuracy:.4f}")

        # Model save
        if other.get('model_save', False) and save_flag:
            model.save(model_path)
            with open(pickle_path, 'wb') as f:
                pickle.dump(hyperparams, f)
            with open(json_path, 'w') as f:
                json.dump(hyperparams, f, indent=2)

        # Closing message
        print(f"Stage {i}({hyperparams['stage_name']}): elapsed time = {time.time() - start_time:.2f} seconds")
        print(f"\n")




















    ## Stage 0: Model creation
    #model = models.Sequential()
    #model.add(layers.Conv2D(6, kernel_size=5, activation='relu', padding='same'))
    #model.add(layers.AveragePooling2D())
    #model.add(layers.Conv2D(16, kernel_size=5, activation='relu'))
    #model.add(layers.AveragePooling2D())
    #model.add(layers.Flatten())
    #model.add(layers.Dense(120, activation='relu'))
    #model.add(layers.Dense(84, activation='relu'))
    #model.add(layers.Dense(categories, activation='softmax'))

    #model.build(input_shape=input_shape)

    #print(f"#####################################################")
    #print(f"Summary")
    #model.summary(line_length=100)
    #print(f"#####################################################\n")



    ## Stage 1: Initial training
    #print(f"#####################################################")
    #print(f"Stage 1: Training")
    #model.compile(
    #    optimizer=Adam(learning_rate=stage0_hyperparams['stage_parameters']['learning_rate']),
    #    loss='categorical_crossentropy',
    #    metrics=['accuracy']
    #)
    #if stage0_hyperparams['stage_parameters']['epochs'] > 0:
    #    model.fit(x_train, y_train,
    #              batch_size=stage0_hyperparams['stage_parameters']['batch_size'],
    #              epochs=stage0_hyperparams['stage_parameters']['epochs'],
    #              validation_split=stage0_hyperparams['stage_parameters']['validation_split']
    #              )
    #print(f"Stage 1: Evaluation")
    #loss, accuracy = model.evaluate(x=x_test, y=y_test)
    #print(f"#####################################################\n")



    ## Stage 2: BN folding
    #print(f"#####################################################")
    #print(f"Stage 2: BN folding")
    #fmodel = apply_bn_folding(model, merge_activation=True)
    #fmodel.build(input_shape=input_shape)
    #fmodel.compile(
    #    optimizer=Adam(learning_rate=stage2_hyperparams['stage_parameters']['learning_rate']),
    #    loss="categorical_crossentropy",
    #    metrics=["accuracy"],
    #)
    #print(f"Folded model summary")
    #fmodel.summary(line_length=100)
    #print(f"Stage 2: Evaluation")
    #loss, accuracy = fmodel.evaluate(x=x_test, y=y_test)
    #print(f"#####################################################\n")



    ## Stage 3: Post BN folding training
    #print(f"#####################################################")
    #print(f"Stage 3: PBNF-Training")
    #if stage2_hyperparams['stage_parameters']['epochs'] > 0:
    #    fmodel.fit(x_train, y_train,
    #               batch_size=stage2_hyperparams['stage_parameters']['batch_size'],
    #               epochs=stage2_hyperparams['stage_parameters']['epochs'],
    #               validation_split=0.1
    #               )
    #print(f"Stage 3: Evaluation")
    #loss, accuracy = fmodel.evaluate(x=x_test, y=y_test)
    #print(f"#####################################################\n")



    ## Stage 4: Model quantization
    #print(f"#####################################################")
    #print(f"Stage 4: Model quantization")
    #print(f"QConfig")
    #for i, qc in enumerate(qconfig.values()):
    #    print(f"{i:04d}: {qc}")
    #print(f"------------------")
    #qmodel = apply_quantization(fmodel, qconfig)
    #qmodel.build(input_shape=input_shape)
    #qmodel.compile(
    #    loss="categorical_crossentropy",
    #    metrics=["accuracy"],
    #)
    #print(f"Stage 4: Evaluation")
    #loss, accuracy = qmodel.evaluate(x=x_test, y=y_test)
    #print(f"#####################################################\n")



    ## Stage 5: Initialize alpha values
    #print("#####################################################")
    #print("Stage 5: Alpha initialization (Weights and Activations)")
    #alpha_dict = compute_alpha_dict(fmodel, x_train)
    #for i, (layer_name, weights_dict) in enumerate(alpha_dict.items()):
    #    print(f"{i:04d}: {layer_name}")
    #    for key, alpha_value in weights_dict.items():
    #        print(f"    {key}: {alpha_value}")
    #qmodel = apply_alpha_dict(qmodel, alpha_dict)
    #print(f"Stage 5: Evaluation")
    #loss, accuracy = qmodel.evaluate(x=x_test, y=y_test)
    #print(f"#####################################################\n")


    #callback_tuples = [(CaptureWeightCallback(qlayer), qconfig[layer.name]) for layer, qlayer in zip(model.layers, qmodel.layers) if layer.name in qconfig]


    ## Stage 6: QAT
    #print(f"#####################################################")
    #print(f"Stage 6: QAT")
    #qmodel.compile(
    #    optimizer=Adam(learning_rate=stage5_hyperparams['stage_parameters']['learning_rate']),
    #    loss="categorical_crossentropy",
    #    metrics=["accuracy"],
    #)
    #if stage5_hyperparams['stage_parameters']['epochs'] > 0:
    #    hist = qmodel.fit(
    #        x_train,
    #        y_train,
    #        epochs=stage5_hyperparams['stage_parameters']['epochs'],
    #        batch_size=stage5_hyperparams['stage_parameters']['batch_size'],
    #        validation_split=0.1,
    #        callbacks=[callback for callback, _ in callback_tuples],
    #    )
    #print(f"#####################################################\n")

    #print(f"#####################################################")
    #print(f"Stage 6: Evaluation")
    #loss, accuracy = qmodel.evaluate(x=x_test, y=y_test)
    #print(f"#####################################################\n")


    #output_dict = {}
    #for callback, qconfig in callback_tuples:
    #    output_dict[callback.layer.name] = {}
    #    output_dict[callback.layer.name]["history"] = callback.get_history()
    #    output_dict[callback.layer.name]["qconfig"] = qconfig
    #with open(output_path / "output_dict.pkl", "wb") as f:
    #    pickle.dump(output_dict, f)



