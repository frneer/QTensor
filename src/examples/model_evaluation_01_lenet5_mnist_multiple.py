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
                'kernel'      : None, # Needs to be defined
                'bias'        : [
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

    for bits in range(1,11):
        print(f"Uniform, uniform arithmetic, bits={bits}\n")
        model = None
        ref_model = None
        previous_hash = None

        kernel = [
            {'type': "uniform", 'bits': bits},
            {'type': "uniform", 'bits': bits},
            {'type': "uniform", 'bits': bits},
            {'type': "uniform", 'bits': bits},
            {'type': "uniform", 'bits': bits},
            ]
        stages_hyperparams[4]['hyperparams']['stage_parameters']['kernel'] = kernel

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






    for bits in [2, 4, 6, 8, 10]:
        for n_levels in [2, 3, 4, 5, 6, 7, 8, 12, 16, 20]:
            print(f"Flexible, uniform arithmetic, bits_params={bits}, n_levels_param={n_levels}\n")
            if n_levels > 2**bits:
                print("SKIPPED\n")
                continue
            model = None
            ref_model = None
            previous_hash = None

            kernel = [
                {'type': "flexible", 'bits': bits, 'n_levels': n_levels},
                {'type': "flexible", 'bits': bits, 'n_levels': n_levels},
                {'type': "flexible", 'bits': bits, 'n_levels': n_levels},
                {'type': "flexible", 'bits': bits, 'n_levels': n_levels},
                {'type': "flexible", 'bits': bits, 'n_levels': n_levels},
                ]
            stages_hyperparams[4]['hyperparams']['stage_parameters']['kernel'] = kernel

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







    raise NotImplementedError(
            "Functions func_to_compute_bits(bits_params, sizes) and func_to_compute_bits(n_levels_param, bits, sizes) are not implemented"
            )
    for bits_param in range(1,11):
        print(f"Uniform, mixed arithmetic, bits_params={bits_params}\n")
        model = None
        ref_model = None
        previous_hash = None

        for i, v in enumerate(stages_hyperparams):
            hyperparams = v['hyperparams']
            other       = v['other']

            sizes = get_layers_size(model)
            bits = func_to_compute_bits(bits_params, sizes)
            kernel = [
                {'type': "uniform", 'bits': bits[0]},
                {'type': "uniform", 'bits': bits[1]},
                {'type': "uniform", 'bits': bits[2]},
                {'type': "uniform", 'bits': bits[3]},
                {'type': "uniform", 'bits': bits[4]},
                ]
            stages_hyperparams[4]['hyperparams']['stage_parameters']['kernel'] = kernel

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






    for bits_params in [2, 4, 6, 8, 10]:
        for n_levels_param in [2, 3, 4, 5, 6, 7, 8, 12, 16, 20]:
            print(f"Flexible, mixed arithmetic, bits_params={bits_params}, n_levels_param={n_levels_param}\n")
            model = None
            ref_model = None
            previous_hash = None

            for i, v in enumerate(stages_hyperparams):
                hyperparams = v['hyperparams']
                other       = v['other']

                sizes = get_layers_size(model)
                bits = func_to_compute_bits(bits_params, sizes)
                n_levels = func_to_compute_bits(n_levels_param, bits, sizes)
                kernel = [
                    {'type': "flexible", 'bits': bits[0], 'n_levels': n_levels[0]},
                    {'type': "flexible", 'bits': bits[1], 'n_levels': n_levels[1]},
                    {'type': "flexible", 'bits': bits[2], 'n_levels': n_levels[2]},
                    {'type': "flexible", 'bits': bits[3], 'n_levels': n_levels[3]},
                    {'type': "flexible", 'bits': bits[4], 'n_levels': n_levels[4]},
                    ]
                stages_hyperparams[4]['hyperparams']['stage_parameters']['kernel'] = kernel

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


























