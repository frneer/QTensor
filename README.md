# QTensor

**QTensor** is an extended repository for Quantization-Aware Training (QAT) with TensorFlow. Focusing on using different quantizers with the goal of benchmarking different ones.

## Developing

To get started with QTensor, clone the repository and run:

1. Clone the repo
```bash
git clone https://github.com/username/qtensor.git
cd qtensor
```

2. Run the docker container

```bash
./docker/run.sh
```

3. Run examples

  - Select a model under the [models](./src/examples/models) module, fo example `mlp`.
  - Select one of the qconfigs associated with the selected model (the available ones are the keys of the `qconfigs` dictionary e.g. for [mlp](./src/examples/models/mlp.py#L20)).
  - Select a dataset to use to evaluate the model from the [available datasets](./src/examples/datasets/).
  ```
  cd src/examples
  ./run.py --model mlp --qconfig qconfig --dataset mnist`
  ```
> Note: Run `./run.py --help` for extra options

## Contributing


## Working on the thesis article

To run and develop the document with latex, you can compile the file by running `compile.sh`.

### If you are working with VSC

If you use visual studio code as your IDE you can customize it so `ctrl+enter` compiles the latex document.

1. add this `task.json` file to the `.vscode` dir in the root of the project.

```
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "compile latex",
            "type": "shell",
            "command": "docker/compile.sh",
            "presentation": {
                "echo": false,
                "reveal": "silent",
                "focus": false,
                "panel": "shared",
                "showReuseMessage": false,
                "clear": true,
            }
        }
    ]
}
```
2. Add this shortcut to your `keybindings`

```
[{
    "key": "ctrl+enter",
    "command": "workbench.action.tasks.runTask",
    "args": ["compile latex"]
  }
]
```

## How to contribute

Please before pushing, install pre-commits and run them

```bash
pip install pre-commit==2.20.0
pre-commit install
pre-commit run --all-files --verbose --show-diff-on-failure
```
