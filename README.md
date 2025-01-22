# QTensor

**QTensor** is an extended repository for Quantization-Aware Training (QAT) with TensorFlow. Focusing on using different quantizers with the goal of benchmarking different ones.

## Developing

To get started with QTensor, clone the repository and run:

```bash
git clone https://github.com/username/qtensor.git
cd qtensor
docker compose run ./docker/run.sh
```

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

Please before pushing, do a linter pass:

```bash
./src/linter.sh
```
