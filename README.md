# QTensor

**QTensor** is an extended repository for Quantization-Aware Training (QAT) with TensorFlow. Focusing on using different quantizers with the goal of benchmarking different ones.

## Installation

To get started with QTensor, clone the repository and install the required dependencies:

```bash
git clone https://github.com/username/qtensor.git
cd qtensor
docker compose run ./docker/run.sh
```

## How to contribute

Please before pushing, do a linter pass:

```bash
./src/linter.sh
```


## Improvements:

Alpha in the quantizers it's been used as an analog value.
Consider using a quantized alpha for the forward pass, but the alpha is still there as the trainable param of the model.
This would need to alpha_q(alpha) do ste.
