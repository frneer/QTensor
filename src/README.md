

### Alpha parameter (clipping-value)
Is NOT an hyperparameter. Is tf-trainable,

Two modes of training (continuos range, pot (2^bits))
[constraint to paremeters of tf](https://www.tensorflow.org/api_docs/python/tf/keras/constraints/Constraint)

### Signed or Unsigned in quantizers
Must be differentiable sign and unsigned
Center or not.

#### Unsigned
range=(0, \alpha)
resolution=alpha/2^n

#### Signed
range=(-alpha, alpha)
resolution=2 * alpha / 2^n


# NUUQ

Evaluar la calidad de la convergencia, comparacion entre usar derivadas >=2.

## STE

## piecewise-STE

## Smooth functions
### cos/sen
### Stochastic differentiation

Quantizer
, grads

Register gradient can define the gradient for a method

# Uniform, vs NUUQ
# Comparar espacio que ocupa en memoria.
# PAra cada una de las implementaciones de estos cuantizer
# Calidad y velocidad de convergencia

# QAT en general se hace fine-tuning
# Desde 0 es mas complejo
# Primero punto flotante, que converja
# Segundo QAT

# Out of scope
## Loop over a matrix of the qHyperparameters.
### Random search
### Grid search
### Bayesian optimization
### Reinforcement learning
### Heuristic search

# NuuQuantizer(_QuantizeHelper, Quantizer):


## Archs

### VGG

### Inception (2 or 3 versions)

### ResNet

### DenseNet

### MobileNet <--

## Datasets

### Toy (CIFAR10, CIFAR 100)

### nth: (ImageNet)

# Meta

model (N layers)
qConfig_n (one for each layer) (mapa de config a layer)
config:
  - qType (Uniform, FlexA, FlexB, FlexC) (flex different grads)
  - nbits
  - FlexOnly: c_levels
  - Signed, Unsigned

q_model = QuantizeModel(model, Qconfig); See if qregistry is possible.
Each step of the hParameter optimziation, will change the values of our configs, how can we handle that?

## How to test

Quantizer_Ops

inputs -> quantize ->


Integration test
One layer without activation, inputs a mano
menos 1 y ves que a la salida tiene que salir quantizada.
y = W*x+b, W =[...] b = 0_vec x = [0000 10000]
y / x => W1_Q

Change the x, all w are expected.
