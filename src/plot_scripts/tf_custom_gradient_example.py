#!/usr/bin/env python3

import tensorflow as tf


# Función con gradiente personalizado
@tf.custom_gradient
def custom_mul(a, b):
    y = a * b

    def grad(dy):
        da = 2 * b * dy
        db = 2 * a * dy
        return da, db

    return y, grad


# Valores de entrada
a = tf.constant(3.0)
b = tf.constant(4.0)


# Función normal: usa la autodiferenciación de TensorFlow
def mul(a, b):
    return a * b


# Forward normal
with tf.GradientTape(persistent=True) as tape1:
    tape1.watch([a, b])
    y_auto = mul(a, b)  # f(a,b) = 3*4 = 12

da_auto = tape1.gradient(y_auto, a)  # debería ser b = 4
db_auto = tape1.gradient(y_auto, b)  # debería ser a = 3

# Forward con gradiente custom
with tf.GradientTape(persistent=True) as tape2:
    tape2.watch([a, b])
    y_custom = custom_mul(a, b)  # también 12 en forward

da_custom = tape2.gradient(y_custom, a)  # devuelve 2*b = 8
db_custom = tape2.gradient(y_custom, b)  # devuelve 2*a = 6

print("Forward pass (f(a,b) = a*b):", y_auto.numpy())
print("\nGradientes automáticos:")
print("df/da =", da_auto.numpy())  # 4
print("df/db =", db_auto.numpy())  # 3

print("\nGradientes con custom_gradient:")
print("df/da =", da_custom.numpy())  # 8
print("df/db =", db_custom.numpy())  # 6)
