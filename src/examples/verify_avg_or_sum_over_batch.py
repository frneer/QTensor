#!/usr/bin/env python3

import tensorflow as tf

# Dummy model & data
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(5,))
])
loss_fn = tf.keras.losses.MeanSquaredError(
        #reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
        )
print(loss_fn.reduction)
print(dir(tf.keras.losses.Reduction))


for batch_size in (1, 10, 100, 1000, 10000,):

    x = tf.random.normal((batch_size, 5))
    y = tf.random.normal((batch_size, 10))

    with tf.GradientTape() as tape:
        preds = model(x)
        loss = loss_fn(y, preds)

    grads = tape.gradient(loss, model.trainable_variables)

    # Print gradients
    for i, g in enumerate(grads):
        print(f"Grad {i} mean: {tf.reduce_mean(g).numpy()}")


