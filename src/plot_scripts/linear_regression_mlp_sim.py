#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Semilla
np.random.seed(42)
tf.random.set_seed(42)

n_samples = 200
X = np.random.uniform(-2, 2, (n_samples, 1))

noise = np.sin(3 * X) + np.random.randn(n_samples, 1) * 0.5
y = 4 + 3 * X + 0.1 * X**2 + 0.1 * np.sin(3 * X) + noise

# Matriz con bias
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# Estimar parámetros (theta) con pseudo-inversa para robustez
theta = np.linalg.pinv(X_b).dot(y)

# Predicciones con el modelo lineal y MSE correcto
y_hat_linear = X_b.dot(theta)
mse_linear = np.mean((y - y_hat_linear) ** 2)

# Graficar los datos y la recta de regresión (ordenar por X para una línea limpia)
order = np.argsort(X[:, 0])
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label="Datos", color="gray")
plt.plot(
    X[order],
    y_hat_linear[order],
    color="blue",
    label=f"Regresión Lineal (MSE: {mse_linear:.2f})",
    linewidth=2,
)

# Estimar con MLP
model = Sequential([Dense(100, activation="relu", input_shape=(1,)), Dense(1)])

model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=100, verbose=0)
y_hat_mlp = model.predict(X)
mse_mlp = np.mean((y - y_hat_mlp) ** 2)

# Graficar resultados
plt.plot(
    X[order],
    y_hat_mlp[order],
    color="orange",
    label=f"MLP (MSE: {mse_mlp:.2f})",
    linewidth=2,
)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# Guardar figura
out = "linear_regression_mlp_simulation.png"
plt.savefig(out)
print("Guardado en:", out)
