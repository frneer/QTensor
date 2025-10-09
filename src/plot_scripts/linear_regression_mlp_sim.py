#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# --- 1. Generación de datos sintéticos ---
np.random.seed(42)
tf.random.set_seed(42)  # Fijar semilla para la red neuronal también

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Añadir el término de sesgo (bias) x0=1 a cada instancia para la Ecuación Normal
X_b = np.c_[np.ones((100, 1)), X]


# --- 2. Cálculo con la Ecuación Normal (Mínimos Cuadrados) ---

print("Calculando los parámetros con la Ecuación Normal...")
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Parámetros óptimos (Ecuación Normal):")
print(f"  - Intercepto (θ₀): {theta[0][0]:.4f}")
print(f"  - Pendiente  (θ₁): {theta[1][0]:.4f}")

# --- NUEVO: Cálculo del MSE para la Ecuación Normal ---
y_pred_normal_full = X_b.dot(theta)
mse_normal = np.mean((y_pred_normal_full - y) ** 2)
print(f"  - Error Cuadrático Medio (MSE): {mse_normal:.4f}")


# --- 3. Definición y Entrenamiento del Perceptrón Multicapa (MLP) ---

print("\nDefiniendo y entrenando el Perceptrón Multicapa (MLP)...")

model = Sequential([Dense(100, activation="relu", input_shape=(1,)), Dense(1)])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=300, verbose=0)
print("Entrenamiento del MLP finalizado.")

# --- NUEVO: Cálculo del MSE para el MLP ---
# Usamos model.evaluate() para obtener la pérdida en el conjunto de datos
mse_mlp = model.evaluate(X, y, verbose=0)
print(f"  - Error Cuadrático Medio (MSE): {mse_mlp:.4f}")


# --- 4. Visualización Comparativa ---

plt.figure(figsize=(12, 7))

# a) Dibujar los puntos de datos originales
plt.scatter(X, y, alpha=0.7, label="Datos Originales")

# b) Preparar puntos para las líneas de predicción
X_new = np.array([[0], [2]])

# c) Predecir y dibujar la línea de la Ecuación Normal
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict_normal = X_new_b.dot(theta)
plt.plot(
    X_new,
    y_predict_normal,
    "r-",
    linewidth=3,
    label=f"Ecuación Normal (MSE={mse_normal:.4f})",
)

# d) Predecir y dibujar la línea del MLP
y_predict_mlp = model.predict(X_new, verbose=0)
plt.plot(
    X_new, y_predict_mlp, "b--", linewidth=3, label=f"MLP (MSE={mse_mlp:.4f})"
)

# e) Títulos y etiquetas
# plt.title("Comparativa: Ecuación Normal vs. Perceptrón Multicapa", fontsize=16)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.axis([0, 2, 0, 15])
# plt.tight_layout()

# f) Guardar y mostrar el gráfico
plt.savefig(
    "../docs/pics/regression/comparativa_regresion_vs_mlp_con_mse.png", dpi=300
)
# plt.show()
