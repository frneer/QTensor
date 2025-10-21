#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential


# --- 1. Definición de funciónes (sin cambios) ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(1 / len(y_true)) * np.sum(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )


# --- 2. Generación de datos sintéticos (sin cambios) ---
np.random.seed(42)
tf.random.set_seed(42)

mean_0 = [2, 3]
cov_0 = [[1, 0.5], [0.5, 1]]
X0 = np.random.multivariate_normal(mean_0, cov_0, 100)
y0 = np.zeros(100)

mean_1 = [6, 7]
cov_1 = [[1.5, -0.5], [-0.5, 1.5]]
X1 = np.random.multivariate_normal(mean_1, cov_1, 100)
y1 = np.ones(100)

X = np.vstack((X0, X1))
y = np.hstack((y0, y1))
X_b = np.c_[np.ones((X.shape[0], 1)), X]

# --- 3. Entrenamiento del modelo con Descenso por Gradiente (sin cambios) ---
print("Iniciando entrenamiento del modelo de Regresión Logística...")
learning_rate = 0.1
n_iterations = 2000
m = len(y)
theta = np.random.randn(3, 1)
y_col = y.reshape(-1, 1)

for iteration in range(n_iterations):
    z = X_b.dot(theta)
    predictions = sigmoid(z)
    gradients = (1 / m) * X_b.T.dot(predictions - y_col)
    theta = theta - learning_rate * gradients
    if iteration == n_iterations - 1:
        loss_manual = cross_entropy_loss(y_col, predictions)
        accuracy_manual = np.mean((predictions > 0.5).astype(int) == y_col)

print("Entrenamiento finalizado.")
print("Parámetros finales (theta):", theta.flatten())
print(f"  - Costo (Cross-Entropy): {loss_manual:.4f}")
print(f"  - Precisión (Accuracy): {accuracy_manual:.4f}")

# --- 4. Definición y Entrenamiento del Perceptrón Multicapa (MLP) (sin cambios) ---
print("\nDefiniendo y entrenando el Perceptrón Multicapa (MLP)...")

model = Sequential(
    [
        Dense(30, activation="relu", input_shape=(2,)),
        Dense(1),
        Activation("sigmoid"),
    ]
)

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

history = model.fit(X, y, epochs=150, verbose=0)
print("Entrenamiento del MLP finalizado.")

# --- 5. Cálculo de Métricas Finales para el MLP (sin cambios) ---
loss_mlp, accuracy_mlp = model.evaluate(X, y, verbose=0)
print(f"  - Costo (Cross-Entropy): {loss_mlp:.4f}")
print(f"  - Precisión (Accuracy): {accuracy_mlp:.4f}")

# --- 6. Preparación de datos para la visualización (sin cambios) ---
z_manual = X_b.dot(theta).flatten()

intermediate_layer_model = tf.keras.Model(
    inputs=model.input, outputs=model.layers[-2].output
)
z_mlp = intermediate_layer_model.predict(X, verbose=0).flatten()

# --- 7. VISUALIZACIÓN MODIFICADA: DOS GRÁFICOS SEPARADOS ---


# Función auxiliar de ploteo (se elimina el argumento 'title')
def plot_sigmoid_data(ax, z_data, y_data):
    jitter_strength = 0.05
    y_jittered = y_data + np.random.uniform(
        -jitter_strength, jitter_strength, size=len(y_data)
    )

    correct_0_mask = (y_data == 0) & (z_data <= 0)
    misclassified_0_mask = (y_data == 0) & (z_data > 0)
    correct_1_mask = (y_data == 1) & (z_data > 0)
    misclassified_1_mask = (y_data == 1) & (z_data <= 0)

    z_range = np.linspace(z_data.min() - 2, z_data.max() + 2, 500)
    ax.plot(
        z_range,
        sigmoid(z_range),
        color="purple",
        linewidth=2.5,
        label="Función Sigmoidea",
        zorder=5,
    )

    ax.scatter(
        z_data[correct_0_mask],
        y_jittered[correct_0_mask],
        color="blue",
        alpha=0.7,
        label="Clase 0 (Correcto)",
        s=60,
        zorder=3,
    )
    ax.scatter(
        z_data[correct_1_mask],
        y_jittered[correct_1_mask],
        color="red",
        alpha=0.7,
        label="Clase 1 (Correcto)",
        s=60,
        zorder=3,
    )
    ax.scatter(
        z_data[misclassified_0_mask],
        y_jittered[misclassified_0_mask],
        color="blue",
        alpha=0.9,
        s=80,
        edgecolors="red",
        linewidth=2,
        label="Clase 0 (Mal clasificado)",
        zorder=4,
    )
    ax.scatter(
        z_data[misclassified_1_mask],
        y_jittered[misclassified_1_mask],
        color="red",
        alpha=0.9,
        s=80,
        edgecolors="blue",
        linewidth=2,
        label="Clase 1 (Mal clasificado)",
        zorder=4,
    )

    ax.axhline(y=0.5, color="gray", linestyle="--")
    ax.axvline(x=0, color="black", linestyle=":", linewidth=2)
    # Se elimina ax.set_title() para que no tenga título
    ax.set_xlabel("z (Entrada a la Sigmoide)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", fontsize=10)


# --- Gráfico 1: Regresión Logística Manual ---
print("\nGenerando plot_logistica.png...")
fig1, ax1 = plt.subplots(figsize=(10, 8))
plot_sigmoid_data(ax1, z_manual, y)
ax1.set_ylabel("Probabilidad", fontsize=12)
ax1.set_yticks([0, 0.5, 1], ["Clase 0", "0.5", "Clase 1"])
# plt.tight_layout()
plt.savefig("../docs/pics/regression/plot_logistica.png", dpi=300)
plt.close(fig1)  # Cierra la figura para liberar memoria

# --- Gráfico 2: Perceptrón Multicapa (Keras) ---
print("Generando plot_mlp.png...")
fig2, ax2 = plt.subplots(figsize=(10, 8))
plot_sigmoid_data(ax2, z_mlp, y)
ax2.set_ylabel("Probabilidad", fontsize=12)
ax2.set_yticks([0, 0.5, 1], ["Clase 0", "0.5", "Clase 1"])
# plt.tight_layout()
plt.savefig("../docs/pics/regression/plot_mlp.png", dpi=300)
plt.close(fig2)  # Cierra la figura

print("Archivos de imagen generados correctamente.")
