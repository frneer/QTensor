#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# --- 1. Definición de funciónes ---


def sigmoid(z):
    """Función de activación sigmoidea."""
    return 1 / (1 + np.exp(-z))


def cross_entropy_loss(y_true, y_pred):
    """Función de costo de entropía cruzada para monitorear el
    entrenamiento."""
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -(1 / len(y_true)) * np.sum(
        y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
    )


# --- 2. Generación de datos sintéticos ---
np.random.seed(42)  # Para reproducibilidad

# Clase 0
mean_0 = [2, 3]
cov_0 = [[1, 0.5], [0.5, 1]]
X0 = np.random.multivariate_normal(mean_0, cov_0, 100)
y0 = np.zeros(100)

# Clase 1
mean_1 = [6, 7]
cov_1 = [[1.5, -0.5], [-0.5, 1.5]]
X1 = np.random.multivariate_normal(mean_1, cov_1, 100)
y1 = np.ones(100)

# Combinar los datos
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Añadir término de sesgo (bias)

# --- 3. Entrenamiento del modelo con Descenso por Gradiente ---

print("Iniciando entrenamiento del modelo...")

learning_rate = 0.1
n_iterations = 2000
m = len(y)

# Inicialización aleatoria de los parámetros (theta)
theta = np.random.randn(3, 1)
y_col = y.reshape(-1, 1)  # Asegurar que 'y' sea un vector columna

for iteration in range(n_iterations):
    # 1. Calcular la combinación lineal
    z = X_b.dot(theta)
    # 2. Aplicar la función sigmoidea para obtener probabilidades
    predictions = sigmoid(z)
    # 3. Calcular el gradiente
    gradients = (1 / m) * X_b.T.dot(predictions - y_col)
    # 4. Actualizar los parámetros
    theta = theta - learning_rate * gradients

    # Opcional: Imprimir el costo cada 200 iteraciones para ver el progreso
    if iteration % 200 == 0:
        loss = cross_entropy_loss(y_col, predictions)
        print(f"Iteración {iteration:4d}: Costo = {loss:.4f}")

print("\nEntrenamiento finalizado.")
print("Parámetros finales (theta) encontrados:")
print(theta)


# --- 4. Preparación de datos para la visualización ---

# Calcular 'z' para todos los puntos usando los parámetros ENTRENADOS
z_data = X_b.dot(theta).flatten()

# Identificar puntos correctamente clasificados y mal clasificados
# La predicción es 1 si z > 0 (prob > 0.5), y 0 si z <= 0 (prob <= 0.5).
correct_0_mask = (y == 0) & (z_data <= 0)
misclassified_0_mask = (y == 0) & (z_data > 0)

correct_1_mask = (y == 1) & (z_data > 0)
misclassified_1_mask = (y == 1) & (z_data <= 0)

# --- 5. Crear el gráfico combinado final ---

plt.figure(figsize=(10, 6))

# Dibujar la curva sigmoidea
z_range = np.linspace(z_data.min() - 2, z_data.max() + 2, 500)
sigmoid_values = sigmoid(z_range)
plt.plot(
    z_range,
    sigmoid_values,
    color="purple",
    linewidth=2.5,
    label="Función Sigmoidea",
    zorder=5,
)

# Añadir jitter para mejor visualización
jitter_strength = 0.05
y_jittered = y + np.random.uniform(
    -jitter_strength, jitter_strength, size=len(y)
)

# Dibujar los puntos correctamente clasificados
plt.scatter(
    z_data[correct_0_mask],
    y_jittered[correct_0_mask],
    color="blue",
    alpha=0.7,
    label="Clase 0 (Correcto)",
    s=60,
    zorder=3,
)
plt.scatter(
    z_data[correct_1_mask],
    y_jittered[correct_1_mask],
    color="red",
    alpha=0.7,
    label="Clase 1 (Correcto)",
    s=60,
    zorder=3,
)

# Dibujar los puntos mal clasificados (con bordes del otro color)
plt.scatter(
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
plt.scatter(
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

# Líneas de referencia
plt.axhline(
    y=0.5, color="gray", linestyle="--"
)  # label='Umbral de Decisión (0.5)'
plt.axvline(
    x=0, color="black", linestyle=":", linewidth=2
)  # label='Frontera (z=0)'

# Títulos y etiquetas
# plt.title("Regresión Logística", fontsize=16)
plt.xlabel("z = θ₀ + θ₁x₁ + θ₂x₂ (Entrada a la Sigmoide)", fontsize=12)
plt.ylabel("Probabilidad", fontsize=12)
plt.yticks([0, 0.5, 1], ["Clase 0", "0.5", "Clase 1"])
plt.legend(loc="upper left", fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
# plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para que la leyenda no se corte

# Guardar y mostrar el gráfico
plt.savefig("../docs/pics/regression/logistic_regression.png", dpi=300)
