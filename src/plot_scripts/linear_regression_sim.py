#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

# --- 1. Generación de datos sintéticos (igual que antes) ---
np.random.seed(42)  # Para reproducibilidad

X = 2 * np.random.rand(100, 1)
# La relación real es y = 4 + 3x + ruido
y = 4 + 3 * X + np.random.randn(100, 1)

# Añadir el término de sesgo (bias) x0=1 a cada instancia
X_b = np.c_[np.ones((100, 1)), X]


# --- 2. Cálculo de parámetros con la Ecuación Normal (Mínimos Cuadrados) ---

print("Calculando los parámetros con la Ecuación Normal...")

# Esta es la implementación directa de la fórmula de mínimos cuadrados
# theta = (X^T * X)^(-1) * X^T * y
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("\nCálculo finalizado.")
print("Parámetros óptimos encontrados (theta):")
print(f"  - Intercepto (θ₀): {theta[0][0]:.4f}")
print(f"  - Pendiente  (θ₁): {theta[1][0]:.4f}")
# Los valores deberían ser muy cercanos a los teóricos (4 y 3) y a los de Gradiente Descendiente.


# --- 3. Visualización de los resultados (igual que antes) ---

plt.figure(figsize=(10, 6))

# a) Dibujar los puntos de datos originales
plt.scatter(X, y, alpha=0.7, label="datos")

# b) Dibujar la línea de regresión encontrada
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta)
plt.plot(X_new, y_predict, "r-", linewidth=3, label="regresión")

# c) Opcional: Dibujar los errores (residuos)
y_predicted_points = X_b.dot(theta)
for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], y_predicted_points[i]], "g--", linewidth=0.8)
plt.plot([], [], "g--", label="error")

# Títulos y etiquetas
# plt.title("Regresión Lineal con Mínimos Cuadrados (Ecuación Normal)", fontsize=16)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.axis([0, 2, 0, 15])
# plt.tight_layout()

# Guardar y mostrar el gráfico
plt.savefig("../docs/pics/regression/linear_regression.png", dpi=300)
