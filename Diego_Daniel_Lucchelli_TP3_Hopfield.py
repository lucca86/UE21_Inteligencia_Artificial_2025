import numpy as np
import matplotlib.pyplot as plt

# Paso 1: Generar una imagen de un círculo hueco (solo borde)
def generate_hollow_circle_matrix(size=12, radius=4, thickness=1, center=None):
    # Crea matriz base
    matrix = -np.ones((size, size), dtype=int)
    if center is None:
        center = (size // 2, size // 2)
    for i in range(size):
        for j in range(size):
            dist_sq = (i - center[0])**2 + (j - center[1])**2
            if (radius - thickness)**2 <= dist_sq <= radius**2:
                matrix[i, j] = 1
    return matrix

# Paso 2: Definir el patrón y entrenar la red de Hopfield
pattern = generate_hollow_circle_matrix()
shape = pattern.shape
N = shape[0] * shape[1]
pattern_flat = pattern.flatten()

# Matriz de pesos usando el producto externo
# Crea la matriz de pesos
W = np.outer(pattern_flat, pattern_flat)
np.fill_diagonal(W, 0)

# Paso 3: Función de actualización
def hopfield_update(x, W, steps=10):
    for _ in range(steps):
        # Activación de neuronas
        x = np.sign(W @ x)
    return x

# Paso 4: Agregar ruido al patrón de prueba
def add_noise(x, noise_level=0.2):
    noisy = x.copy()
    n_flip = int(len(x) * noise_level)
    indices = np.random.choice(len(x), n_flip, replace=False)
    noisy[indices] *= -1
    return noisy

# Aplicar ruido aleatorio
test_pattern_flat = add_noise(pattern_flat, noise_level=0.2)

# Recuperar patrón
recovered_flat = hopfield_update(test_pattern_flat, W)
recovered_img = recovered_flat.reshape(shape)

# Paso 5: Calcular el centro del círculo recuperado
indices = np.argwhere(recovered_img == 1)
center_yx = indices.mean(axis=0)
center_coords = tuple(np.round(center_yx).astype(int))

# Paso 6: Visualización de resultados
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(pattern, cmap='gray')
axs[0].set_title("Círculo hueco original")
axs[0].axis('off')

axs[1].imshow(test_pattern_flat.reshape(shape), cmap='gray')
axs[1].set_title("Círculo con ruido")
axs[1].axis('off')

axs[2].imshow(recovered_img, cmap='gray')
axs[2].scatter(center_coords[1], center_coords[0], color='red', label='Centro')
axs[2].legend()
axs[2].set_title("Recuperado y centro")
axs[2].axis('off')

plt.tight_layout()
plt.show()

print("Centro aproximado del círculo (fila, columna):", center_coords)
