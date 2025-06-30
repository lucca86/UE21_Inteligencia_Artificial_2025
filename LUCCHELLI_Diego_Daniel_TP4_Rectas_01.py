import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, feature, exposure
from skimage.transform import hough_line, hough_line_peaks
from tkinter import filedialog, Tk

def seleccionar_imagen():
    ventana = Tk()
    ventana.withdraw()
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    return ruta

def detectar_rectas():
    ruta = seleccionar_imagen()
    if not ruta:
        print("No se seleccionó ninguna imagen.")
        return

    # Cargar imagen y convertir a escala de grises
    imagen_color = io.imread(ruta)
    imagen_gris = color.rgb2gray(imagen_color)

    # Aplicar ecualización para mejorar contraste (opcional)
    imagen_gris = exposure.equalize_adapthist(imagen_gris, clip_limit=0.03)

    # Detección de bordes con Canny (ajustar sigma si no detecta nada)
    bordes = feature.canny(imagen_gris, sigma=1.5)

    # Mostrar bordes para ver si se detectaron correctamente
    plt.figure(figsize=(6, 5))
    plt.imshow(bordes, cmap='gray')
    plt.title("Bordes detectados (Canny)")
    plt.axis('off')
    plt.show()

    # Aplicar Transformada de Hough
    h, theta, d = hough_line(bordes)

    # Obtener líneas más destacadas
    acumulador, angulos, distancias = hough_line_peaks(h, theta, d, threshold=0.2 * np.max(h))

    # Visualización
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Imagen original
    ax[0].imshow(imagen_color)
    ax[0].set_title("Imagen original")
    ax[0].axis('off')

    # Imagen con líneas
    ax[1].imshow(imagen_color.copy())
    if len(angulos) == 0:
        ax[1].set_title("⚠️ No se detectaron líneas.")
        print("No se detectaron líneas. Revisa el contraste o ajusta parámetros.")
    else:
        for angle, dist in zip(angulos, distancias):
            (h, w) = imagen_gris.shape
            x0 = 0
            y0 = (dist - x0 * np.cos(angle)) / np.sin(angle)
            x1 = w
            y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
            ax[1].plot((x0, x1), (y0, y1), '-r', linewidth=2)
        ax[1].set_title(f"Líneas detectadas: {len(angulos)}")

    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

# Ejecutar
detectar_rectas()
