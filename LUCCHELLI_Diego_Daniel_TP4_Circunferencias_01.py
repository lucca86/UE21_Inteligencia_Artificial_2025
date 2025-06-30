import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, exposure, filters
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from tkinter import filedialog, Tk

def seleccionar_imagen():
    ventana = Tk()
    ventana.withdraw()
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
    )
    return ruta

def detectar_circunferencias():
    ruta = seleccionar_imagen()
    if not ruta:
        print("No se seleccionó ninguna imagen.")
        return

    imagen_color = io.imread(ruta)
    imagen_gris = color.rgb2gray(imagen_color)
    imagen_gris = exposure.equalize_adapthist(imagen_gris, clip_limit=0.03)
    imagen_gris_suave = filters.gaussian(imagen_gris, sigma=1.0)

    bordes = canny(imagen_gris_suave, sigma=1.4, low_threshold=0.05, high_threshold=0.2)

    plt.figure(figsize=(6, 5))
    plt.imshow(bordes, cmap='gray')
    plt.title("Bordes detectados (Canny ajustado)")
    plt.axis('off')
    plt.show()

    radios = np.arange(25, 50, 1)
    hough_res = hough_circle(bordes, radios)

    accums, cx, cy, r = hough_circle_peaks(
        hough_res, radios,
        total_num_peaks=20,
        normalize=True,
        min_xdistance=10,
        min_ydistance=10
    )

    # Filtrar círculos con acumulación mínima (umbral)
    UMBRAL_ACUMULACION = 0.4
    filtrados = [(x, y, rad, acc) for x, y, rad, acc in zip(cx, cy, r, accums) if acc >= UMBRAL_ACUMULACION]

    # Eliminar solapamientos: si centros están muy cerca y radios similares
    def es_similar(c1, c2, tol_dist=10, tol_radio=5):
        dx = abs(c1[0] - c2[0])
        dy = abs(c1[1] - c2[1])
        dr = abs(c1[2] - c2[2])
        return dx <= tol_dist and dy <= tol_dist and dr <= tol_radio

    definitivos = []
    for c in filtrados:
        if not any(es_similar(c, d) for d in definitivos):
            definitivos.append(c)

    # Dibujar resultados
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(imagen_color)
    ax[0].set_title("Imagen original")
    ax[0].axis('off')

    imagen_detectada = imagen_color.copy()

    if not definitivos:
        print("⚠️ No se detectaron circunferencias válidas.")
        ax[1].imshow(imagen_color)
        ax[1].set_title("No se detectaron circunferencias")
    else:
        for i, (x, y, rad, acc) in enumerate(definitivos):
            circy, circx = circle_perimeter(y, x, rad)
            circy = np.clip(circy, 0, imagen_color.shape[0] - 1)
            circx = np.clip(circx, 0, imagen_color.shape[1] - 1)
            imagen_detectada[circy, circx] = (0, 255,0) if i != 0 else (255, 0, 0)  # rojo el mejor

        ax[1].imshow(imagen_detectada)
        ax[1].set_title(f"Círculos finales: {len(definitivos)} (mejor marcado en rojo)")

    ax[1].axis('off')
    plt.tight_layout()
    plt.show()

    # Mostrar resultados
    print("\n🔍 Círculos finales detectados:")
    for i, (x, y, rad, acc) in enumerate(definitivos):
        marca = "⭐ Candidato más fuerte" if i == 0 else ""
        print(f" - Centro: ({x}, {y}), Radio: {rad}px, Acumulación: {acc:.3f} {marca}")

# Ejecutar
detectar_circunferencias()
