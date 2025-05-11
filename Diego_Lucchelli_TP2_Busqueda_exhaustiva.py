import matplotlib.pyplot as plt

# --------------------------
# 1. Definir espacio de estados
# --------------------------
H = list(range(7))  # Posiciones horizontales: 0 al 6
pos_inicial = 3     # Posición teórica (B)
pos_objetivo = 5    # Posición real del punto de montaje (A)

# --------------------------
# 2. Búsqueda exhaustiva 1D
# --------------------------
def busqueda_horizontal_exhaustiva(H, inicio, objetivo):
    pasos = [inicio]
    offset = 1  # Incremento ΔH
    direccion = 1  # 1: derecha, -1: izquierda
    i = 1

    # Alternar entre izquierda y derecha con incremento creciente
    while True:
        # Calcular nuevo paso
        desplazamiento = direccion * i
        nueva_pos = inicio + desplazamiento

        if 0 <= nueva_pos < len(H):
            pasos.append(nueva_pos)
            if nueva_pos == objetivo:
                return pasos  # Objetivo alcanzado

        # Alternar dirección
        direccion *= -1
        if direccion == 1:
            i += 1  # Incrementar distancia cada vez que se completa un ciclo izquierda-derecha

        if len(pasos) > len(H)*2:
            return None  # No se encontró el objetivo

# Ejecutar búsqueda
camino = busqueda_horizontal_exhaustiva(H, pos_inicial, pos_objetivo)

# --------------------------
# 3. Mostrar resultados
# --------------------------
print("Camino explorado (posición horizontal H):")
print(" → ".join(map(str, camino)) if camino else "No se encontró el objetivo.")

# --------------------------
# 4. Graficar las posiciones exploradas
# --------------------------
colores = ['lightgray'] * len(H)
for i in camino:
    colores[i] = 'skyblue'
colores[pos_inicial] = 'orange'   # B: posición inicial
colores[pos_objetivo] = 'green'   # A: objetivo real

plt.figure(figsize=(10, 2))
plt.bar(H, [1]*len(H), color=colores, edgecolor='black')
plt.xticks(H)
plt.yticks([])
plt.xlabel("Posición sobre eje H")
plt.title("Búsqueda exhaustiva 1D desde B ({}), hacia A ({})".format(pos_inicial, pos_objetivo))
plt.grid(axis='x')
plt.tight_layout()
plt.show()
