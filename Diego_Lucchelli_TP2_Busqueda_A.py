import heapq
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --------------------------
# 1. Parámetros
# --------------------------
nodos = list(range(8))     # Posiciones de 0 a 7
inicio = 2                 # Posición inicial (B)
objetivo = 6               # Posición de montaje (A)

# --------------------------
# 2. Heurística: distancia absoluta a la meta
# --------------------------
def h(n):
    return abs(n - objetivo)

# --------------------------
# 3. A* Search
# --------------------------
def a_estrella(inicio, objetivo):
    frontera = []
    heapq.heappush(frontera, (h(inicio), [inicio]))  # (f(n), camino)

    visitados = set()
    costos = {inicio: 0}

    while frontera:
        _, camino = heapq.heappop(frontera)
        actual = camino[-1]

        if actual == objetivo:
            return camino

        if actual in visitados:
            continue
        visitados.add(actual)

        for vecino in [actual - 1, actual + 1]:  # Solo moverse a izquierda o derecha
            if 0 <= vecino < len(nodos):
                nuevo_costo = costos[actual] + 1  # g(n) = pasos dados
                if vecino not in costos or nuevo_costo < costos[vecino]:
                    costos[vecino] = nuevo_costo
                    prioridad = nuevo_costo + h(vecino)  # f(n) = g(n) + h(n)
                    nuevo_camino = list(camino)
                    nuevo_camino.append(vecino)
                    heapq.heappush(frontera, (prioridad, nuevo_camino))

    return None

# --------------------------
# 4. Ejecutar búsqueda
# --------------------------
camino = a_estrella(inicio, objetivo)

# --------------------------
# 5. Mostrar resultados
# --------------------------
print("Camino encontrado por A* desde nodo {} hasta {}:".format(inicio, objetivo))
print(" → ".join(map(str, camino)) if camino else "No se encontró una solución.")

# --------------------------
# 6. Visualización con NetworkX
# --------------------------
G = nx.DiGraph()
for i in range(len(camino) - 1):
    G.add_edge(camino[i], camino[i+1])

pos = nx.spring_layout(G, seed=42)

colores = []
for nodo in G.nodes:
    if nodo == inicio:
        colores.append('orange')  # Inicio (B)
    elif nodo == objetivo:
        colores.append('green')   # Objetivo (A)
    else:
        colores.append('skyblue')

plt.figure(figsize=(10, 5))
nx.draw(G, pos, with_labels=True, node_color=colores, node_size=800, font_size=14, font_weight='bold')
nx.draw_networkx_edges(G, pos, edge_color='blue', width=3, arrows=True)
plt.title("Búsqueda A* sobre el eje horizontal (8 nodos)")
plt.axis('off')
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.show()
