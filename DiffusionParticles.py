import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parámetros de la simulación
M, N = 50, 50  # Tamaño del grid
T = 100        # Tiempo total de simulación
K = 0.5        # Parámetro de velocidad de difusión
P = 100         # Número de partículase
r = 26        # Altura de la región bloqueada
s = 26         # Ancho de la región bloqueada
mask = np.ones((M, N))
Nexp = 100       # Número de experimentos

if r is not None and s is not None:
    mask[:r, s:] = 0

initial_distribution = np.zeros((M, N))
initial_distribution[25, 25] = 1.0
initial_distribution[12, 25] = 1.0
initial_distribution[40, 10] = 1.0
initial_distribution[5, 5] = 1.0

# Distribución inicial u(0)
# u0 = np.random.rand(M, N)
u0 = initial_distribution
u0 /= u0.sum()  # Normalizar para que la suma sea 1

# Función para realizar un experimento
def simulate_diffusion(M, N, P, K, T):
    particles = np.random.choice(M * N, size=P, p=u0.ravel())  # Distribuir partículas según u(0)
    particle_positions = np.unravel_index(particles, (M, N))
    density0 = np.zeros((M, N), dtype=int)
    
    for x, y in zip(particle_positions[0], particle_positions[1]):
        density0[x, y] += 1

    # Listas para guardar las densidades en el tiempo
    densities = []
    densities.append(density0)

    # Movimientos posibles: 8 vecinos + quedarse en la misma posición
    moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (0, 0)]

    for t in range(T):
        new_grid = np.zeros((M, N), dtype=int)
        for i in range(M):
            for j in range(N):
                if mask[i, j] == 0:
                    new_grid[i, j] = 0
                    continue
                if density0[i, j] > 0:
                    for _ in range(density0[i, j]):
                        if np.random.rand() < K:
                            move = moves[np.random.randint(0, 8)]
                            ni, nj = (i + move[0]) % M, (j + move[1]) % N  # Periodic boundary conditions
                            new_grid[ni, nj] += 1
                        else:
                            new_grid[i, j] += 1
        density0 = new_grid
        densities.append(density0.copy())

    return densities

# Simulación y promedios
average_density = np.zeros((M, N), dtype=float)

for exp in range(Nexp):
    densities = simulate_diffusion(M, N, P, K, T)
    for t, density in enumerate(densities):
        if exp == 0:
            all_densities = np.array([density.copy() for _ in range(T + 1)], dtype='float')
        else:
            all_densities[t] += density

# Normalizar
all_densities /= (Nexp * P)

def animate(i):
    plt.clf()  # Limpiamos el gráfico
    plt.imshow(all_densities[i], cmap='gray', interpolation='nearest')
    plt.title(f'Tiempo: {i}')
    plt.colorbar()
    
# Crear la animación
fig = plt.figure()
ani = animation.FuncAnimation(fig, animate, frames=T, interval=100)

ani.save('./animations/inciso_2/diffusion_simulation.mp4', writer='ffmpeg', fps=10)

plt.show()

