import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from random import randint

class DiffusionParticles:
    def __init__(self, M, N, T, K, r=None, s=None, initial_distribution=None, Nexp=100, P= None):
        self.M = M
        self.N = N
        self.T = T
        self.K = K
        self.r = r
        self.s = s
        self.Nexp = Nexp
        self.P = P if P is not None else randint(1,2000)
        self.mask = np.ones((M, N))
        self.neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        # Si se proporcionan r y s, establecer las celdas correspondientes en 0
        if r is not None and s is not None:
            self.mask[:r, s:] = 0
        
        # Si se proporciona una distribución inicial, se utiliza, de lo contrario se genera una aleatoria
        if initial_distribution is not None:
            self.u0 = initial_distribution
            self.u0 /= np.sum(self.u0)  # Normalizar para que la suma sea 1
        else:
            self.u0 = np.random.rand(M, N)
            self.u0 /= np.sum(self.u0)  # Normalizar para que la suma sea 1
            
    def simulate_diffusion(self):
        particles = np.random.choice(self.M * self.N, size=self.P, p=self.u0.ravel())  # Distribuir partículas según u(0)
        particle_positions = np.unravel_index(particles, (self.M, self.N))
        grid = np.zeros((self.M, self.N), dtype=int)
        
        for x, y in zip(particle_positions[0], particle_positions[1]):
            grid[x, y] += 1

        # Listas para guardar las densidades en el tiempo
        densities = [grid.copy()]

        for t in range(self.T):
            new_grid = np.zeros((self.M, self.N), dtype=int)
            for i in range(self.M):
                for j in range(self.N):
                    if self. mask[i, j] == 0:
                        new_grid[i, j] = 0
                        continue
                    if grid[i, j] > 0:
                        for _ in range(grid[i, j]):
                            if np.random.rand() < self.K:
                                move = self.neighbors[np.random.randint(0, 8)]
                                ni, nj = (i + move[0]) % self.M, (j + move[1]) % self.N
                                new_grid[ni, nj] += 1
                            else:
                                new_grid[i, j] += 1
            grid = new_grid
            densities.append(grid.copy())

        return densities
    
    # Simulación y promedios
    def getSimulation(self):
        all_densities = np.array([density.copy() for _ in range(self.T + 1)], dtype='float')
        for exp in range(self.Nexp):
            densities = self.simulate_diffusion(self)
            for t, density in enumerate(densities):
                if exp != 0:
                    all_densities[t] += density

        # Normalizar
        all_densities /= (self.Nexp * self.P)
        return all_densities

    # Función para crear la animación de la simulación
    def animate_simulation(self, densities):
        def animate(i):
            plt.clf()  # Limpiamos el gráfico
            plt.imshow(densities[i], cmap='gray', interpolation='nearest')
            plt.title(f'Tiempo: {i}')
            plt.colorbar()
            
        # Crear la animación
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, animate, frames=self.T, interval=100)
        return ani