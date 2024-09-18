import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DiffusionModel:
    def __init__(self, M, N, T, K, r=None, s=None, initial_distribution=None):
        self.M = M
        self.N = N
        self.T = T
        self.K = K
        self.r = r
        self.s = s
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
        
        self.history = [self.u0]
        
    # Función para actualizar el estado del grid en cada paso de tiempo
    def diffusion_step(self, u, K, M, N):
        u_new = np.copy(u)  # Crear una copia del grid actual
        
        # Iterar sobre cada celda y actualizar su valor según la ecuación de difusión
        for i in range(0, M):
            for j in range(0, N):
                
                if self.mask[i, j] == 0:
                    u_new[i, j] = 0
                    continue
                
                # Sumar los valores de los 8 vecinos (asegurarse de no salir de los límites del array)
                neighbors_sum = 0
                count = 0
                for di, dj in self.neighbors:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < M and 0 <= nj < N and self.mask[ni, nj] == 1:  # Considera solo celdas activas
                        neighbors_sum += u[ni, nj]
                        count += 1
                        
                # Aplicar la fórmula completa de difusión para las celdas internas
                u_new[i, j] = (1 - K) * u[i, j] + (K / count) * neighbors_sum
        
        return u_new
    
    # Función para simular la difusión en el grid
    def simulate_diffusion(self):
        u = self.u0
        for t in range(self.T):
            u = self.diffusion_step(u, self.K, self.M, self.N)
            self.history.append(u)
        return self.history
    
    # Función para crear la animación de la simulación
    def animate_simulation(self):
        def animate(i):
            plt.clf()  # Limpiamos el gráfico
            plt.imshow(self.history[i], cmap='gray', interpolation='nearest')
            plt.title(f'Tiempo: {i}')
            plt.colorbar()
            
        # Crear la animación
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, animate, frames=self.T, interval=100)
        return ani
        
