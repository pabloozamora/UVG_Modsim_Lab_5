import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class DiffusionModel:
    def __init__(self, M, N, T, K):
        self.M = M
        self.N = N
        self.T = T
        self.K = K
        self.u0 = np.random.rand(M, N)
        self.u0 /= np.sum(self.u0)
        self.history = [self.u0]
        
    # Función para actualizar el estado del grid en cada paso de tiempo
    def diffusion_step(self, u, K, M, N):
        u_new = np.copy(u)  # Crear una copia del grid actual
        # Iterar sobre cada celda y actualizar su valor según la ecuación de difusión
        for i in range(1, M-1):
            for j in range(1, N-1):
                # Sumar los valores de los 8 vecinos
                neighbors_sum = (
                    u[i-1, j-1] + u[i-1, j] + u[i-1, j+1] +
                    u[i, j-1]               + u[i, j+1] +
                    u[i+1, j-1] + u[i+1, j] + u[i+1, j+1]
                )
                # Aplicar la fórmula de difusión
                u_new[i, j] = (1 - K) * u[i, j] + (K / 8) * neighbors_sum
        return u_new
    
    # Función para simular la difusión en el grid
    def simulate_diffusion(self):
        u = self.u0
        for t in range(self.T):
            u = self.diffusion_step(u, self.K, self.M, self.N)
            self.history.append(u)
        return self.history
    
    def animate_simulation(self):
        def animate(i):
            plt.clf()  # Limpiamos el gráfico
            plt.imshow(self.history[i], cmap='gray', interpolation='nearest')
            plt.title(f'Tiempo: {i}')
            plt.colorbar()
            
        # Creamos la animación
        fig = plt.figure()
        ani = animation.FuncAnimation(fig, animate, frames=self.T, interval=1000)
        return ani
        
# Parámetros de la simulación
M, N = 50, 50  # Tamaño del grid
T = 100        # Tiempo total de simulación
K = 0.5        # Parámetro de velocidad de difusión
u0 = np.random.rand(M, N)  # Distribución inicial aleatoria
u0 /= np.sum(u0)  # Normalizar para que la suma sea 1

# Creamos una instancia de la clase DiffusionModel
model = DiffusionModel(M, N, T, K)
model.u0 = u0  # Asignamos la distribución inicial aleatoria
model.simulate_diffusion()  # Ejecutamos la simulación
ani = model.animate_simulation()  # Creamos la animación


# Mostramos la animación
plt.show()
