import numpy as np
import matplotlib.pyplot as plt

class HMMMedico:
    
    def __init__(self):
        self.estados = ['Saludable', 'Resfriado', 'Gripe']
        self.n_estados = 3
        self.observaciones = ['Temp Normal', 'Fiebre Leve', 'Fiebre Alta']
        self.n_obs = 3
        
        self.pi = np.array([0.7, 0.2, 0.1])
        
        self.A = np.array([
            [0.80, 0.15, 0.05],
            [0.40, 0.50, 0.10],
            [0.20, 0.20, 0.60]
        ])
        
        self.B = np.array([
            [0.85, 0.13, 0.02],
            [0.30, 0.55, 0.15],
            [0.05, 0.25, 0.70]
        ])
        
        self.colores_estados = {0: 'green', 1: 'orange', 2: 'red'}
        self.colores_obs = {0: 'blue', 1: 'gold', 2: 'darkred'}
    
    def simular_paciente(self, dias=20):
        estados = np.zeros(dias, dtype=int)
        observaciones = np.zeros(dias, dtype=int)
        estado_actual = np.random.choice(self.n_estados, p=self.pi)
        
        for t in range(dias):
            estados[t] = estado_actual
            observaciones[t] = np.random.choice(self.n_obs, p=self.B[estado_actual])
            estado_actual = np.random.choice(self.n_estados, p=self.A[estado_actual])
        
        return estados, observaciones
    
    def viterbi(self, observaciones):
        T = len(observaciones)
        n = self.n_estados
        V = np.zeros((n, T))
        camino = np.zeros((n, T), dtype=int)
        
        for s in range(n):
            V[s, 0] = self.pi[s] * self.B[s, observaciones[0]]
        
        for t in range(1, T):
            for s in range(n):
                probabilidades = []
                for s_prev in range(n):
                    prob = V[s_prev, t-1] * self.A[s_prev, s] * self.B[s, observaciones[t]]
                    probabilidades.append(prob)
                mejor = np.argmax(probabilidades)
                V[s, t] = probabilidades[mejor]
                camino[s, t] = mejor
        
        estados_inferidos = np.zeros(T, dtype=int)
        estados_inferidos[T-1] = np.argmax(V[:, T-1])
        
        for t in range(T-2, -1, -1):
            estados_inferidos[t] = camino[estados_inferidos[t+1], t+1]
        
        return estados_inferidos, V
    
    def calcular_precision(self, reales, inferidos):
        return np.mean(reales == inferidos)
    
    def graficar_simulacion(self, reales, observaciones, inferidos, V=None):
        dias = range(1, len(reales) + 1)
        
        plt.figure(figsize=(12, 3))
        for i, e in enumerate(reales):
            plt.scatter(dias[i], e, color=self.colores_estados[e], s=80)
        plt.plot(dias, reales, 'k-', alpha=0.3)
        plt.yticks([0, 1, 2], self.estados)
        plt.title('1. Estado Real del Paciente')
        plt.xlabel('Días')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 3))
        for i, o in enumerate(observaciones):
            plt.scatter(dias[i], o, color=self.colores_obs[o], s=80)
        plt.plot(dias, observaciones, 'k-', alpha=0.3)
        plt.yticks([0, 1, 2], self.observaciones)
        plt.title('2. Síntomas Observados')
        plt.xlabel('Días')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(12, 3))
        plt.plot(dias, reales, 'g-o', label='Real', markersize=5)
        plt.plot(dias, inferidos, 'r--x', label='Diagnosticado', markersize=5)
        plt.yticks([0, 1, 2], self.estados)
        plt.title('3. Comparación: Real vs Diagnóstico')
        plt.xlabel('Días')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        if V is not None:
            plt.figure(figsize=(12, 3))
            confianza = np.max(V, axis=0)
            plt.plot(dias, confianza, 'b-o', markersize=5)
            plt.fill_between(dias, confianza, alpha=0.3)
            plt.title('4. Confianza del Diagnóstico')
            plt.xlabel('Días')
            plt.ylabel('Probabilidad')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def recomendacion_medica(self, ultimo_estado):
        if ultimo_estado == 0:
            return "✅ Paciente estable. Continuar monitoreo."
        elif ultimo_estado == 1:
            return "⚠️ Resfriado. Reposo y líquidos."
        else:
            return "🚨 ALERTA: Posible gripe. Valoración médica."