"""
================================================================================
MODELO OCULTO DE MARKOV (HMM) PARA DIAGNÓSTICO MÉDICO
================================================================================

DESCRIPCIÓN DEL PROBLEMA:
    En la práctica médica, el estado exacto de salud de un paciente 
    (saludable, resfriado o gripe) no es directamente observable sin pruebas
    de laboratorio costosas. Sin embargo, el médico puede registrar síntomas
    observables como la temperatura corporal (normal, fiebre leve, fiebre alta).

OBJETIVO DE LA SIMULACIÓN:
    Demostrar cómo un Modelo Oculto de Markov (HMM) puede inferir el estado
    de salud oculto a partir de observaciones de temperatura, utilizando
    el algoritmo de Viterbi para encontrar la secuencia más probable de estados.

FUNDAMENTO TEÓRICO:
    Un HMM se define por la tupla λ = (π, A, B) donde:
    - π: Distribución de probabilidad inicial de los estados
    - A: Matriz de transición entre estados ocultos
    - B: Matriz de emisión (probabilidad de observar cada síntoma dado un estado)
    
    El algoritmo de Viterbi encuentra la secuencia de estados Q que maximiza:
    P(Q | O, λ) ∝ π[q1]·B[q1,o1] · Π(t=2..T) A[q(t-1),q(t)]·B[q(t),o(t)]

JUSTIFICACIÓN DE LAS MATRICES (basada en el prompt del profesor):
    Los valores numéricos fueron asignados con base en conocimiento clínico:
    - Una persona saludable tiene 80% de mantenerse bien, 15% de resfriarse
    - El resfriado empeora a gripe en 20% de los casos
    - La gripe presenta fiebre alta en 60% de los días
    - La prevalencia inicial es 70% saludable, 20% resfriado, 10% gripe
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class HMMMedico:
    """
    Modelo Oculto de Markov para diagnóstico médico.
    
    Estados ocultos (no observables):
        0: Saludable - Sin síntomas, temperatura normal
        1: Resfriado - Infección leve, posible fiebre baja
        2: Gripe     - Infección grave, fiebre alta
    
    Observaciones (medibles):
        0: Temperatura Normal (≤ 37.2°C)
        1: Fiebre Leve (37.3°C - 38.5°C)
        2: Fiebre Alta (> 38.5°C)
    
    Parámetros del modelo:
        pi: Probabilidad inicial de cada estado
        A: Matriz de transición [3x3]
        B: Matriz de emisión [3x3]
    """
    
    def __init__(self):
        """
        Inicializa el HMM con matrices definidas clínicamente.
        Basado en el prompt original del profesor de simulación.
        """
        
        # ====================================================================
        # 1. DEFINICIÓN DE ESTADOS Y OBSERVACIONES
        # ====================================================================
        self.estados = ['Saludable', 'Resfriado', 'Gripe']
        self.n_estados = 3
        self.observaciones = ['Temp Normal', 'Fiebre Leve', 'Fiebre Alta']
        self.n_obs = 3
        
        # ====================================================================
        # 2. PROBABILIDADES INICIALES (π)
        # ====================================================================
        # Justificación: 70% saludable, 20% resfriado, 10% gripe
        self.pi = np.array([0.7, 0.2, 0.1])
        
        # ====================================================================
        # 3. MATRIZ DE TRANSICIÓN (A)
        # ====================================================================
        # Filas: estado actual (0=Saludable, 1=Resfriado, 2=Gripe)
        # Columnas: próximo estado (0=Saludable, 1=Resfriado, 2=Gripe)
        #
        # Justificación clínica (basada en prompt del profesor):
        # - Saludable → Saludable: 80% (se mantiene bien)
        # - Saludable → Resfriado: 15% (se resfría)
        # - Saludable → Gripe: 5% (gripe directa)
        # - Resfriado → Saludable: 40% (se recupera)
        # - Resfriado → Resfriado: 40% (persiste)
        # - Resfriado → Gripe: 20% (empeora)
        # - Gripe → Saludable: 20% (recuperación total)
        # - Gripe → Resfriado: 30% (mejora parcial)
        # - Gripe → Gripe: 50% (persiste)
        # ====================================================================
        self.A = np.array([
            [0.80, 0.15, 0.05],   # Desde Saludable
            [0.40, 0.40, 0.20],   # Desde Resfriado
            [0.20, 0.30, 0.50]    # Desde Gripe
        ])
        
        # ====================================================================
        # 4. MATRIZ DE EMISIÓN (B)
        # ====================================================================
        # Filas: estado oculto (0=Saludable, 1=Resfriado, 2=Gripe)
        # Columnas: observación (0=Normal, 1=Fiebre Leve, 2=Fiebre Alta)
        #
        # Justificación clínica (basada en prompt del profesor):
        # - Saludable: 90% normal, 8% fiebre leve (ejercicio), 2% fiebre alta (error)
        # - Resfriado: 30% normal (asintomático), 50% fiebre leve, 20% fiebre alta
        # - Gripe: 10% normal (inicio), 30% fiebre leve, 60% fiebre alta
        # ====================================================================
        self.B = np.array([
            [0.90, 0.08, 0.02],   # Desde Saludable
            [0.30, 0.50, 0.20],   # Desde Resfriado
            [0.10, 0.30, 0.60]    # Desde Gripe
        ])
        
        # Colores para visualización
        self.colores_estados = {0: 'green', 1: 'orange', 2: 'red'}
        self.colores_obs = {0: 'blue', 1: 'gold', 2: 'darkred'}
        
        # Verificar que las matrices sean válidas
        self._validar_matrices()
    
    def _validar_matrices(self):
        """Verifica que las matrices sean probabilísticas (cada fila suma 1.0)."""
        for i, fila in enumerate(self.A):
            suma = np.sum(fila)
            if abs(suma - 1.0) > 1e-6:
                print(f"⚠️ Advertencia: Fila {i} de A suma {suma}")
        
        for i, fila in enumerate(self.B):
            suma = np.sum(fila)
            if abs(suma - 1.0) > 1e-6:
                print(f"⚠️ Advertencia: Fila {i} de B suma {suma}")
        
        suma_pi = np.sum(self.pi)
        if abs(suma_pi - 1.0) > 1e-6:
            print(f"⚠️ Advertencia: π suma {suma_pi}")
    
    # =========================================================================
    # SIMULACIÓN DEL PACIENTE
    # =========================================================================
    
    def simular_paciente(self, dias=20):
        """
        Genera una secuencia aleatoria de estados y observaciones.
        
        Parámetros:
            dias (int): Número de días a simular (default: 20)
        
        Retorna:
            tuple: (estados_reales, observaciones)
        """
        estados = np.zeros(dias, dtype=int)
        observaciones = np.zeros(dias, dtype=int)
        
        # Seleccionar estado inicial según π
        estado_actual = np.random.choice(self.n_estados, p=self.pi)
        
        for t in range(dias):
            estados[t] = estado_actual
            observaciones[t] = np.random.choice(self.n_obs, p=self.B[estado_actual])
            estado_actual = np.random.choice(self.n_estados, p=self.A[estado_actual])
        
        return estados, observaciones
    
    # =========================================================================
    # ALGORITMO DE VITERBI (INFERENCIA)
    # =========================================================================
    
    def viterbi(self, observaciones):
        """
        Algoritmo de Viterbi: encuentra la secuencia de estados más probable.
        
        Parámetros:
            observaciones (np.array): Lista de observaciones (enteros 0,1,2)
        
        Retorna:
            tuple: (estados_inferidos, V)
                - estados_inferidos: lista con la secuencia más probable
                - V: matriz de probabilidades (n_estados x T)
        
        Complejidad temporal: O(T * n²) donde T = días, n = número de estados
        """
        T = len(observaciones)
        n = self.n_estados
        
        # Matriz de probabilidades V[t][s] = max probabilidad de estar en estado s en t
        V = np.zeros((n, T))
        
        # Matriz de caminos: guarda el estado anterior que maximizó la probabilidad
        camino = np.zeros((n, T), dtype=int)
        
        # ====================================================================
        # INICIALIZACIÓN (t = 0)
        # ====================================================================
        for s in range(n):
            V[s, 0] = self.pi[s] * self.B[s, observaciones[0]]
        
        # ====================================================================
        # RECURSIÓN (t = 1 .. T-1)
        # ====================================================================
        for t in range(1, T):
            for s in range(n):
                # Calcular probabilidad de llegar a s desde cada estado anterior
                probabilidades = []
                for s_prev in range(n):
                    prob = V[s_prev, t-1] * self.A[s_prev, s] * self.B[s, observaciones[t]]
                    probabilidades.append(prob)
                
                # Elegir el estado anterior que maximiza la probabilidad
                mejor = np.argmax(probabilidades)
                V[s, t] = probabilidades[mejor]
                camino[s, t] = mejor
        
        # ====================================================================
        # TERMINACIÓN Y BACKTRACKING
        # ====================================================================
        estados_inferidos = np.zeros(T, dtype=int)
        estados_inferidos[T-1] = np.argmax(V[:, T-1])
        
        # Recorrer hacia atrás reconstruyendo la secuencia
        for t in range(T-2, -1, -1):
            estados_inferidos[t] = camino[estados_inferidos[t+1], t+1]
        
        return estados_inferidos, V
    
    # =========================================================================
    # MÉTRICAS Y PRECISIÓN
    # =========================================================================
    
    def calcular_precision(self, reales, inferidos):
        """Calcula el porcentaje de aciertos."""
        return np.mean(reales == inferidos)
    
    def recomendacion_medica(self, ultimo_estado):
        """Genera recomendaciones médicas basadas en el último estado inferido."""
        if ultimo_estado == 0:
            return "✅ Paciente estable. Continuar monitoreo."
        elif ultimo_estado == 1:
            return "⚠️ Resfriado. Reposo y líquidos. Monitorear evolución."
        else:
            return "🚨 ALERTA: Posible gripe. Valoración médica urgente."
    
    # =========================================================================
    # VISUALIZACIONES
    # =========================================================================
    
    def graficar_simulacion(self, reales, observaciones, inferidos, V=None, guardar=True):
        """
        Genera 4 gráficos para visualizar los resultados.
        
        Parámetros:
            reales (np.array): Estados reales
            observaciones (np.array): Observaciones
            inferidos (np.array): Estados inferidos por Viterbi
            V (np.array): Matriz de probabilidades de Viterbi
            guardar (bool): Si es True, guarda los gráficos en la carpeta 'resultados/'
        """
        dias = range(1, len(reales) + 1)
        
        # Crear carpeta de resultados si no existe
        if guardar:
            if not os.path.exists('resultados'):
                os.makedirs('resultados')
        
        # ====================================================================
        # GRÁFICO 1: ESTADOS REALES DEL PACIENTE
        # ====================================================================
        plt.figure(figsize=(12, 4))
        for i, e in enumerate(reales):
            plt.scatter(dias[i], e, color=self.colores_estados[e], s=80, alpha=0.7)
        plt.plot(dias, reales, 'k-', alpha=0.3, linewidth=1)
        plt.yticks([0, 1, 2], self.estados)
        plt.title('1. Estado Real del Paciente (No Observable Directamente)', fontsize=12)
        plt.xlabel('Días de Evolución')
        plt.ylabel('Estado de Salud')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if guardar:
            plt.savefig('resultados/grafico_estados_reales.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # ====================================================================
        # GRÁFICO 2: OBSERVACIONES (TEMPERATURA)
        # ====================================================================
        plt.figure(figsize=(12, 4))
        for i, o in enumerate(observaciones):
            plt.scatter(dias[i], o, color=self.colores_obs[o], s=80, alpha=0.7, edgecolors='black')
        plt.plot(dias, observaciones, 'k-', alpha=0.3, linewidth=1)
        plt.yticks([0, 1, 2], self.observaciones)
        plt.title('2. Temperatura Observada (Síntoma Medible)', fontsize=12)
        plt.xlabel('Días de Evolución')
        plt.ylabel('Temperatura')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if guardar:
            plt.savefig('resultados/grafico_observaciones.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # ====================================================================
        # GRÁFICO 3: COMPARACIÓN REAL VS INFERIDO
        # ====================================================================
        plt.figure(figsize=(12, 4))
        plt.plot(dias, reales, 'g-o', label='Estado Real', linewidth=2, markersize=6)
        plt.plot(dias, inferidos, 'r--x', label='Estado Inferido (Viterbi)', linewidth=2, markersize=6)
        plt.yticks([0, 1, 2], self.estados)
        plt.title('3. Comparación: Estado Real vs Diagnóstico del Modelo', fontsize=12)
        plt.xlabel('Días de Evolución')
        plt.ylabel('Estado de Salud')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if guardar:
            plt.savefig('resultados/comparacion_real_inferido.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # ====================================================================
        # GRÁFICO 4: CONFIANZA DEL DIAGNÓSTICO
        # ====================================================================
        if V is not None:
            plt.figure(figsize=(12, 4))
            confianza = np.max(V, axis=0)
            plt.plot(dias, confianza, 'b-o', markersize=6, linewidth=2)
            plt.fill_between(dias, confianza, alpha=0.3, color='blue')
            plt.axhline(y=0.7, color='red', linestyle='--', label='Umbral de confianza (70%)')
            plt.title('4. Confianza del Diagnóstico (Probabilidad Máxima de Viterbi)', fontsize=12)
            plt.xlabel('Días de Evolución')
            plt.ylabel('Nivel de Confianza')
            plt.ylim(0, 1.05)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            if guardar:
                plt.savefig('resultados/confianza_diagnostico.png', dpi=150, bbox_inches='tight')
            plt.show()