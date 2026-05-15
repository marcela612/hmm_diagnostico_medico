"""
Configuración del proyecto HMM Médico
"""

# Parámetros del estudio clínico
NUM_SIMULACIONES = 30      # Número de pacientes a simular
DIAS_POR_PACIENTE = 20     # Días de evolución por paciente
VERBOSE = True             # Mostrar progreso durante la simulación

# Semilla aleatoria para resultados reproducibles (opcional)
SEMILLA_ALEATORIA = 42
# import numpy as np
# np.random.seed(SEMILLA_ALEATORIA)

# Rutas del proyecto
RESULTADOS_DIR = 'resultados'
DOCS_DIR = 'docs'
DATA_DIR = 'data'