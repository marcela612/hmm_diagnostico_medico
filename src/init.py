"""
src - Paquete principal del proyecto HMM Médico
"""

from .hmm_model import HMMMedico
from .simulaciones import (
    ejecutar_estudio_simulacion,
    mostrar_estadisticas,
    mostrar_matriz_confusion_global,
    mostrar_caso_representativo
)

__all__ = [
    'HMMMedico',
    'ejecutar_estudio_simulacion',
    'mostrar_estadisticas',
    'mostrar_matriz_confusion_global',
    'mostrar_caso_representativo'
]