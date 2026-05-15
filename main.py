"""
================================================================================
PROGRAMA PRINCIPAL - HMM PARA DIAGNÓSTICO MÉDICO
================================================================================
Ejecuta el estudio clínico completo:
1. Simula múltiples pacientes
2. Calcula estadísticas globales
3. Muestra matriz de confusión
4. Presenta un caso representativo con gráficos
"""

from src.hmm_model import HMMMedico
from src.simulaciones import (
    ejecutar_estudio_simulacion,
    mostrar_estadisticas,
    mostrar_matriz_confusion_global,
    mostrar_caso_representativo
)
from config import NUM_SIMULACIONES, DIAS_POR_PACIENTE


def main():
    """
    Función principal que ejecuta el estudio clínico completo.
    """
    print("\n" + "=" * 60)
    print("🏥 MODELO HMM PARA DIAGNÓSTICO MÉDICO ASISTIDO")
    print("=" * 60)
    print("\n📌 Algoritmo de Viterbi para inferencia de estados ocultos")
    print("📌 Basado en observaciones de temperatura corporal")
    
    # 1. Ejecutar estudio clínico
    print("\n" + "-" * 60)
    resultados, precisiones = ejecutar_estudio_simulacion(
        num_simulaciones=NUM_SIMULACIONES,
        dias=DIAS_POR_PACIENTE,
        verbose=True
    )
    
    # 2. Mostrar estadísticas globales
    mostrar_estadisticas(precisiones, resultados)
    
    # 3. Mostrar matriz de confusión global
    mostrar_matriz_confusion_global(resultados)
    
    # 4. Mostrar caso representativo (cercano a la media)
    caso = mostrar_caso_representativo(resultados, precisiones)
    
    # 5. Resumen final
    print("\n" + "=" * 60)
    print("✅ ESTUDIO CLÍNICO COMPLETADO")
    print("=" * 60)
    print(f"📊 Resumen final:")
    print(f"   - Pacientes simulados: {NUM_SIMULACIONES}")
    print(f"   - Días por paciente: {DIAS_POR_PACIENTE}")
    print(f"   - Precisión promedio: {np.mean(precisiones)*100:.1f}%")
    print(f"   - Gráficos guardados en: resultados/")
    print("=" * 60)


if __name__ == "__main__":
    import numpy as np
    main()