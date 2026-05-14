from src.hmm_model import HMMMedico
from src.simulaciones import (
    ejecutar_estudio_simulacion,
    mostrar_estadisticas,
    mostrar_caso_representativo
)
from config import NUM_SIMULACIONES, DIAS_POR_PACIENTE

def main():
    print("\n" + "=" * 60)
    print("🏥 MODELO HMM PARA DIAGNÓSTICO MÉDICO")
    print("=" * 60)
    
    resultados, precisiones = ejecutar_estudio_simulacion(
        num_simulaciones=NUM_SIMULACIONES,
        dias=DIAS_POR_PACIENTE
    )
    
    mostrar_estadisticas(precisiones, resultados)
    mostrar_caso_representativo(resultados, precisiones)
    
    print("\n✅ ESTUDIO COMPLETADO")

if __name__ == "__main__":
    main()