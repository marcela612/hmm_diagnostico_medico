import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .hmm_model import HMMMedico

def ejecutar_estudio_simulacion(num_simulaciones=30, dias=20, verbose=True):
    modelo = HMMMedico()
    resultados = []
    precisiones = []
    
    print("=" * 60)
    print(f"INICIANDO ESTUDIO CLÍNICO - {num_simulaciones} pacientes")
    print("=" * 60)
    
    for i in range(num_simulaciones):
        reales, obs = modelo.simular_paciente(dias)
        inferidos, V = modelo.viterbi(obs)
        precision = modelo.calcular_precision(reales, inferidos)
        precisiones.append(precision)
        
        resultados.append({
            'paciente_id': i + 1,
            'estados_reales': reales,
            'observaciones': obs,
            'estados_inferidos': inferidos,
            'precision': precision,
            'V': V
        })
        
        if verbose and (i + 1) % 5 == 0:
            print(f"  Simulados {i+1}/{num_simulaciones} pacientes...")
    
    print(f"\n✅ Precisión promedio: {np.mean(precisiones)*100:.1f}%")
    return resultados, precisiones

def mostrar_estadisticas(precisiones, resultados):
    print("\n" + "=" * 60)
    print("RESULTADOS DEL ESTUDIO")
    print("=" * 60)
    
    print(f"\nPRECISIÓN DIAGNÓSTICA:")
    print(f"  Promedio:   {np.mean(precisiones)*100:.1f}%")
    print(f"  Máxima:     {np.max(precisiones)*100:.1f}%")
    print(f"  Mínima:     {np.min(precisiones)*100:.1f}%")
    
    total_dias = len(resultados) * 20
    saludable = sum(np.sum(r['estados_reales'] == 0) for r in resultados)
    resfriado = sum(np.sum(r['estados_reales'] == 1) for r in resultados)
    gripe = sum(np.sum(r['estados_reales'] == 2) for r in resultados)
    
    print(f"\nDÍAS POR CONDICIÓN:")
    print(f"  Saludable:  {saludable} ({saludable/total_dias*100:.1f}%)")
    print(f"  Resfriado:  {resfriado} ({resfriado/total_dias*100:.1f}%)")
    print(f"  Gripe:      {gripe} ({gripe/total_dias*100:.1f}%)")
    
    plt.figure(figsize=(10, 5))
    plt.hist(precisiones, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(np.mean(precisiones), color='red', linestyle='--', 
                label=f'Media: {np.mean(precisiones)*100:.1f}%')
    plt.xlabel('Precisión')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Precisión Diagnóstica')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def mostrar_caso_representativo(resultados, precisiones):
    modelo = HMMMedico()
    media = np.mean(precisiones)
    idx = np.argmin(np.abs(np.array(precisiones) - media))
    caso = resultados[idx]
    
    print("\n" + "=" * 60)
    print(f"CASO REPRESENTATIVO (Paciente {caso['paciente_id']})")
    print("=" * 60)
    print(f"  Precisión: {caso['precision']*100:.1f}%")
    
    print(f"\n  Evolución real (días 1-10):")
    reales_str = [modelo.estados[e] for e in caso['estados_reales'][:10]]
    print(f"    {reales_str}")
    
    print(f"\n  Diagnóstico (días 1-10):")
    inferidos_str = [modelo.estados[e] for e in caso['estados_inferidos'][:10]]
    print(f"    {inferidos_str}")
    
    print(f"\n  Recomendación: {modelo.recomendacion_medica(caso['estados_inferidos'][-1])}")
    
    print(f"\n  Generando gráficos...")
    modelo.graficar_simulacion(
        caso['estados_reales'],
        caso['observaciones'],
        caso['estados_inferidos'],
        caso['V']
    )
    
    return caso