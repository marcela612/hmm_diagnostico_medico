"""
Módulo de simulaciones para el HMM Médico.
Ejecuta estudios clínicos con múltiples pacientes y genera estadísticas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .hmm_model import HMMMedico


def ejecutar_estudio_simulacion(num_simulaciones=30, dias=20, verbose=True):
    """
    Ejecuta múltiples simulaciones de pacientes.
    
    Parámetros:
        num_simulaciones (int): Número de pacientes a simular
        dias (int): Días de evolución por paciente
        verbose (bool): Muestra progreso
    
    Retorna:
        tuple: (resultados, precisiones)
    """
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
    """
    Muestra estadísticas globales del estudio clínico.
    """
    print("\n" + "=" * 60)
    print("RESULTADOS DEL ESTUDIO CLÍNICO")
    print("=" * 60)
    
    print(f"\n📊 PRECISIÓN DIAGNÓSTICA:")
    print(f"  Promedio:   {np.mean(precisiones)*100:.1f}%")
    print(f"  Máxima:     {np.max(precisiones)*100:.1f}%")
    print(f"  Mínima:     {np.min(precisiones)*100:.1f}%")
    print(f"  Desviación: {np.std(precisiones)*100:.1f}%")
    
    # Distribución de días por condición
    total_dias = len(resultados) * 20
    saludable = sum(np.sum(r['estados_reales'] == 0) for r in resultados)
    resfriado = sum(np.sum(r['estados_reales'] == 1) for r in resultados)
    gripe = sum(np.sum(r['estados_reales'] == 2) for r in resultados)
    
    print(f"\n📈 DÍAS POR CONDICIÓN (total {total_dias} días):")
    print(f"  Saludable:  {saludable} ({saludable/total_dias*100:.1f}%)")
    print(f"  Resfriado:  {resfriado} ({resfriado/total_dias*100:.1f}%)")
    print(f"  Gripe:      {gripe} ({gripe/total_dias*100:.1f}%)")
    
    # Histograma de precisiones
    plt.figure(figsize=(10, 5))
    plt.hist(precisiones, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(np.mean(precisiones), color='red', linestyle='--', linewidth=2,
                label=f'Media: {np.mean(precisiones)*100:.1f}%')
    plt.axvline(np.median(precisiones), color='green', linestyle='--', linewidth=2,
                label=f'Mediana: {np.median(precisiones)*100:.1f}%')
    plt.xlabel('Precisión')
    plt.ylabel('Frecuencia (número de pacientes)')
    plt.title('Distribución de Precisión Diagnóstica en el Estudio Clínico')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('resultados/histograma_precisiones.png', dpi=150, bbox_inches='tight')
    plt.show()


def mostrar_matriz_confusion_global(resultados):
    """
    Calcula y muestra la matriz de confusión global del estudio.
    
    Parámetros:
        resultados (list): Lista de resultados de simulación
    """
    modelo = HMMMedico()
    todos_reales = []
    todos_inferidos = []
    
    for r in resultados:
        todos_reales.extend(r['estados_reales'])
        todos_inferidos.extend(r['estados_inferidos'])
    
    cm = confusion_matrix(todos_reales, todos_inferidos)
    
    print("\n" + "=" * 60)
    print("MATRIZ DE CONFUSIÓN GLOBAL")
    print("=" * 60)
    print("(Filas = Estado Real, Columnas = Estado Inferido)")
    print()
    
    print(f"{'':<12} {'Saludable':<12} {'Resfriado':<12} {'Gripe':<12}")
    print(f"{'Saludable':<12} {cm[0,0]:<12} {cm[0,1]:<12} {cm[0,2]:<12}")
    print(f"{'Resfriado':<12} {cm[1,0]:<12} {cm[1,1]:<12} {cm[1,2]:<12}")
    print(f"{'Gripe':<12} {cm[2,0]:<12} {cm[2,1]:<12} {cm[2,2]:<12}")
    
    # Sensibilidad por clase
    print("\n--- Sensibilidad por enfermedad (capacidad de detectar cada condición) ---")
    for i, estado in enumerate(['Saludable', 'Resfriado', 'Gripe']):
        if np.sum(cm[i, :]) > 0:
            sensibilidad = cm[i, i] / np.sum(cm[i, :])
            print(f"  {estado}: {sensibilidad:.2%}")
    
    # Especificidad por clase
    print("\n--- Especificidad por enfermedad (capacidad de descartar cada condición) ---")
    for i, estado in enumerate(['Saludable', 'Resfriado', 'Gripe']):
        # Verdaderos negativos = total de otras clases correctamente clasificadas
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"  {estado}: {especificidad:.2%}")
    
    return cm


def mostrar_tabla_primeros_dias(caso, n_dias=10):
    """
    Muestra una tabla con los primeros n días del caso representativo.
    
    Parámetros:
        caso (dict): Resultado de un paciente
        n_dias (int): Número de días a mostrar
    """
    modelo = HMMMedico()
    
    print("\n" + "=" * 60)
    print(f"📋 TABLA DE RESULTADOS (Primeros {n_dias} días del Paciente {caso['paciente_id']})")
    print("=" * 60)
    print(f"{'Día':<6} {'Estado Real':<15} {'Observación':<18} {'Estado Inferido':<15} {'Acierto?':<8}")
    print("-" * 75)
    
    aciertos = 0
    for i in range(min(n_dias, len(caso['estados_reales']))):
        acierto = caso['estados_reales'][i] == caso['estados_inferidos'][i]
        if acierto:
            aciertos += 1
        acierto_str = "✓" if acierto else "✗"
        
        print(f"{i+1:<6} {modelo.estados[caso['estados_reales'][i]]:<15} "
              f"{modelo.observaciones[caso['observaciones'][i]]:<18} "
              f"{modelo.estados[caso['estados_inferidos'][i]]:<15} {acierto_str:<8}")
    
    print("-" * 75)
    print(f"Precisión en primeros {n_dias} días: {aciertos/n_dias:.1%}")


def mostrar_caso_representativo(resultados, precisiones):
    """
    Muestra un caso representativo (cercano a la precisión media).
    
    Parámetros:
        resultados (list): Lista de resultados de simulación
        precisiones (list): Lista de precisiones
    
    Retorna:
        dict: Caso representativo seleccionado
    """
    modelo = HMMMedico()
    media = np.mean(precisiones)
    idx = np.argmin(np.abs(np.array(precisiones) - media))
    caso = resultados[idx]
    
    print("\n" + "=" * 60)
    print(f"🔍 CASO REPRESENTATIVO (Paciente {caso['paciente_id']})")
    print("=" * 60)
    print(f"  Precisión: {caso['precision']*100:.1f}% (cercana a la media del estudio)")
    
    # Mostrar tabla de primeros días
    mostrar_tabla_primeros_dias(caso, n_dias=10)
    
    # Recomendación médica
    print(f"\n💊 RECOMENDACIÓN MÉDICA:")
    print(f"  {modelo.recomendacion_medica(caso['estados_inferidos'][-1])}")
    
    # Generar gráficos
    print(f"\n📊 Generando gráficos del caso representativo...")
    modelo.graficar_simulacion(
        caso['estados_reales'],
        caso['observaciones'],
        caso['estados_inferidos'],
        caso['V'],
        guardar=True
    )
    
    return caso