# 🩺 Modelo Oculto de Markov para Diagnóstico Médico Asistido

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Descripción del Proyecto

Este proyecto implementa un **Modelo Oculto de Markov (HMM)** para simular el diagnóstico médico de un paciente a lo largo de 20 días. El sistema infiere el estado de salud no observable (*Saludable*, *Resfriado* o *Gripe*) a partir de observaciones diarias de temperatura (*Normal*, *Fiebre Leve*, *Fiebre Alta*).

## 🎯 Objetivo

Demostrar cómo los HMM pueden modelar procesos clínicos con información parcialmente observable, generando diagnósticos automáticos con métricas de confianza mediante el algoritmo de Viterbi.

## 🏥 Planteamiento del Problema

### Contexto clínico
En la práctica médica real, el estado exacto de salud de un paciente no es directamente observable sin pruebas de laboratorio costosas. Sin embargo, el médico puede registrar síntomas observables como la temperatura corporal.

### Pregunta de investigación
**¿Qué tan bien puede un HMM inferir el estado de salud oculto basándose únicamente en observaciones de temperatura, y qué métricas clínicas útiles puede generar?**

### Justificación del uso de HMM
- Los estados de salud evolucionan con dependencia temporal (propiedad de Markov)
- Existe una relación probabilística entre estados ocultos (enfermedad) y observaciones (síntomas)
- El algoritmo de Viterbi encuentra la secuencia más probable de estados

## 📊 Modelo Matemático

### Estados ocultos (S)
| Índice | Estado | Descripción |
|--------|--------|-------------|
| 0 | Saludable | Sin síntomas, temperatura normal |
| 1 | Resfriado | Infección leve, posible fiebre baja |
| 2 | Gripe | Infección grave, fiebre alta |

### Observaciones (O)
| Índice | Observación | Criterio clínico |
|--------|-------------|------------------|
| 0 | Temperatura Normal | ≤ 37.2°C |
| 1 | Fiebre Leve | 37.3°C - 38.5°C |
| 2 | Fiebre Alta | > 38.5°C |

### Matriz de Transición (A)
*Probabilidad de cambiar de un estado a otro al día siguiente*

| Desde \ Hacia | Saludable | Resfriado | Gripe |
|---------------|-----------|-----------|-------|
| **Saludable** | 0.80 | 0.15 | 0.05 |
| **Resfriado** | 0.40 | 0.40 | 0.20 |
| **Gripe** | 0.20 | 0.30 | 0.50 |

**Justificación clínica:**
- Saludable → Saludable: 80% (se mantiene bien)
- Saludable → Resfriado: 15% (se resfría)
- Saludable → Gripe: 5% (gripe directa)
- Resfriado → Saludable: 40% (se recupera)
- Resfriado → Resfriado: 40% (persiste)
- Resfriado → Gripe: 20% (empeora)
- Gripe → Saludable: 20% (recuperación total)
- Gripe → Resfriado: 30% (mejora parcial)
- Gripe → Gripe: 50% (persiste)

### Matriz de Emisión (B)
*Probabilidad de observar cada temperatura dado el estado*

| Estado \ Observación | Normal | Fiebre Leve | Fiebre Alta |
|---------------------|--------|-------------|-------------|
| **Saludable** | 0.90 | 0.08 | 0.02 |
| **Resfriado** | 0.30 | 0.50 | 0.20 |
| **Gripe** | 0.10 | 0.30 | 0.60 |

**Justificación clínica:**
- Saludable: 90% normal (ruido del 10% por ejercicio/medición)
- Resfriado: pico en fiebre leve (50%)
- Gripe: predominio de fiebre alta (60%)

### Probabilidades Iniciales (π)