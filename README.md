# 🏥 Modelo Oculto de Markov para Diagnóstico Médico

## 📋 Descripción del Proyecto

Este proyecto implementa un **Modelo Oculto de Markov (HMM)** para simular la evolución de pacientes con enfermedades respiratorias (resfriado y gripe) y evaluar la precisión del diagnóstico basado únicamente en síntomas observables (temperatura corporal).

### Estados del modelo

| Estado | Descripción |
|--------|-------------|
| Saludable | Sin síntomas |
| Resfriado | Síntomas leves |
| Gripe | Síntomas severos |

### Observaciones

| Observación | Descripción |
|-------------|-------------|
| Temperatura Normal | < 37.5°C |
| Fiebre Leve | 37.5°C - 38.5°C |
| Fiebre Alta | > 38.5°C |

## 📁 Estructura del Proyecto
