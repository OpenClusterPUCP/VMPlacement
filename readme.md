# Algoritmos de VM Placement para Entornos Cloud

Explicación de las ideas/borradores para el VM placement (aun no se usan la data real del módulo de monitoreo).

## 🚀 VM Placement de Santi "nivel kinder" (`vm_placement.py`)

### Enfoque y Fundamentos

El algoritmo de VM Placement implementa una solucion basada en **Programación Lineal Entera Mixta (MILP)** para asignar máquinas virtuales individuales a servidores físicos de manera óptima. Este enfoque matemático garantiza encontrar la solución más eficiente posible dentro de las restricciones establecidas.

### Funcionamiento del Algoritmo

1. **Definición de la Función Objetivo**:
   - Maximiza la utilidad total de las asignaciones
   - La utilidad se calcula considerando los recursos (vCPUs, RAM, disco) y puede incluir factores adicionales como rol de usuario o prioridad

2. **Restricciones del Modelo**:
   - Cada VM puede asignarse a máximo un servidor
   - No exceder los límites de vCPUs disponibles en cada servidor
   - No exceder los límites de RAM disponible en cada servidor
   - No exceder los límites de disco disponible en cada servidor

3. **Proceso de Resolución**:
   - Formulación matematica mediante matrices de coeficientes y restricciones
   - Resolución usando el optimizador MILP de SciPy
   - Extracción y validación de la solución generada

### Consideraciones de Rendimiento

- **Recursos Disponibles**: Evalúa los recursos actualmente disponibles en cada servidor (disponible = total - usado)
- **Criterio de Rechazo**: Si los recursos totales no son suficientes, rechaza la solicitud completa
- **Detalle de Fallos**: Genera información detallada sobre las razones por las que una VM no puede ser asignada

### Limitaciones

Este algoritmo trata cada VM de manera individual, sin considerar relaciones entre ellas, lo que puede resultar en una fragmentacion de VMs relacionadas entre diferentes servidores. No considera el concepto de "slices" o grupos de VMs que deberian permanecer juntas para un mejor rendimiento.

## 🔄 Slice-Based Placement (`slice_placement.py`)

### Enfoque y Fundamentos

El algoritmo de Slice-Based Placement implementa una estrategia **"Cluster-First"** diseñada específicamente para optimizar la asignación de grupos relacionados de VMs (slices) a servidores físicos, priorizando la localidad de las VMs y la gestión eficiente de recursos considerando el ciclo de vida del slice.

### Funcionamiento del Algoritmo

1. **Evaluación de Capacidad**:
   - Analiza si algún servidor individual puede alojar el slice completo
   - Considera límites de sobreaprovisionamiento específicos para el perfil del usuario

2. **Estrategia Cluster-First**:
   - Identifica el servidor que puede alojar la mayor cantidad de VMs del slice
   - Prioriza mantener juntas las VMs relacionadas en un mismo servidor
   - Distribuye las VMs restantes utilizando el mínimo número de servidores adicionales

3. **Gestión Avanzada de Recursos**:
   - Considera el uso estimado real basado en perfiles de usuario (alumno, JP, maestro, investigador)
   - Aplica factores de variabilidad para anticipar picos durante el ciclo de vida
   - Implemnta límites de sobreaprovisionamiento diferenciados según el SLA

### Consideraciones de Rendimiento y SLA

- **Sobreaprovisionamiento (falta ajustarlo, se puso loca la cosa)**:

- **Estimacion de Congestión**:
  - Calcula la congestión estimada para cada tipo de recurso tras asignar el slice
  - Estima tiempos de espera en cola utilizando un modelo no lineal
  - Considera el rendimiento relativo de cada servidor

- **Criterios de Éxito**:
  - Maximizar la localidad (mayor cantidad de VMs en un solo servidor)
  - Minimizar el número de servidores utilizados
  - Garantizar que los recuros reales necesarios no excedan límites seguros

### Ventajas Clave

- ✅ **Tratamiento de Slices como Unidad**: Reconoce y mantiene la relación entre VMs del mismo slice
- ✅ **Estimacion de Uso Real**: No se basa únicamente en los recursos solicitados por el flavor sino en estimaciones de uso real
- ✅ **Gestión del Ciclo de Vida**: Anticipa la variabilidad del uso de recursos durante la vida del slice
- ✅ **Informes Detallados de Fallos**: Proporciona explicaciones precisas sobre por que no se puede asignar un slice

Este algoritmo resulta especialmente adecuado para entornos donde la localidad entre VMs relacionadas es crítica y donde los patrones de uso son variables según el tipo de usuario y carga de trabajo.

## 📊 Visualizaciones y Análisis

Ambos algoritmos incluyen capacidades de visualizacion que permiten:

- Análisis gráfico del uso de recursos por servidor
- Distribución de VMs en los servidores
- Tablas detalladas de asignaciones
- Exportación de resultados a imágenes para informes

## 📈 Cumplimiento de Requisitos

| Requisito | Implementación |
|-----------|---------------|
| **Recursos disponibles y asignados** | Ambos algoritmos consideran los recursos actualmente disponibles y ya asignados en cada servidor para evitar congestión. |
| **Función objetivo bien definida** | El algoritmo MILP maximiza la utilidad total, mientras que el Slice-Based maximiza la localidad y minimiza servidores. |
| **Slices como unidad** | El algoritmo Slice-Based trata todo el conjunto de VMs como una unidad cohesiva, manteniendo las VMs relacionadas en el mismo servidor cuando es posible. |
| **Ciclo de vida y congestión** | Se implementan factores de variabilidad y perfiles de usuario que anticipan la congestión durante el ciclo de vida completo del slice. |
| **Asignación segun disponibilidad** | Ambos algoritmos realizan una asignación óptima cuando hay capacidad disponible. |
| **Rechazo con explicación** | Cuando no hay recursos suficientes, se genera un informe detallado explicando las razones específicas del rechazo. |

## 🔍 Casos de Uso Recomendados

- **VM Placement (MILP)**: Ideal para entornos heterogéneos donde la optimizacion individual de recursos es prioritaria.
- **Slice-Based Placement**: Recomendado para aplicaciones distribuidas donde la localidad entre VMs relacionadas mejora significativamente el rendimiento.

---
