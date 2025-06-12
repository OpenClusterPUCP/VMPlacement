# 🖥️ VM Placement utilizando MILP 🖥️


Este repositrio implementa un algoritmo de asignación de máquinas virtuales (VMs) a servidores físicos utilizando Programación Lineal Entera Mixta (MILP). El sistema está diseñado para minimzar el número de servidores físicos necesarios para desplegar un conjunto de VMs relacionadas (slice), considerando los perfiles de usuario y recursos disponibles en tiempo real.

---

## 📋 Tabla de Contenidos

- Funcionamiento Principal
- Modelo Matemático
  - Variables de decisión
  - Función objetivo
  - Restricciones
- Implementación Técnica
- Consideraciones adicionales

---

## 🔄 Funcionamiento Principal

El algoritmo funciona de la siguente manera:

1. **Entrada de datos**: Recibe un conjunto de VMs con sus requerimientos (vCPUs, RAM, disco) a través de una API REST.

2. **Ajuste por perfil de usuario**: Aplica factores de uso estimado según el perfil del usuario:
   - 👨‍🎓 **Alumno**: 60%
   - 👨‍🏫 **JP**: 70%
   - 👨‍🔬 **Maestro**: 80%
   - 🧪 **Investigador**: 100%

3. **Validación de recursos disponibles**: Consulta los recursos en tiempo real y verifica si existe capacidad suficiente.

4. **Optimización MILP**: Formula y resuelve el problema de minimización del número de servidores físicos mediante MILP.

5. **Visualización de resultados**: Genera gráficos de asignación y uso de recursos.

---

## 📐 Modelo Matemático

El problema se formula siguiendo estas ecuaciones:

### Variables de decisión

$$x_{i,j} = \begin{cases}
1 & \text{si la VM } i \text{ es asignada al servidor } j \\
0 & \text{en caso contrario}
\end{cases}$$

$$y_{j} = \begin{cases}
1 & \text{si el servidor } j \text{ es utilizado} \\
0 & \text{en caso contrario}
\end{cases}$$

Donde:
- $i = 1, 2, ..., n$ representa las máquinas virtuales
- $j = 1, 2, ..., m$ representa los servidores físicos

### Función objetivo

Minimizar el número de servidores físicos utilizados:

$$\min Z = \sum_{j=1}^{m} y_j$$

### Restricciones

#### ⚠️ Restricción de asignación obligatoria 
Cada VM debe asignarse exactamente a un servidor:

$$\sum_{j=1}^{m} x_{i,j} = 1 \quad \forall i \in \{1,2,...,n\}$$

#### ⚡ Restricción de activación de servidor 
Si una VM es asignada a un servidor, ese servidor debe estar activo:

$$x_{i,j} \leq y_j \quad \forall i \in \{1,2,...,n\}, \forall j \in \{1,2,...,m\}$$

#### 🔢 Restricción de capacidad de vCPUs 
No exceder la capacidad disponible de vCPUs en cada servidor:

$$\sum_{i=1}^{n} vcpu_i \cdot f_p \cdot x_{i,j} \leq CPU_j \cdot y_j \quad \forall j \in \{1,2,...,m\}$$

#### 💾 Restricción de capacidad de RAM 
No exceder el 80% de la capacidad disponible de RAM:

$$\sum_{i=1}^{n} ram_i \cdot f_p \cdot x_{i,j} \leq 0.8 \cdot RAM_j \cdot y_j \quad \forall j \in \{1,2,...,m\}$$

#### 💿 Restricción de capacidad de disco 
No exceder el 80% de la capacidad disponible de disco:

$$\sum_{i=1}^{n} disk_i \cdot f_p \cdot x_{i,j} \leq 0.8 \cdot DISK_j \cdot y_j \quad \forall j \in \{1,2,...,m\}$$

> **Donde:**
>
> - $vcpu_i$, $ram_i$, $disk_i$: Recursos requeridos por la VM $i$
> - $CPU_j$, $RAM_j$, $DISK_j$: Capacidades disponibles del servidor $j$
> - $f_p$: Factor de uso según el perfil del usuario (0.6 para alumno, 0.7 para jp, etc.)
> - El factor 0.8 para RAM y disco representa el límite del 80% para evitar sobrecarga

---

## 💻 Implementación Técnica

El sistema está implementado como una API REST utilizando Flask que ofrece los siguientes endpoints:

| Endpoint | Descripción |
|----------|-------------|
| `/placement` | Resuelve el problema de placement para un slice completo |
| `/test-data` | Genera datos de prueba para facilitar las pruebas |
| `/health` | Verifica el estado del servicio |

Ejemplo de solicitud al endpoint `/placement`:

```json
{
    "slice_id": 123,
    "slice_name": "slice-test",
    "user_profile": "investigador",
    "virtual_machines": [
        {
            "id": 1,
            "name": "VM-1",
            "flavor_id": 1
        },
        {
            "id": 2,
            "name": "VM-2",
            "flavor_id": 2
        }
    ]
}
```

La solución utiliza el solver MILP de SciPy para encontrar la asignación óptima de VMs a servidores. Los resultados se visualizan mediante matplotlib, generando gráficos detallados del uso de recursos y distribución de VMs.

---

## Definición de Función Q (Tiempo relativo de espera en cola para el slice)

La función $$( Q_i $$) representa el tiempo relativo de espera en cola para un slice, basado en la congestión del servidor. Se define mediante una función sigmoide suave:

$$
Q_i = \frac{1}{1 + e^{-a(c_{vcpu} - b)}}
$$

Donde:

- $$( c_{vcpu} $$): congestión del servidor (valor entre 0 y 1)
- $$( a = 12 $$): pendiente de la curva
- $$( b = 0.7 $$): punto de inflexión (~70% de congestión)

Esta fórmula permite una transición progresiva y continua en la estimación del tiempo de espera, evitando saltos abruptos cuando la congestión aumenta.

🔧 Con **a = 12**, se logra:

- Q ≈ 0.05 cuando congestión = 0.4
- Q ≈ 0.5 cuando congestión = 0.7
- Q ≈ 0.95 cuando congestión = 1.0

Gráfico:

![Captura de pantalla 2025-06-12 175449](https://github.com/user-attachments/assets/854a939d-6a03-4032-a36a-865b24a02465)

Esta curva da una transición suave entre baja congestión (espera casi nula) y alta cogestión (espera significativa)

Comportamiento Realista

| Congestión $c_{vcpu}$ | Interpetación de $Q_i$                  |
| ------------------- | --------------------------------------- |
| Cerca de 0          | Tiempo de espera ≈ 0 (servidor libre)   |
| Alrededor de 0.7    | Tiempo de espera ≈ 0.5 (zona crítica)   |
| Cerca de 1          | Tiempo de espera ≈ 1 (saturación total) |

## ⚙️ Consideraciones adicionales

- 🔄 Los recursos en tiempo real se obtienen mediante una API externa (simulada en esta versión)
- 🛡️ Se limita el uso de RAM y disco al 80% para mantener un margen de seguridad
- 👤 El perfil de usuario permite estimar el uso real de recursos durante la vida del slice
- 📊 La visualización genera gráficos detallados que facilitan el análisis de las asignaciones

---
