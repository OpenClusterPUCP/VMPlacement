# ==============================================================================
# | ARCHIVO: slice_placement.py
# ==============================================================================
# | DESCRIPCIÓN:

# | Módulo API REST que implementa un algoritmo de asignación de conjuntos de 
# | máquinas virtuales (slices) a servidores físicos, optimizando la localidad
# | y distribución de recursos. Implementa un enfoque de cluster-first para
# | maximizar la coubicación de VMs relacionadas y gestionar el sobreaprovisionamiento.
# ==============================================================================
# | CONTENIDO PRINCIPAL:
# | 1. CONFIGURACIÓN INICIAL
# |    - Importaciones y configuración Flask
# |    - Configuración de base de datos MySQL
# |    - Pool de conexiones y transacciones
# |    - Logger personalizado para debugging
# |
# | 2. MODELOS DE DATOS
# |    - Flavor: Representa una configuración de recursos para VM
# |    - UserProfile: Define perfiles de usuario con patrones de consumo
# |    - WorkloadType: Define tipos de cargas de trabajo
# |    - VirtualMachine: Representa una máquina virtual con su flavor
# |    - Slice: Representa un conjunto de VMs relacionadas
# |    - PhysicalServer: Representa un servidor físico con capacidades
# |    - PlacementResult: Resultado de la colocación de slices
# |
# | 3. SUBMÓDULOS PRINCIPALES
# |    - DatabaseManager: Gestiona conexiones y consultas a la BD
# |    - SliceBasedPlacementSolver: Resuelve el placement con enfoque de slice
# |    - DataManager: Conversión entre formatos de datos
# |
# | 4. ALGORITMO DE PLACEMENT
# |    - Evaluación de ajuste óptimo para cada slice
# |    - Sobreaprovisionamiento controlado según perfil de usuario
# |    - Enfoque cluster-first para maximizar localidad
# |    - Distribución en múltiples servidores cuando es necesario
# |
# | 5. VISUALIZACIÓN DE RESULTADOS
# |    - Generación de gráficas de uso de recursos por servidor
# |    - Visualización de asignaciones VM-servidor 
# |    - Tabla detallada de VMs asignadas
# |    - Exportación a imágenes en carpeta 'resultados'
# |
# | 6. API ENDPOINTS
# |    - /health: Verificación del servicio
# |    - /placement: Endpoint para resolver el placement de slices
# |    - /test-data: Generación de datos de prueba adaptados a slices
# ==============================================================================

# ===================== IMPORTACIONES =====================
from flask import Flask, request, jsonify

# Sistema:
import os
import traceback

# Utilidades:
import json
import random
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Union

# Bibliotecas científicas:
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds

# Acceso a datos:
import mysql.connector
from mysql.connector import pooling

# ===================== CONFIGURACIÓN DE FLASK =====================
app = Flask(__name__)
host = '0.0.0.0'
port = 6001
debug = False

# ===================== CONFIGURACIÓN BD =====================
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "port": 3306,
    "database": "cloud_v3"
}

POOL_CONFIG = {
    "pool_name": "cloudpool",
    "pool_size": 5,
    **DB_CONFIG
}

class DatabaseManager:
    """
    Gestiona las conexiones y consultas a la base de datos MySQL
    """
    _pool = None
    
    @classmethod
    def init_pool(cls):
        """Inicializa el pool de conexiones si no existe"""
        if cls._pool is None:
            try:
                cls._pool = mysql.connector.pooling.MySQLConnectionPool(**POOL_CONFIG)
                Logger.success("Pool de conexiones a la base de datos inicializado correctamente")
            except Exception as e:
                Logger.error(f"Error al inicializar el pool de conexiones: {str(e)}")
                raise
    
    @classmethod
    def get_connection(cls):
        """Obtiene una conexión del pool"""
        if cls._pool is None:
            cls.init_pool()
        return cls._pool.get_connection()
    
    @classmethod
    def execute_query(cls, query, params=None, fetch=True):
        """Ejecuta una consulta SQL y devuelve los resultados"""
        connection = None
        try:
            connection = cls.get_connection()
            cursor = connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            
            if fetch:
                result = cursor.fetchall()
            else:
                connection.commit()
                result = cursor.rowcount
                
            cursor.close()
            return result
        except Exception as e:
            Logger.error(f"Error en consulta SQL: {str(e)}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                connection.close()
    
    @classmethod
    def get_active_flavors(cls):
        """Obtiene todos los flavors activos de la base de datos"""
        query = """
            SELECT id, name, vcpus, ram, disk 
            FROM flavor 
            WHERE state = 'active'
            ORDER BY id
        """
        try:
            flavors = cls.execute_query(query)
            Logger.info(f"Se obtuvieron {len(flavors)} flavors activos desde la BD")
            return flavors
        except Exception as e:
            Logger.error(f"Error al obtener flavors activos: {str(e)}")
            return []
    
    @classmethod
    def get_physical_servers(cls):
        """
        Obtiene todos los servidores físicos disponibles
        NOTA: Se generan valores aleatorios para used_vcpus, used_ram y used_disk
        cuando son 0 - SOLO PARA DEMOSTRACIÓN (no exceden el 60% del total)
        """
        query = """
            SELECT 
                id, 
                name, 
                total_vcpu as total_vcpus, 
                total_ram, 
                total_disk,
                used_vcpu as used_vcpus,
                used_ram,
                used_disk
            FROM physical_server
            WHERE server_type = 'worker' AND status = 'active'
            ORDER BY id
        """
        try:
            servers = cls.execute_query(query)
            
            # IMPORTANTE: Solo para demostración, generamos valores aleatorios 
            # para los campos que tienen 0 o 0.0 (máx 60% del total)
            import random
            for server in servers:
                # Si used_vcpus es 0, generamos un valor aleatorio (máx 60% del total)
                if server['used_vcpus'] == 0:
                    # Calcular máximo exacto (60% del total)
                    max_used = max(1, int(server['total_vcpus'] * 0.6))
                    server['used_vcpus'] = random.randint(1, max_used)
                    Logger.warning(f"Server {server['name']}: Valor used_vcpus asignado provisionalmente: {server['used_vcpus']} (máx 60% de {server['total_vcpus']})")
                
                # Si used_ram es 0, generamos un valor aleatorio (máx 60% del total)
                if server['used_ram'] == 0:
                    # Calcular máximo exacto (60% del total)
                    max_used = max(1, int(server['total_ram'] * 0.6))
                    server['used_ram'] = random.randint(1, max_used)
                    Logger.warning(f"Server {server['name']}: Valor used_ram asignado provisionalmente: {server['used_ram']} (máx 60% de {server['total_ram']})")
                
                # Si used_disk es 0.0, generamos un valor aleatorio (máx 60% del total)
                if float(server['used_disk']) == 0.0:
                    # Calcular máximo exacto (60% del total)
                    max_used = max(1.0, float(server['total_disk']) * 0.6)
                    server['used_disk'] = round(random.uniform(1.0, max_used), 1)
                    Logger.warning(f"Server {server['name']}: Valor used_disk asignado provisionalmente: {server['used_disk']} (máx 60% de {server['total_disk']})")
                
            Logger.info(f"Se obtuvieron {len(servers)} servidores físicos desde la BD")
            return servers
        except Exception as e:
            Logger.error(f"Error al obtener servidores físicos: {str(e)}")
            return []

# ===================== UTILIDADES =====================
class Logger:
    """Clase para manejar el formato y presentación de logs del sistema"""

    # Colores ANSI
    HEADER = '\033[95m'    # Morado
    BLUE = '\033[94m'      # Azul
    GREEN = '\033[92m'     # Verde
    YELLOW = '\033[93m'    # Amarillo
    RED = '\033[91m'       # Rojo
    CYAN = '\033[96m'      # Cyan
    WHITE = '\033[97m'     # Blanco brillante
    ENDC = '\033[0m'       # Reset color
    BOLD = '\033[1m'       # Negrita
    DIM = '\033[2m'        # Tenue

    @staticmethod
    def _get_timestamp():
        """Retorna el timestamp actual en formato legible"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def major_section(title):
        """Imprime una sección principal con timestamp y separadores grandes"""
        timestamp = Logger._get_timestamp()
        print(f"\n{Logger.HEADER}{Logger.BOLD}")
        print("═" * 100)
        print(f"║ {Logger.CYAN}{timestamp}{Logger.HEADER} ║ {Logger.WHITE}{title}")
        print("═" * 100)
        print(f"{Logger.ENDC}")

    @staticmethod
    def section(title):
        """Imprime una sección con separadores medianos"""
        timestamp = Logger._get_timestamp()
        print(f"\n{Logger.BLUE}{Logger.BOLD}")
        print("─" * 100)
        print(f"┌─[{Logger.CYAN}{timestamp}{Logger.BLUE}]")
        print(f"└─▶ {Logger.WHITE}{title}")
        print("─" * 100)
        print(f"{Logger.ENDC}")

    @staticmethod
    def subsection(title):
        """Imprime una subsección con separadores pequeños"""
        timestamp = Logger._get_timestamp()
        print(f"\n{Logger.BLUE}{Logger.BOLD}")
        print("·" * 80)
        print(f"▶ [{timestamp}{Logger.BLUE}] {Logger.WHITE}{title}")
        print("·" * 80)
        print(f"{Logger.ENDC}")

    @staticmethod
    def info(message):
        """Imprime mensaje informativo con timestamp"""
        timestamp = Logger._get_timestamp()
        print(f"{Logger.GREEN}[{Logger.DIM}{timestamp}{Logger.GREEN}] ℹ {message}{Logger.ENDC}")

    @staticmethod
    def warning(message):
        """Imprime advertencia con timestamp"""
        timestamp = Logger._get_timestamp()
        print(f"{Logger.YELLOW}[{Logger.DIM}{timestamp}{Logger.YELLOW}] ⚠ {message}{Logger.ENDC}")

    @staticmethod
    def error(message):
        """Imprime error con timestamp"""
        timestamp = Logger._get_timestamp()
        print(f"{Logger.RED}[{Logger.DIM}{timestamp}{Logger.RED}] ✗ {message}{Logger.ENDC}")

    @staticmethod
    def debug(message):
        """Imprime mensaje de debug con timestamp"""
        timestamp = Logger._get_timestamp()
        print(f"{Logger.BLUE}[{Logger.DIM}{timestamp}{Logger.BLUE}] 🔍 {message}{Logger.ENDC}")

    @staticmethod
    def success(message):
        """Imprime mensaje de éxito con timestamp"""
        timestamp = Logger._get_timestamp()
        print(f"{Logger.GREEN}[{Logger.DIM}{timestamp}{Logger.GREEN}] ✓ {message}{Logger.ENDC}")

    @staticmethod
    def failed(message):
        """Imprime mensaje de fallo con timestamp"""
        timestamp = Logger._get_timestamp()
        print(f"{Logger.RED}[{Logger.DIM}{timestamp}{Logger.RED}] ⨯ {message}{Logger.ENDC}")

# ===================== MODELOS DE DATOS =====================
@dataclass
class Flavor:
    """
    Representa una configuración de recursos para una VM
    """
    id: int
    name: str
    vcpus: int
    ram: int  # En MB (entero en BD)
    disk: float  # En GB (decimal en BD con 1 decimal)
    
    def __str__(self):
        return f"Flavor {self.id}: {self.name} (vCPUs: {self.vcpus}, RAM: {self.ram} MB, Disk: {self.disk:.1f} GB)"

@dataclass
class UserProfile:
    """Define perfiles de usuario con diferentes patrones de consumo de recursos"""
    ALUMNO = "alumno"      # Uso básico, bajo consumo
    JP = "jp"              # Uso intermedio
    MAESTRO = "maestro"    # Uso avanzado
    INVESTIGADOR = "investigador"  # Uso intensivo
    
    @staticmethod
    def get_resource_usage_factors(profile: str) -> Dict[str, float]:
        """
        Devuelve factores de uso real de recursos para cada perfil
        (porcentaje del flavor que realmente utilizan en promedio)
        """
        profiles = {
            UserProfile.ALUMNO: {"vcpu": 0.3, "ram": 0.4, "disk": 0.2},
            UserProfile.JP: {"vcpu": 0.5, "ram": 0.6, "disk": 0.3},
            UserProfile.MAESTRO: {"vcpu": 0.7, "ram": 0.8, "disk": 0.4},
            UserProfile.INVESTIGADOR: {"vcpu": 0.9, "ram": 0.9, "disk": 0.5}
        }
        return profiles.get(profile.lower(), profiles[UserProfile.ALUMNO])
    
    @staticmethod
    def get_workload_variability(profile: str) -> float:
        """
        Devuelve factor de variabilidad de carga por perfil
        (qué tanto puede variar el uso en picos durante el ciclo de vida)
        """
        variability = {
            UserProfile.ALUMNO: 1.2,      # 20% de variabilidad
            UserProfile.JP: 1.5,          # 50% de variabilidad
            UserProfile.MAESTRO: 1.8,     # 80% de variabilidad
            UserProfile.INVESTIGADOR: 2.0 # 100% de variabilidad (puede duplicar uso)
        }
        return variability.get(profile.lower(), 1.5)
    
    @staticmethod
    def get_overprovisioning_limits(profile: str) -> Dict[str, float]:
        """
        Devuelve límites de sobreaprovisionamiento permitidos por SLA
        según el perfil de usuario (LÍMITES REDUCIDOS)
        """
        # Perfiles con mayor prioridad tienen límites más estrictos
        # TODOS LOS VALORES REDUCIDOS PARA MAYOR SEGURIDAD
        limits = {
            UserProfile.ALUMNO: {"vcpu": 2.0, "ram": 1.2, "disk": 3.0},  # Reducidos de 3.0/1.5/5.0
            UserProfile.JP: {"vcpu": 1.7, "ram": 1.1, "disk": 2.5},      # Reducidos de 2.5/1.4/4.0
            UserProfile.MAESTRO: {"vcpu": 1.5, "ram": 1.1, "disk": 2.0}, # Reducidos de 2.0/1.3/3.0
            UserProfile.INVESTIGADOR: {"vcpu": 1.3, "ram": 1.05, "disk": 1.5} # Reducidos de 1.5/1.2/2.0
        }
        return limits.get(profile.lower(), limits[UserProfile.ALUMNO])

@dataclass
class WorkloadType:
    """Define tipos de cargas de trabajo con diferentes patrones de uso"""
    GENERAL = "general"               # Uso equilibrado de recursos
    CPU_INTENSIVE = "cpu_intensive"   # Alto uso de CPU
    MEMORY_INTENSIVE = "memory_intensive"  # Alto uso de memoria
    IO_INTENSIVE = "io_intensive"     # Alto uso de disco/red
    
    @staticmethod
    def get_resource_weights(workload_type: str) -> Dict[str, float]:
        """
        Devuelve pesos relativos para cada tipo de recurso según workload
        """
        weights = {
            WorkloadType.GENERAL: {"vcpu": 0.4, "ram": 0.4, "disk": 0.2},
            WorkloadType.CPU_INTENSIVE: {"vcpu": 0.7, "ram": 0.2, "disk": 0.1},
            WorkloadType.MEMORY_INTENSIVE: {"vcpu": 0.2, "ram": 0.7, "disk": 0.1},
            WorkloadType.IO_INTENSIVE: {"vcpu": 0.2, "ram": 0.3, "disk": 0.5}
        }
        return weights.get(workload_type.lower(), weights[WorkloadType.GENERAL])

@dataclass
class VirtualMachine:
    """
    Representa una máquina virtual con su flavor asociado
    """
    id: int
    name: str
    flavor: Flavor
    utility: float = 0.0
    
    def __str__(self):
        return f"VM {self.id}: {self.name} - {self.flavor.name} (Utilidad: {self.utility:.2f})"

@dataclass
class Slice:
    """Representa un conjunto de VMs relacionadas a desplegar como unidad"""
    id: int
    name: str
    vms: List[VirtualMachine]
    user_profile: str = UserProfile.ALUMNO
    workload_type: str = WorkloadType.GENERAL
    
    def get_total_nominal_resources(self) -> Dict[str, Union[int, float]]:
        """Calcula recursos totales nominales del slice (según flavors)"""
        return {
            "vcpu": sum(vm.flavor.vcpus for vm in self.vms),
            "ram": sum(vm.flavor.ram for vm in self.vms),
            "disk": sum(vm.flavor.disk for vm in self.vms)
        }
    
    def get_estimated_usage(self) -> Dict[str, float]:
        """
        Calcula estimación de uso real de recursos considerando el perfil
        de usuario, tipo de workload y variabilidad
        """
        nominal = self.get_total_nominal_resources()
        usage_factors = UserProfile.get_resource_usage_factors(self.user_profile)
        variability = UserProfile.get_workload_variability(self.user_profile)
        
        return {
            "vcpu": nominal["vcpu"] * usage_factors["vcpu"] * variability,
            "ram": nominal["ram"] * usage_factors["ram"] * variability,
            "disk": nominal["disk"] * usage_factors["disk"] * variability
        }
    
    def get_resource_weights(self) -> Dict[str, float]:
        """Obtiene pesos relativos para cada tipo de recurso según el workload"""
        return WorkloadType.get_resource_weights(self.workload_type)
    
    def calculate_utility(self) -> float:
        """
        Calcula la utilidad del slice completo basada en sus recursos y perfil
        """
        # Obtener estimación de uso real
        estimated = self.get_estimated_usage()
        
        # Obtener pesos por tipo de recurso según workload
        weights = self.get_resource_weights()
        
        # Cálculo de utilidad ponderada (prioriza recursos según tipo de workload)
        utility = (
            weights["vcpu"] * estimated["vcpu"] + 
            weights["ram"] * (estimated["ram"] / 1024) +  # Convertir a GB para escalar
            weights["disk"] * estimated["disk"]
        )
        
        # Factores adicionales según el rol del usuario
        role_factors = {
            UserProfile.ALUMNO: 1.0,
            UserProfile.JP: 1.5,
            UserProfile.MAESTRO: 2.0,
            UserProfile.INVESTIGADOR: 3.0
        }
        
        role_factor = role_factors.get(self.user_profile, 1.0)
        utility *= role_factor
        
        return utility

@dataclass
class PhysicalServer:
    """
    Representa un servidor físico con sus capacidades, uso actual y métricas de congestión
    """
    id: int
    name: str
    total_vcpus: int
    total_ram: int  # En MB
    total_disk: float  # En GB
    used_vcpus: int = 0
    used_ram: int = 0
    used_disk: float = 0.0
    
    # Factores de sobreaprovisionamiento predeterminados
    vcpu_overprovisioning_factor: float = 1.7  # Sobreaprovisionamiento de CPU
    ram_overprovisioning_factor: float = 1.1   # Sobreaprovisionamiento de RAM
    disk_overprovisioning_factor: float = 1.7  # Sobreaprovisionamiento de disco
    
    # Factor de rendimiento relativo del servidor
    performance_factor: float = 1.0
    
    @property
    def available_vcpus(self) -> int:
        """vCPUs disponibles sin sobreaprovisionamiento"""
        return self.total_vcpus - self.used_vcpus
    
    @property
    def available_ram(self) -> int:
        """RAM disponible sin sobreaprovisionamiento (MB)"""
        return self.total_ram - self.used_ram
    
    @property
    def available_disk(self) -> float:
        """Disco disponible sin sobreaprovisionamiento (GB)"""
        return self.total_disk - self.used_disk
    
    @property
    def max_vcpus_with_overprovisioning(self) -> float:
        """Capacidad máxima de vCPUs con sobreaprovisionamiento"""
        return self.total_vcpus * self.vcpu_overprovisioning_factor
    
    @property
    def max_ram_with_overprovisioning(self) -> float:
        """Capacidad máxima de RAM con sobreaprovisionamiento"""
        return self.total_ram * self.ram_overprovisioning_factor
    
    @property
    def max_disk_with_overprovisioning(self) -> float:
        """Capacidad máxima de disco con sobreaprovisionamiento"""
        return self.total_disk * self.disk_overprovisioning_factor
    
    def get_real_time_resources(self) -> Dict[str, Union[int, float]]:
        """
        Obtiene recursos disponibles en tiempo real mediante API externa
        NOTA: En una implementación real, esto consultaría una API de monitoreo.
              Por ahora, simulamos con los valores actuales calculados.
        """
        # TODO: Reemplazar con llamada real a API de monitoreo cuando esté disponible
        return {
            "available_vcpus": self.available_vcpus,
            "available_ram": self.available_ram,
            "available_disk": self.available_disk
        }
    
    def get_current_congestion(self) -> Dict[str, float]:
        """
        Calcula niveles actuales de congestión por tipo de recurso (0-1)
        Valores más altos indican mayor congestión
        """
        # Congestión normalizada por tipo de recurso
        vcpu_congestion = min(1.0, self.used_vcpus / self.max_vcpus_with_overprovisioning)
        ram_congestion = min(1.0, self.used_ram / self.max_ram_with_overprovisioning)
        disk_congestion = min(1.0, self.used_disk / self.max_disk_with_overprovisioning)
        
        # Ponderación de congestión (mayor peso a CPU y RAM)
        weighted_congestion = 0.5 * vcpu_congestion + 0.35 * ram_congestion + 0.15 * disk_congestion
        
        return {
            "vcpu": vcpu_congestion,
            "ram": ram_congestion,
            "disk": disk_congestion,
            "weighted": weighted_congestion
        }
    
    def estimate_congestion_after_slice(self, slice: Slice) -> Dict[str, float]:
        """
        Estima la congestión resultante después de asignar un slice completo
        """
        # Obtener uso estimado de recursos del slice
        estimated_usage = slice.get_estimated_usage()
        
        # Calcular nuevos niveles de uso
        new_used_vcpus = self.used_vcpus + estimated_usage["vcpu"]
        new_used_ram = self.used_ram + estimated_usage["ram"]
        new_used_disk = self.used_disk + estimated_usage["disk"]
        
        # Calcular congestión por tipo de recurso
        vcpu_congestion = min(1.0, new_used_vcpus / self.max_vcpus_with_overprovisioning)
        ram_congestion = min(1.0, new_used_ram / self.max_ram_with_overprovisioning)
        disk_congestion = min(1.0, new_used_disk / self.max_disk_with_overprovisioning)
        
        # Ponderar según el tipo de workload del slice
        resource_weights = slice.get_resource_weights()
        weighted_congestion = (
            resource_weights["vcpu"] * vcpu_congestion +
            resource_weights["ram"] * ram_congestion +
            resource_weights["disk"] * disk_congestion
        )
        
        return {
            "vcpu": vcpu_congestion,
            "ram": ram_congestion,
            "disk": disk_congestion,
            "weighted": weighted_congestion
        }
    
    def estimate_queue_time(self, slice: Slice) -> float:
        """
        Estima tiempo relativo de espera en cola para el slice (0-1)
        Valores más altos indican mayor tiempo de espera
        """
        # Estimar congestión tras asignar el slice
        congestion = self.estimate_congestion_after_slice(slice)
        
        # Función no lineal para estimar tiempo de espera según congestión
        # (crece exponencialmente cuando la congestión es alta)
        vcpu_congestion = congestion["vcpu"]
        
        if vcpu_congestion <= 0.7:
            # Crecimiento lineal hasta 70% de congestión
            queue_time = vcpu_congestion * 0.4  # máximo 0.28 cuando 70%
        else:
            # Crecimiento exponencial después del 70%
            # Tiempo base + incremento exponencial
            base = 0.28  # valor a 70%
            queue_time = base + ((vcpu_congestion - 0.7) / 0.3) ** 2 * 0.72  # máximo 1.0
        
        return queue_time
    
    def can_host_slice(self, slice: Slice) -> bool:
        """
        Determina si el servidor puede alojar todo el slice considerando
        los límites de sobreaprovisionamiento para evaluación, pero verificando
        contra la capacidad real del servidor para la asignación efectiva.
        """
        # Obtener uso estimado del slice
        estimated_usage = slice.get_estimated_usage()
        
        # Obtener recursos disponibles en tiempo real (simulado)
        real_time_resources = self.get_real_time_resources()
        
        # Obtener límites de sobreaprovisionamiento según el perfil (más conservadores)
        overprovisioning_limits = UserProfile.get_overprovisioning_limits(slice.user_profile)
        
        # REDUCIR LÍMITES: Aplicar factores más conservadores
        # Reducir los límites originales a valores más seguros
        reduced_limits = {
            "vcpu": min(overprovisioning_limits["vcpu"] * 0.8, 1.7),  # Máximo 1.7x en vez de 2.0x
            "ram": min(overprovisioning_limits["ram"] * 0.8, 1.1),    # Máximo 1.1x en vez de 1.5x  
            "disk": min(overprovisioning_limits["disk"] * 0.8, 1.7)   # Máximo 1.7x en vez de 2.0x
        }
        
        # Calcular capacidades máximas permitidas según SLA y límites reducidos
        max_vcpus = self.total_vcpus * reduced_limits["vcpu"]
        max_ram = self.total_ram * reduced_limits["ram"]
        max_disk = self.total_disk * reduced_limits["disk"]
        
        # Verificar recursos contra capacidad con sobreaprovisionamiento reducido
        sufficient_vcpus = (self.used_vcpus + estimated_usage["vcpu"]) <= max_vcpus
        sufficient_ram = (self.used_ram + estimated_usage["ram"]) <= max_ram
        sufficient_disk = (self.used_disk + estimated_usage["disk"]) <= max_disk
        
        # También verificar contra recursos REALMENTE disponibles
        # Esto es crítico para evitar asignar más de lo físicamente posible
        real_vcpus_check = estimated_usage["vcpu"] <= real_time_resources["available_vcpus"] * 1.3  # Máx 30% sobreprovisionamiento
        real_ram_check = estimated_usage["ram"] <= real_time_resources["available_ram"] * 1.05      # Máx 5% sobreprovisionamiento 
        real_disk_check = estimated_usage["disk"] <= real_time_resources["available_disk"]          # Sin sobreprovisionamiento
        
        # Verificación extra: nunca exceder 95% de la capacidad física total
        physical_safety_check = (
            (self.used_vcpus + estimated_usage["vcpu"]) <= self.total_vcpus * 0.95 and
            (self.used_ram + estimated_usage["ram"]) <= self.total_ram * 0.95 and
            (self.used_disk + estimated_usage["disk"]) <= self.total_disk * 0.95
        )
        
        # Loguear información detallada sobre la evaluación
        Logger.debug(f"Evaluando servidor {self.name} (ID: {self.id}) para slice {slice.name}:")
        Logger.debug(f"  vCPUs: {self.used_vcpus:.1f} + {estimated_usage['vcpu']:.1f} <= {max_vcpus:.1f} ({'✓' if sufficient_vcpus else '✗'})")
        Logger.debug(f"  RAM: {self.used_ram:.1f} + {estimated_usage['ram']:.1f} <= {max_ram:.1f} ({'✓' if sufficient_ram else '✗'})")
        Logger.debug(f"  Disk: {self.used_disk:.1f} + {estimated_usage['disk']:.1f} <= {max_disk:.1f} ({'✓' if sufficient_disk else '✗'})")
        Logger.debug(f"  Recursos reales - vCPUs: {real_vcpus_check}, RAM: {real_ram_check}, Disk: {real_disk_check}")
        Logger.debug(f"  Límites físicos seguros: {physical_safety_check}")
        
        # Comprobar todos los recursos y verificaciones
        return (sufficient_vcpus and sufficient_ram and sufficient_disk and 
                real_vcpus_check and real_ram_check and real_disk_check and
                physical_safety_check)

    def get_slice_fit_score(self, slice: Slice) -> float:
        """
        Calcula un puntaje que indica qué tan bien se ajusta un slice a este servidor.
        Mayor puntaje = mejor ajuste (0 = no es viable)
        """
        # Si no puede hospedar el slice, retornar 0
        if not self.can_host_slice(slice):
            return 0.0
        
        # Factor 1: Congestión tras asignar el slice (menor congestión = mayor puntaje)
        congestion = self.estimate_congestion_after_slice(slice)
        congestion_factor = 1.0 - congestion["weighted"]  # 0-1, mayor es mejor
        
        # Factor 2: Tiempo de espera en cola (menor tiempo = mayor puntaje)
        queue_time = self.estimate_queue_time(slice)
        queue_factor = 1.0 - queue_time  # 0-1, mayor es mejor
        
        # Factor 3: Rendimiento del servidor
        performance_factor = self.performance_factor  # configurable por servidor
        
        # Factor 4: Bonus por consolidación (preferimos agrupar VMs relacionadas)
        # Mayor bonus para slices con más VMs
        consolidation_bonus = 1.0 + (min(len(slice.vms), 10) / 20.0)  # máx +50%
        
        # Calcular puntaje final combinado
        score = (
            0.4 * congestion_factor +  # 40% del peso
            0.3 * queue_factor +       # 30% del peso
            0.3 * performance_factor   # 30% del peso
        ) * consolidation_bonus
        
        return score
    
    def can_host_partial_slice(self, slice: Slice, vm_subset: List[VirtualMachine]) -> bool:
        """
        Determina si el servidor puede alojar un subconjunto específico de VMs del slice
        """
        # Crear un slice parcial con el subconjunto de VMs
        partial_slice = Slice(
            id=slice.id,
            name=f"{slice.name}_partial",
            vms=vm_subset,
            user_profile=slice.user_profile,
            workload_type=slice.workload_type
        )
        
        # Verificar si puede hospedar este slice parcial
        return self.can_host_slice(partial_slice)

@dataclass
class PlacementResult:
    """
    Almacena el resultado de la colocación de VMs
    """
    success: bool
    assignments: Dict[int, int] = field(default_factory=dict)  # VM_id -> Server_id
    message: str = ""
    objective_value: float = 0.0
    unassigned_details: List[Dict] = field(default_factory=list)

    def to_dict(self, vms=None, servers=None):
        """
        Convierte el resultado a un diccionario para respuesta JSON con información detallada
        
        Args:
            vms: Lista de VMs usadas en el placement
            servers: Lista de servidores usados en el placement
        """
        # Diccionario básico
        result_dict = {
            "success": self.success,
            "message": self.message,
            "objective_value": round(self.objective_value, 2)
        }
        
        if self.unassigned_details:
            result_dict["unassigned_details"] = self.unassigned_details
        
        # Si tenemos VMs y servidores disponibles, generamos información más detallada
        if vms and servers and self.assignments:
            # Crear asignaciones detalladas
            assignments_list = []
            unassigned_vms = []
            
            # Mapeo de IDs a objetos
            vm_map = {vm.id: vm for vm in vms}
            server_map = {server.id: server for server in servers}
            
            # Procesar asignaciones
            for vm_id, server_id in self.assignments.items():
                if vm_id in vm_map and server_id in server_map:
                    vm = vm_map[vm_id]
                    server = server_map[server_id]
                    assignments_list.append({
                        "vm_id": vm_id,
                        "vm_name": vm.name,
                        "server_id": server_id,
                        "server_name": server.name
                    })
            
            # Encontrar VMs no asignadas
            for vm in vms:
                if vm.id not in self.assignments:
                    unassigned_vms.append({
                        "vm_id": vm.id,
                        "vm_name": vm.name,
                        "flavor": {
                            "name": vm.flavor.name,
                            "vcpus": vm.flavor.vcpus,
                            "ram": vm.flavor.ram,
                            "disk": vm.flavor.disk
                        }
                    })
            
            # Ordenar por IDs para más claridad
            assignments_list.sort(key=lambda x: x["vm_id"])
            
            # Agregar al resultado
            result_dict["assignments"] = assignments_list
            result_dict["unassigned_vms"] = unassigned_vms
            
            # Calcular y agregar información de uso de recursos si hay servidores
            if servers and self.assignments:
                # Crear mapa de servidor a VMs asignadas
                server_to_vms = {}
                for vm_id, server_id in self.assignments.items():
                    if server_id not in server_to_vms:
                        server_to_vms[server_id] = []
                    server_to_vms[server_id].append(vm_id)
                
                # Calcular uso de recursos por servidor
                servers_usage = []
                total_used_vcpus = 0
                total_used_ram = 0
                total_used_disk = 0.0
                total_available_vcpus = 0
                total_available_ram = 0
                total_available_disk = 0.0
                
                for server in servers:
                    if server.id in server_to_vms:
                        vm_ids = server_to_vms[server.id]
                        vcpus_used = 0
                        ram_used = 0
                        disk_used = 0.0
                        
                        for vm_id in vm_ids:
                            if vm_id in vm_map:
                                vm = vm_map[vm_id]
                                vcpus_used += vm.flavor.vcpus
                                ram_used += vm.flavor.ram
                                disk_used += vm.flavor.disk
                        
                        # Agregar a totales
                        total_used_vcpus += vcpus_used
                        total_used_ram += ram_used
                        total_used_disk += disk_used
                        total_available_vcpus += server.total_vcpus  # Usar capacidad total real, no disponible
                        total_available_ram += server.total_ram      # Usar capacidad total real, no disponible
                        total_available_disk += server.total_disk    # Usar capacidad total real, no disponible
                        
                        # Calcular porcentajes respecto a capacidad total REAL, no disponible
                        vcpu_percent = (vcpus_used / server.total_vcpus * 100) if server.total_vcpus > 0 else 0
                        ram_percent = (ram_used / server.total_ram * 100) if server.total_ram > 0 else 0
                        disk_percent = (disk_used / server.total_disk * 100) if server.total_disk > 0 else 0
                        
                        servers_usage.append({
                            "id": server.id,
                            "name": server.name,
                            "vms_count": len(vm_ids),
                            "resources": {
                                "vcpus": {
                                    "used": vcpus_used,
                                    "total": server.total_vcpus,  # Capacidad TOTAL, no disponible
                                    "percent": round(vcpu_percent, 2)
                                },
                                "ram": {
                                    "used": ram_used,
                                    "total": server.total_ram,    # Capacidad TOTAL, no disponible
                                    "percent": round(ram_percent, 2)
                                },
                                "disk": {
                                    "used": round(disk_used, 1),
                                    "total": round(server.total_disk, 1),  # Capacidad TOTAL, no disponible
                                    "percent": round(disk_percent, 2)
                                }
                            }
                        })
                
                # Calcular porcentajes totales
                total_vcpu_percent = (total_used_vcpus / total_available_vcpus * 100) if total_available_vcpus > 0 else 0
                total_ram_percent = (total_used_ram / total_available_ram * 100) if total_available_ram > 0 else 0
                total_disk_percent = (total_used_disk / total_available_disk * 100) if total_available_disk > 0 else 0
                
                # Agregar información de recursos
                result_dict["resource_usage"] = {
                    "servers": servers_usage,
                    "total": {
                        "vcpus": {
                            "used": total_used_vcpus,
                            "total": total_available_vcpus,
                            "percent": round(total_vcpu_percent, 2)
                        },
                        "ram": {
                            "used": total_used_ram,
                            "total": total_available_ram,
                            "percent": round(total_ram_percent, 2)
                        },
                        "disk": {
                            "used": round(total_used_disk, 1),
                            "total": round(total_available_disk, 1),
                            "percent": round(total_disk_percent, 2)
                        }
                    }
                }
        else:
            # Versión simplificada si no tenemos información detallada
            result_dict["assignments"] = {str(vm_id): server_id for vm_id, server_id in self.assignments.items()}
        
        return result_dict

# ===================== SUBMÓDULOS =====================
class SliceBasedPlacementSolver:
    """
    Resuelve el problema de asignación de slices completos a servidores físicos
    optimizando para la performance global y priorizando la localidad de las VMs.
    """
    
    def __init__(self, slice: Slice, servers: List[PhysicalServer]):
        """
        Inicializa el solucionador con un slice y los servidores disponibles
        
        Args:
            slice: El slice completo a asignar
            servers: Lista de servidores físicos disponibles
        """
        self.slice = slice
        self.servers = servers
    
    def solve(self) -> PlacementResult:
        """
        Resuelve el problema de placement para el slice completo
        
        Returns:
            PlacementResult con los resultados de la asignación
        """
        Logger.section(f"Iniciando placement para slice: {self.slice.name}")
        Logger.info(f"Perfil: {self.slice.user_profile}, Tipo: {self.slice.workload_type}")
        Logger.info(f"Total de VMs en el slice: {len(self.slice.vms)}")
        
        # Mostrar información detallada de las VMs
        Logger.info("Detalle de VMs en el slice:")
        for i, vm in enumerate(self.slice.vms):
            if i < 10:  # Limitar a 10 para no saturar los logs
                Logger.info(f"  VM {vm.name} (ID: {vm.id}): vCPUs: {vm.flavor.vcpus}, RAM: {vm.flavor.ram} MB, Disk: {vm.flavor.disk} GB")
            elif i == 10:
                Logger.info(f"  ... y {len(self.slice.vms) - 10} VMs más")
        
        # Mostrar información de los servidores disponibles
        Logger.info("Servidores físicos disponibles:")
        for i, server in enumerate(self.servers):
            Logger.info(f"  Servidor {server.name} (ID: {server.id}):")
            Logger.info(f"    Capacidad total: vCPUs: {server.total_vcpus}, RAM: {server.total_ram} MB, Disk: {server.total_disk} GB")
            Logger.info(f"    Uso actual: vCPUs: {server.used_vcpus}, RAM: {server.used_ram} MB, Disk: {server.used_disk} GB")
            Logger.info(f"    Disponible: vCPUs: {server.available_vcpus}, RAM: {server.available_ram} MB, Disk: {server.available_disk} GB")
            
            # Calcular límites de sobreaprovisionamiento
            limits = UserProfile.get_overprovisioning_limits(self.slice.user_profile)
            Logger.info(f"    Límites de sobreaprovisionamiento para perfil {self.slice.user_profile}:")
            Logger.info(f"      vCPUs: {server.total_vcpus * limits['vcpu']:.1f}, RAM: {server.total_ram * limits['ram']:.1f} MB, Disk: {server.total_disk * limits['disk']:.1f} GB")
        
        # Obtener recursos totales nominales y estimados
        nominal = self.slice.get_total_nominal_resources()
        estimated = self.slice.get_estimated_usage()
        
        Logger.info(f"Recursos nominales - vCPUs: {nominal['vcpu']}, RAM: {nominal['ram']} MB, Disco: {nominal['disk']:.1f} GB")
        Logger.info(f"Uso estimado - vCPUs: {estimated['vcpu']:.1f}, RAM: {estimated['ram']:.1f} MB, Disco: {estimated['disk']:.1f} GB")
        
        # Mostrar factores de uso y variabilidad
        usage_factors = UserProfile.get_resource_usage_factors(self.slice.user_profile)
        variability = UserProfile.get_workload_variability(self.slice.user_profile)
        Logger.info(f"Factores de uso para perfil {self.slice.user_profile}: {usage_factors}")
        Logger.info(f"Variabilidad para perfil {self.slice.user_profile}: {variability}")
        
        # 1. Verificar si hay suficientes recursos totales
        if not self._check_total_resources():
            error_msg = "No hay suficientes recursos totales sumando todos los servidores, considerando los límites de sobreaprovisionamiento pero respetando la capacidad real."

            Logger.error(error_msg)
            return self._create_failure_result(error_msg)
        
        # 2. Verificar si al menos un servidor puede alojar el slice completo
        viable_servers = [server for server in self.servers if server.can_host_slice(self.slice)]
        
        if viable_servers:
            # Al menos un servidor puede alojar todo el slice - elegir el óptimo
            Logger.info(f"Se encontraron {len(viable_servers)} servidores capaces de alojar el slice completo")
            return self._solve_single_server(viable_servers)
        else:
            # Ningún servidor puede alojar todo el slice - intentar distribuir con enfoque
            # que maximice la cantidad de VMs en un solo servidor
            Logger.info("Ningún servidor puede alojar el slice completo, intentando distribuir VMs")
            return self._solve_cluster_first_distribution()

    def _check_total_resources(self) -> bool:
        """
        Verifica si hay suficientes recursos totales sumando todos los servidores,
        considerando los límites de sobreaprovisionamiento pero respetando la capacidad real.
        """
        estimated_usage = self.slice.get_estimated_usage()
        overprovisioning_limits = UserProfile.get_overprovisioning_limits(self.slice.user_profile)
        
        # Limitar factores de sobreaprovisionamiento a valores razonables
        max_vcpu_factor = min(overprovisioning_limits["vcpu"], 2.0)
        max_ram_factor = min(overprovisioning_limits["ram"], 1.5)
        max_disk_factor = min(overprovisioning_limits["disk"], 2.0)
        
        # Sumar recursos disponibles considerando sobreaprovisionamiento según SLA y limitaciones físicas
        total_available_vcpus = sum(
            max(server.total_vcpus * max_vcpu_factor - server.used_vcpus, 0) 
            for server in self.servers
        )
        
        total_available_ram = sum(
            max(server.total_ram * max_ram_factor - server.used_ram, 0)
            for server in self.servers
        )
        
        total_available_disk = sum(
            max(server.total_disk * max_disk_factor - server.used_disk, 0)
            for server in self.servers
        )
        
        # Loguear información detallada sobre los recursos
        Logger.info("Verificando recursos totales disponibles:")
        Logger.info(f"  vCPUs requeridas: {estimated_usage['vcpu']:.1f}, disponibles: {total_available_vcpus:.1f}")
        Logger.info(f"  RAM requerida: {estimated_usage['ram']:.1f} MB, disponible: {total_available_ram:.1f} MB")
        Logger.info(f"  Disco requerido: {estimated_usage['disk']:.1f} GB, disponible: {total_available_disk:.1f} GB")
        
        # Verificar si son suficientes
        if estimated_usage["vcpu"] > total_available_vcpus:
            Logger.warning(f"Recursos insuficientes: vCPU estimado {estimated_usage['vcpu']:.1f}, disponible {total_available_vcpus:.1f}")
            return False
            
        if estimated_usage["ram"] > total_available_ram:
            Logger.warning(f"Recursos insuficientes: RAM estimada {estimated_usage['ram']:.1f} MB, disponible {total_available_ram:.1f} MB")
            return False
            
        if estimated_usage["disk"] > total_available_disk:
            Logger.warning(f"Recursos insuficientes: Disco estimado {estimated_usage['disk']:.1f} GB, disponible {total_available_disk:.1f} GB")
            return False
        
        Logger.success("Hay suficientes recursos totales para alojar el slice completo")
        return True

    def _solve_single_server(self, viable_servers: List[PhysicalServer]) -> PlacementResult:
        """
        Asigna todo el slice a un único servidor (el óptimo)
        """
        Logger.section("Resolviendo placement en servidor único")
        
        # Calcular puntaje de ajuste para cada servidor viable
        server_scores = []
        for server in viable_servers:
            score = server.get_slice_fit_score(self.slice)
            server_scores.append((server, score))
            Logger.info(f"Servidor {server.name} (ID: {server.id}) - Score: {score:.4f}")
            
            # Detallar componentes del score
            congestion = server.estimate_congestion_after_slice(self.slice)
            queue_time = server.estimate_queue_time(self.slice)
            Logger.debug(f"  Congestión estimada: {congestion['weighted']:.2f}")
            Logger.debug(f"  Tiempo de espera estimado: {queue_time:.2f}")
            Logger.debug(f"  Factor de rendimiento: {server.performance_factor:.2f}")
        
        # Ordenar por puntaje descendente
        server_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Seleccionar el mejor servidor
        best_server, best_score = server_scores[0]
        
        Logger.success(f"Servidor óptimo: {best_server.name} (ID: {best_server.id}) con score {best_score:.4f}")
        
        # Verificar recursos disponibles del servidor seleccionado
        Logger.info(f"Recursos disponibles en {best_server.name}:")
        Logger.info(f"  vCPUs: {best_server.available_vcpus} de {best_server.total_vcpus}")
        Logger.info(f"  RAM: {best_server.available_ram} MB de {best_server.total_ram} MB")
        Logger.info(f"  Disco: {best_server.available_disk:.1f} GB de {best_server.total_disk:.1f} GB")
        
        # Verificar recursos requeridos por el slice
        estimated_usage = self.slice.get_estimated_usage()
        Logger.info(f"Recursos estimados para el slice:")
        Logger.info(f"  vCPUs: {estimated_usage['vcpu']:.1f}")
        Logger.info(f"  RAM: {estimated_usage['ram']:.1f} MB")
        Logger.info(f"  Disco: {estimated_usage['disk']:.1f} GB")
        
        # Crear asignaciones para todas las VMs en este servidor
        assignments = {}
        for vm in self.slice.vms:
            assignments[vm.id] = best_server.id
            Logger.success(f"VM {vm.name} (ID: {vm.id}) asignada a servidor {best_server.name} (ID: {best_server.id})")
        
        return PlacementResult(
            success=True,
            assignments=assignments,
            message=f"Slice completo asignado al servidor {best_server.name} optimizando localidad y rendimiento",
            objective_value=best_score
        )
    
    def _solve_cluster_first_distribution(self) -> PlacementResult:
        """
        Distribuye las VMs del slice priorizando colocar la mayor cantidad posible
        en un solo servidor (cluster-first approach) para maximizar la localidad.
        """
        Logger.section("Distribuyendo slice con enfoque de maximizar clusterización")
        
        # Verificar si las VMs individualmente pueden ser asignadas
        unassignable_vms = []
        for vm in self.slice.vms:
            # Crear un mini-slice con solo esta VM
            mini_slice = Slice(
                id=self.slice.id,
                name=f"mini_{vm.name}",
                vms=[vm],
                user_profile=self.slice.user_profile,
                workload_type=self.slice.workload_type
            )
            
            # Verificar si algún servidor puede alojar esta VM
            if not any(server.can_host_slice(mini_slice) for server in self.servers):
                unassignable_vms.append(vm)
                Logger.warning(f"VM {vm.name} (ID: {vm.id}) no puede ser asignada a ningún servidor")
        
        if unassignable_vms:
            # Hay VMs que no pueden ser asignadas
            vm_names = [vm.name for vm in unassignable_vms]
            error_msg = f"Las siguientes VMs no pueden ser asignadas a ningún servidor: {', '.join(vm_names)}"
            Logger.error(error_msg)
            return self._create_failure_result(error_msg, unassignable_vms)
        
        # Algoritmo de distribución con enfoque de cluster first
        return self._solve_with_cluster_first_algorithm()
    
    def _solve_with_cluster_first_algorithm(self) -> PlacementResult:
        """
        Implementa el algoritmo Cluster-First que maximiza el número de VMs
        en un solo servidor antes de distribuir el resto en el menor número
        posible de servidores adicionales.
        """
        Logger.section("Aplicando algoritmo Cluster-First para maximizar localidad")
        
        # Paso 1: Identificar el servidor que puede alojar la mayor cantidad de VMs
        max_vms_count = 0
        best_server = None
        best_vm_subset = []
        
        # Para cada servidor, encontrar la mayor cantidad de VMs que puede alojar
        for server in self.servers:
            # Obtener límites de sobreaprovisionamiento con valores más conservadores
            op_limits = UserProfile.get_overprovisioning_limits(self.slice.user_profile)
            
            # REDUCIR LÍMITES: Aplicar factores más conservadores
            reduced_limits = {
                "vcpu": min(op_limits["vcpu"] * 0.8, 1.7),  # Máximo 1.7x
                "ram": min(op_limits["ram"] * 0.8, 1.1),    # Máximo 1.1x  
                "disk": min(op_limits["disk"] * 0.8, 1.7)   # Máximo 1.7x
            }
            
            # Obtener recursos disponibles en tiempo real
            real_time_resources = server.get_real_time_resources()
            
            # Calcular capacidad restante con límites reducidos
            remaining_vcpus = min(
                server.total_vcpus * reduced_limits["vcpu"] - server.used_vcpus,
                real_time_resources["available_vcpus"] * 1.3  # Máx 30% sobreprovisionamiento
            )
            
            remaining_ram = min(
                server.total_ram * reduced_limits["ram"] - server.used_ram,
                real_time_resources["available_ram"] * 1.05  # Máx 5% sobreprovisionamiento
            )
            
            remaining_disk = min(
                server.total_disk * reduced_limits["disk"] - server.used_disk,
                real_time_resources["available_disk"]  # Sin sobreprovisionamiento
            )
            
            # Verificación extra: nunca exceder 95% de la capacidad física total
            max_safe_vcpus = server.total_vcpus * 0.95 - server.used_vcpus
            max_safe_ram = server.total_ram * 0.95 - server.used_ram
            max_safe_disk = server.total_disk * 0.95 - server.used_disk
            
            # Usar el mínimo entre los límites calculados y los límites seguros
            remaining_vcpus = min(remaining_vcpus, max_safe_vcpus)
            remaining_ram = min(remaining_ram, max_safe_ram)
            remaining_disk = min(remaining_disk, max_safe_disk)
            
            Logger.debug(f"Evaluando servidor {server.name} para cluster principal:")
            Logger.debug(f"  Capacidad restante - vCPUs: {remaining_vcpus:.1f}, RAM: {remaining_ram:.1f} MB, Disco: {remaining_disk:.1f} GB")
            
            # Seleccionar las VMs que cabrían en este servidor (ordenadas por utilidad)
            usage_factors = UserProfile.get_resource_usage_factors(self.slice.user_profile)
            variability = UserProfile.get_workload_variability(self.slice.user_profile)
            
            # Ordenar VMs por utilidad descendente (primero las más valiosas)
            sorted_vms = sorted(self.slice.vms, key=lambda vm: (
                vm.utility if hasattr(vm, 'utility') and vm.utility > 0 
                else vm.flavor.vcpus * usage_factors["vcpu"] * variability +
                    vm.flavor.ram * usage_factors["ram"] * variability / 1024 +
                    vm.flavor.disk * usage_factors["disk"] * variability
            ), reverse=True)
            
            # Intentar colocar tantas VMs como sea posible en este servidor
            current_vcpus = 0
            current_ram = 0
            current_disk = 0
            vm_subset = []
            
            for vm in sorted_vms:
                # Calcular uso estimado de esta VM
                vm_vcpus = vm.flavor.vcpus * usage_factors["vcpu"] * variability
                vm_ram = vm.flavor.ram * usage_factors["ram"] * variability
                vm_disk = vm.flavor.disk * usage_factors["disk"] * variability
                
                # Verificar si cabe en el espacio restante
                if (current_vcpus + vm_vcpus <= remaining_vcpus and
                    current_ram + vm_ram <= remaining_ram and
                    current_disk + vm_disk <= remaining_disk):
                    
                    # VM cabe, agregarla al subconjunto
                    vm_subset.append(vm)
                    current_vcpus += vm_vcpus
                    current_ram += vm_ram
                    current_disk += vm_disk
                    
                    Logger.debug(f"  + VM {vm.name} cabe (vCPUs: {vm_vcpus:.1f}, RAM: {vm_ram:.1f}, Disk: {vm_disk:.1f})")
                else:
                    Logger.debug(f"  - VM {vm.name} no cabe (vCPUs: {vm_vcpus:.1f}, RAM: {vm_ram:.1f}, Disk: {vm_disk:.1f})")
            
            Logger.debug(f"  Servidor {server.name} puede alojar {len(vm_subset)} de {len(self.slice.vms)} VMs")
            
            # Actualizar si este servidor puede alojar más VMs que el mejor actual
            if len(vm_subset) > max_vms_count:
                max_vms_count = len(vm_subset)
                best_server = server
                best_vm_subset = vm_subset
        
        if not best_server or not best_vm_subset:
            Logger.error("No se pudo encontrar un servidor capaz de alojar ninguna VM del slice")
            return self._create_failure_result("No hay servidores disponibles para el slice")
        
        Logger.success(f"Servidor principal: {best_server.name} (ID: {best_server.id}) - alojará {len(best_vm_subset)} de {len(self.slice.vms)} VMs")
        
        # Asignación para el primer grupo de VMs al mejor servidor
        assignments = {}
        for vm in best_vm_subset:
            assignments[vm.id] = best_server.id
            Logger.success(f"VM {vm.name} (ID: {vm.id}) asignada a servidor principal {best_server.name} (ID: {best_server.id})")
        
        # Identificar VMs restantes
        remaining_vms = [vm for vm in self.slice.vms if vm.id not in assignments]
        
        if not remaining_vms:
            # Todas las VMs colocadas en un solo servidor
            return PlacementResult(
                success=True,
                assignments=assignments,
                message=f"Todas las VMs del slice asignadas a un único servidor: {best_server.name}",
                objective_value=1.0  # Máxima puntuación por localidad perfecta
            )
        
        # Distribuir VMs restantes a otros servidores, minimizando el número de servidores usados
        Logger.info(f"Distribuyendo {len(remaining_vms)} VMs restantes usando el mínimo de servidores adicionales")
        
        # Excluir el servidor principal para las asignaciones restantes
        available_servers = [s for s in self.servers if s.id != best_server.id]
        
        # Intentar asignar las VMs restantes usando el mínimo de servidores
        # Para cada servidor, intentamos llenar al máximo su capacidad
        remaining_vms_copy = remaining_vms.copy()  # Para no modificar la lista original durante la iteración
        
        # Ordenar servidores por capacidad disponible total (puntuación ponderada)
        server_capacity = []
        for server in available_servers:
            # Calcular capacidad efectiva considerando los límites de sobreaprovisionamiento
            overprovisioning_limits = UserProfile.get_overprovisioning_limits(self.slice.user_profile)
            
            # Capacidad ponderada (mayor = mejor)
            capacity_score = (
                (server.total_vcpus * overprovisioning_limits["vcpu"] - server.used_vcpus) / server.total_vcpus +
                (server.total_ram * overprovisioning_limits["ram"] - server.used_ram) / server.total_ram +
                (server.total_disk * overprovisioning_limits["disk"] - server.used_disk) / server.total_disk
            ) / 3.0 * server.performance_factor
            
            server_capacity.append((server, capacity_score))
        
        # Ordenar por capacidad disponible (mayor primero)
        server_capacity.sort(key=lambda x: x[1], reverse=True)
        
        # Para cada servidor, intentar asignar la mayor cantidad de VMs posible
        for server, capacity_score in server_capacity:
            if not remaining_vms_copy:
                break  # Todas las VMs ya han sido asignadas
                
            Logger.info(f"Intentando asignar VMs al servidor {server.name} (score: {capacity_score:.2f})")
            
            # Calcular límites según sobreaprovisionamiento para este servidor
            overprovisioning_limits = UserProfile.get_overprovisioning_limits(self.slice.user_profile)
            remaining_vcpus = server.total_vcpus * overprovisioning_limits["vcpu"] - server.used_vcpus
            remaining_ram = server.total_ram * overprovisioning_limits["ram"] - server.used_ram
            remaining_disk = server.total_disk * overprovisioning_limits["disk"] - server.used_disk
            
            # Variables para tracking de asignación
            current_vcpus = 0
            current_ram = 0
            current_disk = 0
            assigned_vms = []
            
            # Intentar asignar la mayor cantidad de VMs posible a este servidor
            for vm in sorted(remaining_vms_copy, key=lambda v: v.utility, reverse=True):
                # Calcular uso estimado de esta VM
                usage_factors = UserProfile.get_resource_usage_factors(self.slice.user_profile)
                variability = UserProfile.get_workload_variability(self.slice.user_profile)
                
                vm_vcpus = vm.flavor.vcpus * usage_factors["vcpu"] * variability
                vm_ram = vm.flavor.ram * usage_factors["ram"] * variability
                vm_disk = vm.flavor.disk * usage_factors["disk"] * variability
                
                # Verificar si cabe en el espacio restante
                if (current_vcpus + vm_vcpus <= remaining_vcpus and
                    current_ram + vm_ram <= remaining_ram and
                    current_disk + vm_disk <= remaining_disk):
                    
                    # VM cabe, asignarla a este servidor
                    assignments[vm.id] = server.id
                    assigned_vms.append(vm)
                    
                    # Actualizar recursos utilizados
                    current_vcpus += vm_vcpus
                    current_ram += vm_ram
                    current_disk += vm_disk
                    
                    Logger.success(f"VM {vm.name} (ID: {vm.id}) asignada a servidor secundario {server.name} (ID: {server.id})")
                else:
                    Logger.debug(f"VM {vm.name} no cabe en servidor {server.name} (recursos insuficientes)")
            
            # Actualizar el uso del servidor para reflejar las nuevas asignaciones
            server.used_vcpus += current_vcpus
            server.used_ram += current_ram
            server.used_disk += current_disk
            
            # Eliminar las VMs asignadas de la lista de restantes
            for vm in assigned_vms:
                remaining_vms_copy.remove(vm)
                
            Logger.info(f"Asignadas {len(assigned_vms)} VMs al servidor {server.name}. Quedan {len(remaining_vms_copy)} por asignar.")
        
        # Verificar si quedaron VMs sin asignar
        if remaining_vms_copy:
            vm_names = [vm.name for vm in remaining_vms_copy]
            error_msg = f"No se pudieron asignar todas las VMs del slice. VMs sin asignar: {', '.join(vm_names)}"
            Logger.error(error_msg)
            return self._create_failure_result(error_msg, remaining_vms_copy)
        
        # Calcular servidores utilizados
        used_servers = len(set(assignments.values()))
        
        # Calcular puntuación de la solución
        # Más alto si hay mayor proporción de VMs en el servidor principal y menos servidores adicionales
        primary_server_count = len(best_vm_subset)
        total_vms = len(self.slice.vms)
        locality_score = primary_server_count / total_vms
        
        # Ajustar score para penalizar el uso de muchos servidores adicionales
        additional_servers = used_servers - 1  # Excluyendo el servidor principal
        if additional_servers > 0:
            # Penalización suave por cada servidor adicional
            locality_score *= (1.0 - 0.05 * additional_servers)
        
        Logger.success(f"Slice distribuido en {used_servers} servidores con {primary_server_count} VMs en el servidor principal")
        
        return PlacementResult(
            success=True,
            assignments=assignments,
            message=f"Slice distribuido en {used_servers} servidores, maximizando localidad con {primary_server_count} de {total_vms} VMs en servidor principal",
            objective_value=locality_score
        )

    def _create_failure_result(self, message: str, problematic_vms: List[VirtualMachine] = None) -> PlacementResult:
        """
        Crea un resultado detallado de fallo explicando las razones
        """
        # Si no se especifican VMs problemáticas, considerar todas
        if problematic_vms is None:
            problematic_vms = self.slice.vms
        
        # Análisis detallado de por qué no se pueden asignar las VMs
        unassigned_details = []
        
        for vm in problematic_vms:
            # Crear un mini-slice con solo esta VM
            mini_slice = Slice(
                id=self.slice.id,
                name=f"mini_{vm.name}",
                vms=[vm],
                user_profile=self.slice.user_profile,
                workload_type=self.slice.workload_type
            )
            
            # Estimación de uso real
            usage_factors = UserProfile.get_resource_usage_factors(self.slice.user_profile)
            variability = UserProfile.get_workload_variability(self.slice.user_profile)
            
            real_vcpus = vm.flavor.vcpus * usage_factors["vcpu"] * variability
            real_ram = vm.flavor.ram * usage_factors["ram"] * variability
            real_disk = vm.flavor.disk * usage_factors["disk"] * variability
            
            # Razones de fallo para esta VM
            reasons = []
            
            # Verificar cada servidor
            for server in self.servers:
                if not server.can_host_slice(mini_slice):
                    # Obtener límites de sobreaprovisionamiento
                    limits = UserProfile.get_overprovisioning_limits(self.slice.user_profile)
                    
                    # Calcular capacidades disponibles según SLA
                    available_vcpus = server.total_vcpus * limits["vcpu"] - server.used_vcpus
                    available_ram = server.total_ram * limits["ram"] - server.used_ram
                    available_disk = server.total_disk * limits["disk"] - server.used_disk
                    
                    # Identificar qué recursos son insuficientes
                    if real_vcpus > available_vcpus:
                        reasons.append(f"vCPUs insuficientes en servidor {server.name}: requiere {real_vcpus:.1f}, disponible {available_vcpus:.1f}")
                    
                    if real_ram > available_ram:
                        reasons.append(f"RAM insuficiente en servidor {server.name}: requiere {real_ram:.1f} MB, disponible {available_ram:.1f} MB")
                    
                    if real_disk > available_disk:
                        reasons.append(f"Disco insuficiente en servidor {server.name}: requiere {real_disk:.1f} GB, disponible {available_disk:.1f} GB")
            
            # Si no hay razones específicas, usar mensaje genérico
            if not reasons:
                reasons.append("No se encontró un servidor compatible según los requisitos de SLA")
            
            # Eliminar duplicados y agregar al resultado
            unassigned_details.append({
                "vm_id": vm.id,
                "vm_name": vm.name,
                "reasons": list(set(reasons))
            })
        
        return PlacementResult(
            success=False,
            message=message,
            unassigned_details=unassigned_details
        )

    def visualize_placement(self, result: PlacementResult, api_response=None):
        """
        Visualiza gráficamente los resultados del placement y guarda en carpeta 'resultados'
        
        Args:
            result: Resultado del placement
            api_response: Respuesta JSON del API con datos de uso de recursos
        """
        if not result.success or not result.assignments:
            Logger.error("No hay una solución válida para visualizar")
            return
        
        Logger.section("Visualización del Placement")
        
        try:
            import matplotlib
            matplotlib.use('Agg') 
            import matplotlib.pyplot as plt
            import numpy as np
            from datetime import datetime
            import os
            import traceback  # Para debug detallado

            # Crear carpeta 'resultados' si no existe
            results_dir = 'resultados'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                Logger.info(f"Creada carpeta '{results_dir}' para guardar visualizaciones")
            
            # Generar nombre de archivo único con timestamp
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = os.path.join(results_dir, f'resultado_{timestamp}.png')
            
            # Extraer datos para visualización
            vm_details = []
            server_usage = {}
            
            # Debug para ver qué contiene api_response
            if api_response:
                Logger.debug(f"API response keys: {list(api_response.keys())}")
                if 'content' in api_response:
                    Logger.debug(f"API content keys: {list(api_response['content'].keys())}")

            # Procesar datos según la fuente disponible
            if api_response and 'content' in api_response:
                # Usar datos de la respuesta de API
                api_content = api_response['content']
                
                # Verificar si hay asignaciones en la respuesta
                if 'assignments' in api_content and len(api_content['assignments']) > 0:
                    # Extraer asignaciones de VMs a servidores
                    for assignment in api_content['assignments']:
                        vm_id = assignment['vm_id']
                        vm_name = assignment['vm_name']
                        server_id = assignment['server_id']
                        server_name = assignment['server_name']
                        
                        # Buscar información de recursos de la VM
                        vm_info = None
                        for vm in self.slice.vms:
                            if vm.id == vm_id:
                                vm_info = {
                                    "vcpus": vm.flavor.vcpus,
                                    "ram": vm.flavor.ram,
                                    "disk": vm.flavor.disk
                                }
                                break
                        
                        if vm_info:
                            vm_details.append({
                                "vm_id": vm_id,
                                "vm_name": vm_name,
                                "server_id": server_id,
                                "server_name": server_name,
                                "vcpus": vm_info["vcpus"],
                                "ram": vm_info["ram"],
                                "disk": vm_info["disk"]
                            })
                        else:
                            # Si no encontramos la info, usar valores predeterminados
                            vm_details.append({
                                "vm_id": vm_id,
                                "vm_name": vm_name,
                                "server_id": server_id,
                                "server_name": server_name,
                                "vcpus": 1,  # valor por defecto
                                "ram": 1024,  # valor por defecto
                                "disk": 10.0  # valor por defecto
                            })
                    
                    # Extraer información de uso de recursos por servidor
                    if 'resource_usage' in api_content and 'servers' in api_content['resource_usage']:
                        for server in api_content['resource_usage']['servers']:
                            server_id = server['id']
                            server_usage[server_id] = {
                                "server_name": server['name'],
                                "vcpus": {
                                    "used": server['resources']['vcpus']['used'],
                                    "total": server['resources']['vcpus']['total'],
                                    "percent": server['resources']['vcpus']['percent']
                                },
                                "ram": {
                                    "used": server['resources']['ram']['used'],
                                    "total": server['resources']['ram']['total'],
                                    "percent": server['resources']['ram']['percent']
                                },
                                "disk": {
                                    "used": server['resources']['disk']['used'],
                                    "total": server['resources']['disk']['total'],
                                    "percent": server['resources']['disk']['percent']
                                },
                                "vms": server['vms_count']
                            }
                else:
                    # No hay asignaciones en la respuesta API, usar la información local
                    Logger.warning("No hay asignaciones en la respuesta API, reconstruyendo datos")
                    self._extract_placement_data_from_result(result, vm_details, server_usage)
            else:
                # No hay respuesta API, usar datos locales
                Logger.info("No hay datos de API disponibles, usando datos calculados localmente")
                self._extract_placement_data_from_result(result, vm_details, server_usage)
            
            # Verificar que tenemos datos para visualizar
            if not vm_details or not server_usage:
                Logger.error("No hay datos suficientes para visualizar")
                Logger.debug(f"vm_details: {vm_details}")
                Logger.debug(f"server_usage: {server_usage}")
                return
            
            # Crear figura más grande para incluir 4 gráficos
            fig = plt.figure(figsize=(16, 12))
            fig.text(0.5, 0.02, f"Generado: {timestamp_text}", 
                    ha='center', fontsize=8, style='italic', color='gray')

            # Configurar subplots
            ax1 = plt.subplot2grid((2, 2), (0, 0))  # vCPUs
            ax2 = plt.subplot2grid((2, 2), (0, 1))  # RAM
            ax3 = plt.subplot2grid((2, 2), (1, 0))  # Disco
            ax4 = plt.subplot2grid((2, 2), (1, 1))  # Asignación de VMs
            
            # Ordenar servidores por ID
            server_ids = sorted(server_usage.keys())
            server_names = [server_usage[sid]["server_name"] for sid in server_ids]
            
            # Graficar uso de vCPUs
            cpu_used = [server_usage[sid]["vcpus"]["used"] for sid in server_ids]
            cpu_total = [server_usage[sid]["vcpus"]["total"] for sid in server_ids]
            
            ax1.bar(server_names, cpu_total, color='lightblue', alpha=0.6, label='Total')
            ax1.bar(server_names, cpu_used, color='blue', label='Usado')
            ax1.set_title('Uso de vCPUs por Servidor')
            ax1.set_ylabel('vCPUs')
            ax1.set_xlabel('Servidores')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Graficar uso de RAM en GB
            ram_used = [server_usage[sid]["ram"]["used"] / 1024 for sid in server_ids]  # Convertir a GB
            ram_total = [server_usage[sid]["ram"]["total"] / 1024 for sid in server_ids]  # Convertir a GB
            
            ax2.bar(server_names, ram_total, color='lightgreen', alpha=0.6, label='Total')
            ax2.bar(server_names, ram_used, color='green', label='Usado')
            ax2.set_title('Uso de RAM por Servidor')
            ax2.set_ylabel('RAM (GB)')
            ax2.set_xlabel('Servidores')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Graficar uso de Disco
            disk_used = [server_usage[sid]["disk"]["used"] for sid in server_ids]
            disk_total = [server_usage[sid]["disk"]["total"] for sid in server_ids]
            
            ax3.bar(server_names, disk_total, color='lightcoral', alpha=0.6, label='Total')
            ax3.bar(server_names, disk_used, color='red', label='Usado')
            ax3.set_title('Uso de Disco por Servidor')
            ax3.set_ylabel('Disco (GB)')
            ax3.set_xlabel('Servidores')
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # ---- GRAFICAR ASIGNACIÓN DE VMs (CUARTA GRÁFICA - CORREGIDA) ----
            
            ax4.set_title('Asignación de VMs a Servidores', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Servidores', fontsize=12)
            ax4.set_ylabel('Recursos Asignados (%)', fontsize=12)
            ax4.grid(True, linestyle='--', alpha=0.7)
            
            # Agrupar VMs por servidor
            servers_vm_map = {}
            for vm in vm_details:
                server_name = vm["server_name"]
                if server_name not in servers_vm_map:
                    servers_vm_map[server_name] = []
                servers_vm_map[server_name].append(vm)
            
            # Verificar que tenemos información de servidores
            if not servers_vm_map:
                Logger.error("No hay información de asignación de VMs a servidores")
                # En lugar de retornar, dibujar un texto explicativo
                ax4.text(0.5, 0.5, "No hay datos de asignación disponibles", 
                        ha='center', va='center', fontsize=14, color='red',
                        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=1'))
            else:
                # Nombres de servidores para visualización
                visualization_server_names = sorted(servers_vm_map.keys())
                
                # Definir colores para cada tipo de recurso
                color_vcpus = '#4287f5'  # Azul para vCPUs
                color_ram = '#42f584'    # Verde para RAM
                color_disk = '#f54242'   # Rojo para Disco
                
                # Crear barras para la gráfica de asignación
                bar_width = 0.25
                positions = np.arange(len(visualization_server_names))
                
                # Configurar tipos de recursos y sus colores
                resource_types = ['vCPUs', 'RAM', 'Disco']
                resource_colors = [color_vcpus, color_ram, color_disk]
                resource_positions = [-1, 0, 1]  # Posición relativa de cada barra
                
                # Crear leyenda para tipos de recursos
                legend_elements = [
                    plt.Rectangle((0,0), 1, 1, color=color_vcpus, label='vCPUs'),
                    plt.Rectangle((0,0), 1, 1, color=color_ram, label='RAM'),
                    plt.Rectangle((0,0), 1, 1, color=color_disk, label='Disk')
                ]
                ax4.legend(handles=legend_elements, loc='upper right', title='Tipos de Recursos',
                        fontsize=10, title_fontsize=11, framealpha=0.9, facecolor='white',
                        edgecolor='gray', fancybox=True, shadow=True)
                
                # Para cada tipo de recurso, generar gráfico de barras
                for i, (resource_type, color, position) in enumerate(zip(resource_types, resource_colors, resource_positions)):
                    for server_idx, server_name in enumerate(visualization_server_names):
                        x_pos = positions[server_idx] + position * bar_width
                        server_id = next((sid for sid in server_ids if server_usage[sid]["server_name"] == server_name), None)
                        
                        if server_id is None:
                            continue
                        
                        # Obtener VMs de este servidor
                        server_vms = servers_vm_map.get(server_name, [])
                        
                        # Si no hay VMs en este servidor, continuar al siguiente
                        if not server_vms:
                            continue
                        
                        # Obtener el porcentaje total de uso
                        if resource_type == 'vCPUs':
                            percent_used = server_usage[server_id]["vcpus"]["percent"]
                        elif resource_type == 'RAM':
                            percent_used = server_usage[server_id]["ram"]["percent"]
                        else:  # Disco
                            percent_used = server_usage[server_id]["disk"]["percent"]
                        
                        # Calcular contribuciones de cada VM
                        if server_vms:
                            if resource_type == 'vCPUs':
                                # Asegurar que todas las VMs tienen valores de vCPU
                                for vm in server_vms:
                                    if 'vcpus' not in vm:
                                        vm['vcpus'] = 1  # Valor por defecto
                                
                                total_resource = max(sum(vm.get('vcpus', 0) for vm in server_vms), 1)  # Evitar división por cero
                                vm_percentages = [(vm.get('vcpus', 0) / total_resource * percent_used) if total_resource > 0 else 0 for vm in server_vms]
                            elif resource_type == 'RAM':
                                # Asegurar que todas las VMs tienen valores de RAM
                                for vm in server_vms:
                                    if 'ram' not in vm:
                                        vm['ram'] = 1024  # Valor por defecto
                                
                                total_resource = max(sum(vm.get('ram', 0) for vm in server_vms), 1)  # Evitar división por cero
                                vm_percentages = [(vm.get('ram', 0) / total_resource * percent_used) if total_resource > 0 else 0 for vm in server_vms]
                            else:  # Disco
                                # Asegurar que todas las VMs tienen valores de disco
                                for vm in server_vms:
                                    if 'disk' not in vm:
                                        vm['disk'] = 10.0  # Valor por defecto
                                
                                total_resource = max(sum(vm.get('disk', 0) for vm in server_vms), 1)  # Evitar división por cero
                                vm_percentages = [(vm.get('disk', 0) / total_resource * percent_used) if total_resource > 0 else 0 for vm in server_vms]
                            
                            # Dibujar barras apiladas para cada VM
                            bottom = 0
                            for j, (vm, vm_percent) in enumerate(zip(server_vms, vm_percentages)):
                                # Crear variaciones de color para distinguir mejor las VMs
                                vm_color = color
                                if j % 2 == 0:  # VMs con índice par ligeramente más oscuras
                                    r, g, b = matplotlib.colors.to_rgb(color)
                                    vm_color = (r*0.8, g*0.8, b*0.8)  # 20% más oscuro
                                
                                # Dibujar segmento para esta VM
                                segment = ax4.bar(
                                    x_pos, 
                                    vm_percent, 
                                    bar_width, 
                                    bottom=bottom, 
                                    color=vm_color, 
                                    edgecolor='white',  # Borde blanco para separar VMs
                                    linewidth=1.0       # Grosor del borde aumentado para mejor visibilidad
                                )
                                
                                # Añadir etiqueta con el nombre de la VM
                                if vm_percent > 3.0:  # Solo etiquetar si hay suficiente espacio
                                    y_pos = bottom + vm_percent/2
                                    # Usar rectángulo con fondo semitransparente para mejorar legibilidad
                                    ax4.text(
                                        x_pos, 
                                        y_pos, 
                                        vm["vm_name"], 
                                        ha='center', 
                                        va='center', 
                                        color='white', 
                                        fontsize=9, 
                                        fontweight='bold',
                                        bbox=dict(
                                            boxstyle="round,pad=0.2", 
                                            fc=vm_color, 
                                            ec='white',    # Borde blanco para el texto
                                            alpha=0.95,    # Más opaco para mejor legibilidad
                                            linewidth=0.5  # Borde más fino
                                        )
                                    )
                                
                                bottom += vm_percent
                        else:
                            # Si no hay VMs, mostrar barra vacía con texto explicativo
                            ax4.bar(x_pos, 0, bar_width, color=color, alpha=0.3)
                
            
                # Añadir línea de referencia al 100%
                ax4.axhline(y=100, color='black', linestyle='--', alpha=0.5)
                
                # Configurar ejes
                ax4.set_xticks(positions)
                ax4.set_xticklabels(visualization_server_names, fontsize=11, fontweight='bold')
                ax4.set_ylim(0, 110)  # Dar espacio para etiquetas
                ax4.set_yticks([0, 25, 50, 75, 100])
                ax4.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Crear tabla detallada de las VMs
            table_filename = os.path.join(results_dir, f'resultado_tabla_{timestamp}.png')
            
            # Preparar datos para la tabla
            table_data = []
            for vm in sorted(vm_details, key=lambda x: x["vm_id"]):
                table_data.append([
                    vm["vm_name"],
                    f"{vm.get('vcpus', '-')}",
                    f"{vm.get('ram', 0)/1024:.2f} GB", 
                    f"{vm.get('disk', 0):.1f} GB",     
                    vm["server_name"]
                ])
            
            # Si no hay datos para la tabla, crear una fila con mensaje
            if not table_data:
                table_data.append(["No hay datos disponibles", "-", "-", "-", "-"])
            
            # Crear figura para la tabla
            fig_table = plt.figure(figsize=(10, len(table_data) * 0.5 + 1.5), facecolor='#f9f9f9')
            ax_table = fig_table.add_subplot(111)
            ax_table.axis('off')
            ax_table.axis('tight')
            
            # Crear tabla con estilo mejorado
            table = ax_table.table(
                cellText=table_data,
                colLabels=['Máquina Virtual', 'vCPUs', 'RAM', 'Disco', 'Servidor'],
                loc='center',
                cellLoc='center'
            )
            
            # Aplicar estilos a la tabla
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Aplicar colores y estilos a las celdas
            header_color = '#4472C4'  # Azul para encabezados
            row_colors = ['#E6F0FF', '#FFFFFF']  # Alternar colores de fila
            
            for k, cell in table.get_celld().items():
                cell.set_edgecolor('#BFBFBF')
                
                if k[0] == 0:  # Encabezados
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor(header_color)
                else:
                    cell.set_facecolor(row_colors[(k[0]) % 2])
                    
                    # Alineación para columnas numéricas
                    if k[1] in [1, 2, 3]:  # vCPUs, RAM, Disco
                        cell.get_text().set_horizontalalignment('right')
            
            # Título de la tabla
            ax_table.set_title("Detalle de VMs Asignadas", 
                            fontsize=14, 
                            fontweight='bold',
                            color='#333333',
                            pad=20)
            
            fig_table.text(0.5, 0.02, f"Generado: {timestamp_text}", 
                        ha='center', fontsize=8, style='italic', color='gray')
            
            # Guardar tabla
            plt.tight_layout()
            plt.savefig(table_filename, dpi=300, bbox_inches='tight', facecolor=fig_table.get_facecolor())
            plt.close(fig_table)
            
            # Guardar gráficos principales
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            Logger.success(f"Visualizaciones guardadas en carpeta '{results_dir}':")
            Logger.success(f"- Gráficos: {os.path.basename(filename)}")
            Logger.success(f"- Tabla detallada: {os.path.basename(table_filename)}")
            
        except Exception as e:
            Logger.error(f"Error al generar visualización: {str(e)}")
            Logger.debug(f"Traceback: {traceback.format_exc()}")

    def _extract_placement_data_from_result(self, result, vm_details, server_usage):
        """
        Método auxiliar para extraer datos de visualización directamente del resultado
        
        Args:
            result: Objeto PlacementResult
            vm_details: Lista para almacenar detalles de VMs (se modifica in-place)
            server_usage: Diccionario para almacenar uso de recursos (se modifica in-place)
        """
        # Crear un mapa de servidores a VMs asignadas
        server_to_vms = {}
        for vm_id, server_id in result.assignments.items():
            if server_id not in server_to_vms:
                server_to_vms[server_id] = []
            server_to_vms[server_id].append(vm_id)
        
        # Procesar cada servidor y sus VMs asignadas
        for server_id, vm_ids in server_to_vms.items():
            # Buscar el objeto servidor
            server = next((s for s in self.servers if s.id == server_id), None)
            if not server:
                continue
                
            vcpus_used = 0
            ram_used = 0
            disk_used = 0.0
            
            for vm_id in vm_ids:
                # Buscar el objeto VM en self.slice.vms
                vm = next((v for v in self.slice.vms if v.id == vm_id), None)
                if not vm:
                    continue
                    
                vcpus_used += vm.flavor.vcpus
                ram_used += vm.flavor.ram
                disk_used += vm.flavor.disk
                
                # Guardar información detallada de la VM
                vm_details.append({
                    "vm_id": vm_id,
                    "vm_name": vm.name,
                    "server_id": server_id,
                    "server_name": server.name,
                    "vcpus": vm.flavor.vcpus,
                    "ram": vm.flavor.ram,
                    "disk": vm.flavor.disk
                })
            
            # Calcular uso total para este servidor
            total_vcpus_used = server.used_vcpus + vcpus_used
            total_ram_used = server.used_ram + ram_used
            total_disk_used = server.used_disk + disk_used
            
            # Calcular porcentajes
            total_vcpus_percent = min(100, (total_vcpus_used / server.total_vcpus * 100)) if server.total_vcpus > 0 else 0
            total_ram_percent = min(100, (total_ram_used / server.total_ram * 100)) if server.total_ram > 0 else 0
            total_disk_percent = min(100, (total_disk_used / server.total_disk * 100)) if server.total_disk > 0 else 0
            
            # Almacenar datos del servidor
            server_usage[server_id] = {
                "server_name": server.name,
                "vcpus": {
                    "used": total_vcpus_used,
                    "total": server.total_vcpus,
                    "percent": total_vcpus_percent
                },
                "ram": {
                    "used": total_ram_used,
                    "total": server.total_ram,
                    "percent": total_ram_percent
                },
                "disk": {
                    "used": total_disk_used,
                    "total": server.total_disk,
                    "percent": total_disk_percent
                },
                "vms": len(vm_ids)
            }

class DataManager:
    """
    Gestiona la conversión entre diferentes formatos de datos para VM placement
    """
    
    @staticmethod
    def load_from_database() -> Tuple[List[VirtualMachine], List[PhysicalServer]]:
        """
        Carga los datos desde la base de datos
        Nota: Solo carga flavors y physical servers, no VMs
        
        Returns:
            Tuple con listas de VMs (vacía) y servidores
        """
        try:
            Logger.section("Cargando datos desde la base de datos")
            
            # Obtener flavors activos desde la BD
            db_flavors = DatabaseManager.get_active_flavors()
            
            # Convertir flavors a objetos
            flavors = {}
            for flavor_data in db_flavors:
                flavor = Flavor(
                    id=flavor_data['id'],
                    name=flavor_data['name'],
                    vcpus=flavor_data['vcpus'],
                    ram=flavor_data['ram'],
                    disk=float(flavor_data['disk'])
                )
                flavors[flavor.id] = flavor
                Logger.debug(f"Flavor cargado: {flavor}")
            
            # Obtener servidores físicos desde la BD
            db_servers = DatabaseManager.get_physical_servers()
            
            # Convertir servidores a objetos
            servers = []
            for server_data in db_servers:
                server = PhysicalServer(
                    id=server_data['id'],
                    name=server_data['name'],
                    total_vcpus=server_data['total_vcpus'],
                    total_ram=server_data['total_ram'],
                    total_disk=float(server_data['total_disk']),
                    used_vcpus=server_data['used_vcpus'],
                    used_ram=server_data['used_ram'],
                    used_disk=float(server_data['used_disk'])
                )
                # Validación extra: los recursos usados nunca deben exceder el 60%
                if server.used_vcpus > server.total_vcpus * 0.6:
                    Logger.warning(f"Servidor {server.name}: used_vcpus ({server.used_vcpus}) excede el 60% del total - ajustando")
                    server.used_vcpus = int(server.total_vcpus * 0.6)
                
                if server.used_ram > server.total_ram * 0.6:
                    Logger.warning(f"Servidor {server.name}: used_ram ({server.used_ram}) excede el 60% del total - ajustando")
                    server.used_ram = int(server.total_ram * 0.6)
                
                if server.used_disk > server.total_disk * 0.6:
                    Logger.warning(f"Servidor {server.name}: used_disk ({server.used_disk}) excede el 60% del total - ajustando")
                    server.used_disk = round(server.total_disk * 0.6, 1)
                
                servers.append(server)
                Logger.debug(f"Servidor físico cargado: {server}")
            
            # No estamos cargando VMs - lista vacía
            vms = []
            
            Logger.success(f"Datos cargados correctamente desde la BD: {len(flavors)} flavors y {len(servers)} servidores")
            return vms, servers
            
        except Exception as e:
            Logger.error(f"Error al cargar los datos desde la base de datos: {str(e)}")
            return [], []

    @staticmethod
    def load_from_json(json_data: str) -> Tuple[List[VirtualMachine], List[PhysicalServer]]:
        """
        Carga los datos desde un string JSON
        
        Args:
            json_data: String con los datos en formato JSON
            
        Returns:
            Tuple con listas de VMs y servidores
        """
        try:
            data = json.loads(json_data)
            
            # Cargar flavors primero (pueden ser referenciados por VMs)
            flavors = {}
            if 'flavors' in data:
                for flavor_data in data['flavors']:
                    flavor = Flavor(
                        id=flavor_data['id'],
                        name=flavor_data['name'],
                        vcpus=flavor_data['vcpus'],
                        ram=flavor_data['ram'],
                        disk=float(flavor_data['disk'])
                    )
                    flavors[flavor.id] = flavor
            
            # Cargar VMs
            vms = []
            for vm_data in data.get('virtual_machines', []):
                # Si la VM tiene detalles de flavor embebidos
                if 'flavor' in vm_data and isinstance(vm_data['flavor'], dict):
                    flavor_data = vm_data['flavor']
                    flavor = Flavor(
                        id=flavor_data.get('id', -1),
                        name=flavor_data.get('name', 'Custom'),
                        vcpus=flavor_data.get('vcpus', 1),
                        ram=flavor_data.get('ram', 1024),
                        disk=float(flavor_data.get('disk', 10.0))
                    )
                # Si la VM solo tiene el id del flavor
                elif 'flavor_id' in vm_data and vm_data['flavor_id'] in flavors:
                    flavor = flavors[vm_data['flavor_id']]
                else:
                    Logger.warning(f"VM {vm_data.get('id', 'desconocida')} no tiene flavor válido, asignando default")
                    flavor = Flavor(id=-1, name="default", vcpus=1, ram=1024, disk=10.0)
                
                # Crear la VM con su utilidad (si existe)
                vm = VirtualMachine(
                    id=vm_data['id'],
                    name=vm_data['name'],
                    flavor=flavor,
                    utility=vm_data.get('utility', 0.0)
                )
                vms.append(vm)
            
            # Cargar servidores físicos
            servers = []
            for server_data in data.get('physical_servers', []):
                server = PhysicalServer(
                    id=server_data['id'],
                    name=server_data['name'],
                    total_vcpus=server_data['total_vcpus'],
                    total_ram=server_data['total_ram'],
                    total_disk=float(server_data['total_disk']),  # Asegurar que es float
                    used_vcpus=server_data.get('used_vcpus', 0),
                    used_ram=server_data.get('used_ram', 0),
                    used_disk=float(server_data.get('used_disk', 0.0))  # Asegurar que es float
                )
                servers.append(server)
            
            Logger.success(f"Datos cargados correctamente: {len(vms)} VMs")
            return vms, servers
            
        except Exception as e:
            Logger.error(f"Error al cargar los datos desde JSON: {str(e)}")
            return [], []
    
    @staticmethod
    def generate_test_data(num_vms=10, num_servers=3, seed=None) -> str:
        """
        Genera datos de prueba aleatorios
        
        Args:
            num_vms: Número de VMs a generar
            num_servers: Número de servidores a generar (0 para no generar)
            seed: Semilla para reproducibilidad
                
        Returns:
            String JSON con los datos generados
        """
        Logger.section("GENERANDO DATOS DE PRUEBA")
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Obtener flavors desde la base de datos
        try:
            db_flavors = DatabaseManager.get_active_flavors()
            if not db_flavors:
                Logger.error("No se pudieron obtener flavors de la BD. Usando flavors predefinidos.")
                # Usar flavors predefinidos como fallback
                flavors = []
                flavor_templates = [
                    {"name": "nano", "vcpus": 1, "ram": 512, "disk": 5.0},
                    {"name": "micro", "vcpus": 1, "ram": 1024, "disk": 10.0},
                    {"name": "small", "vcpus": 2, "ram": 2048, "disk": 20.0},
                    {"name": "medium", "vcpus": 4, "ram": 4096, "disk": 40.0},
                    {"name": "large", "vcpus": 8, "ram": 8192, "disk": 80.0},
                    {"name": "xlarge", "vcpus": 16, "ram": 16384, "disk": 160.0}
                ]
                
                for i, template in enumerate(flavor_templates):
                    flavor = {
                        "id": i+1,
                        "name": template["name"],
                        "vcpus": template["vcpus"],
                        "ram": template["ram"],
                        "disk": template["disk"]
                    }
                    flavors.append(flavor)
            else:
                # Convertir flavors de la BD al formato requerido
                flavors = []
                for flavor_data in db_flavors:
                    flavor = {
                        "id": flavor_data['id'],
                        "name": flavor_data['name'],
                        "vcpus": flavor_data['vcpus'],
                        "ram": flavor_data['ram'],
                        "disk": float(flavor_data['disk'])
                    }
                    flavors.append(flavor)
                Logger.success(f"Usando {len(flavors)} flavors obtenidos de la base de datos")
        except Exception as e:
            Logger.error(f"Error al obtener flavors de la BD: {str(e)}. Usando flavors predefinidos.")
            # Usar flavors predefinidos como fallback
            flavors = []
            flavor_templates = [
                {"name": "nano", "vcpus": 1, "ram": 512, "disk": 5.0},
                {"name": "micro", "vcpus": 1, "ram": 1024, "disk": 10.0},
                {"name": "small", "vcpus": 2, "ram": 2048, "disk": 20.0},
                {"name": "medium", "vcpus": 4, "ram": 4096, "disk": 40.0},
                {"name": "large", "vcpus": 8, "ram": 8192, "disk": 80.0},
                {"name": "xlarge", "vcpus": 16, "ram": 16384, "disk": 160.0}
            ]
            
            for i, template in enumerate(flavor_templates):
                flavor = {
                    "id": i+1,
                    "name": template["name"],
                    "vcpus": template["vcpus"],
                    "ram": template["ram"],
                    "disk": template["disk"]
                }
                flavors.append(flavor)
        
        # Generar servidores físicos si num_servers > 0
        physical_servers = []
        if num_servers > 0:
            for i in range(num_servers):
                # Diferentes perfiles de servidores
                if i % 3 == 0:
                    # Perfil alto en CPU
                    total_vcpus = random.randint(32, 64)
                    total_ram = random.randint(32768, 65536)
                    total_disk = round(random.uniform(500, 1000), 1)
                    
                    # Asegurar que used no excede el 60% del total
                    used_vcpus = random.randint(1, int(total_vcpus * 0.6))
                    used_ram = random.randint(1024, int(total_ram * 0.6))
                    used_disk = round(random.uniform(10, total_disk * 0.6), 1)
                    
                    server = {
                        "id": i+1,
                        "name": f"Worker-{i+1}",
                        "total_vcpus": total_vcpus,
                        "total_ram": total_ram,
                        "total_disk": total_disk,
                        "used_vcpus": used_vcpus,
                        "used_ram": used_ram,
                        "used_disk": used_disk
                    }
                elif i % 3 == 1:
                    # Perfil alto en RAM
                    total_vcpus = random.randint(16, 32)
                    total_ram = random.randint(65536, 131072)
                    total_disk = round(random.uniform(500, 1000), 1)
                    
                    # Asegurar que used no excede el 60% del total
                    used_vcpus = random.randint(1, int(total_vcpus * 0.6))
                    used_ram = random.randint(2048, int(total_ram * 0.6))
                    used_disk = round(random.uniform(10, total_disk * 0.6), 1)
                    
                    server = {
                        "id": i+1,
                        "name": f"Worker-{i+1}",
                        "total_vcpus": total_vcpus,
                        "total_ram": total_ram,
                        "total_disk": total_disk,
                        "used_vcpus": used_vcpus,
                        "used_ram": used_ram,
                        "used_disk": used_disk
                    }
                else:
                    # Perfil alto en almacenamiento
                    total_vcpus = random.randint(16, 32)
                    total_ram = random.randint(32768, 65536)
                    total_disk = round(random.uniform(1000, 2000), 1)
                    
                    # Asegurar que used no excede el 60% del total
                    used_vcpus = random.randint(1, int(total_vcpus * 0.6))
                    used_ram = random.randint(1024, int(total_ram * 0.6))
                    used_disk = round(random.uniform(20, total_disk * 0.6), 1)
                    
                    server = {
                        "id": i+1,
                        "name": f"Worker-{i+1}",
                        "total_vcpus": total_vcpus,
                        "total_ram": total_ram,
                        "total_disk": total_disk,
                        "used_vcpus": used_vcpus,
                        "used_ram": used_ram,
                        "used_disk": used_disk
                    }
                physical_servers.append(server)
        
        # Generar VMs usando los flavors obtenidos de la BD
        virtual_machines = []
        vm_name_prefixes = ["VM"]
        for i in range(num_vms):
            if flavors:
                flavor_id = flavors[random.randint(0, len(flavors)-1)]["id"]
                prefix = random.choice(vm_name_prefixes)
                
                vm = {
                    "id": i+1,
                    "name": f"{prefix}-{i+1}",
                    "flavor_id": flavor_id
                }
                virtual_machines.append(vm)
        
        # Crear el objeto JSON completo
        test_data = {
            "flavors": flavors,
            "physical_servers": physical_servers,
            "virtual_machines": virtual_machines
        }
        
        Logger.success(f"Generados {len(virtual_machines)} VMs, {len(flavors)} flavors y {len(physical_servers)} servidores")
        
        return json.dumps(test_data, indent=2)

# ===================== ENDPOINTS DE LA API =====================
@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint para verificar que el servicio está funcionando
    """
    return jsonify({
        "status": "success",
        "message": "VM Placement está vivito y coleando",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/placement', methods=['POST'])
def solve_placement():
    """
    Endpoint para resolver el problema de placement para un slice completo
    
    Cuerpo del request:
    {
        "slice_id": 123,
        "slice_name": "slice-123",
        "user_profile": "investigador",  // alumno, jp, maestro, investigador
        "workload_type": "cpu_intensive", // general, cpu_intensive, memory_intensive, io_intensive
        "virtual_machines": [
            {
                "id": 1,
                "name": "VM-1",
                "flavor_id": 1
            },
            ...
        ]
    }
    
    Returns:
        Response: Resultado del placement
    """
    try:
        Logger.major_section("API: SLICE PLACEMENT")
        
        # Obtener JSON del request
        request_data = request.get_json()
        
        if not request_data:
            Logger.error("No se proporcionaron datos en el request")
            return jsonify({
                "status": "error",
                "message": "Datos de entrada inválidos",
                "details": "No se proporcionaron datos JSON en el cuerpo de la solicitud"
            }), 400
        
        # Validar que exista el campo 'virtual_machines'
        if 'virtual_machines' not in request_data:
            Logger.error("No se proporcionó el campo 'virtual_machines' en el request")
            return jsonify({
                "status": "error",
                "message": "Datos de entrada inválidos",
                "details": "Se requiere el campo 'virtual_machines' en el JSON"
            }), 400
        
        # Obtener los servidores físicos desde la base de datos
        _, servers = DataManager.load_from_database()
        
        if not servers:
            Logger.error("No se pudieron cargar los servidores de la base de datos")
            return jsonify({
                "status": "error",
                "message": "Error al cargar servidores",
                "details": "No se pudieron obtener los servidores físicos de la base de datos"
            }), 500
        
        try:
            # Obtener flavors desde la BD
            db_flavors = DatabaseManager.get_active_flavors()
            
            # Formatear flavors para el JSON
            flavors = []
            for flavor_data in db_flavors:
                flavor = {
                    "id": flavor_data['id'],
                    "name": flavor_data['name'],
                    "vcpus": flavor_data['vcpus'],
                    "ram": flavor_data['ram'],
                    "disk": float(flavor_data['disk'])
                }
                flavors.append(flavor)
            
            # Crear el JSON para cargar los datos
            placement_data = {
                "flavors": flavors,
                "virtual_machines": request_data['virtual_machines']
            }
            
            # Cargar VMs del JSON
            vms, _ = DataManager.load_from_json(json.dumps(placement_data))
            
            if not vms:
                Logger.error("No se pudieron cargar las VMs del request")
                return jsonify({
                    "status": "error",
                    "message": "Error al cargar VMs",
                    "details": "No se pudieron cargar las máquinas virtuales del request"
                }), 400
            
            # Crear objeto Slice con los parámetros recibidos
            slice_id = request_data.get('slice_id', 1)
            slice_name = request_data.get('slice_name', f"slice-{slice_id}")
            user_profile = request_data.get('user_profile', UserProfile.ALUMNO)
            workload_type = request_data.get('workload_type', WorkloadType.GENERAL)
            
            # Crear objeto Slice
            slice = Slice(
                id=slice_id,
                name=slice_name,
                vms=vms,
                user_profile=user_profile,
                workload_type=workload_type
            )
            
            # Mostrar resumen de datos
            Logger.section("Resumen de datos para placement de slice")
            Logger.info(f"Slice: {slice.name} (ID: {slice.id})")
            Logger.info(f"Perfil de usuario: {slice.user_profile}, Tipo de workload: {slice.workload_type}")
            Logger.info(f"Total de VMs en el slice: {len(slice.vms)}")
            
            # Recursos estimados
            estimated_usage = slice.get_estimated_usage()
            Logger.info(f"Uso estimado - vCPUs: {estimated_usage['vcpu']:.1f}, RAM: {estimated_usage['ram']:.1f} MB, Disco: {estimated_usage['disk']:.1f} GB")
            
            # Resolver el problema con el nuevo solucionador de slices
            solver = SliceBasedPlacementSolver(slice, servers)
            result = solver.solve()
            
            # Generar respuesta según el resultado
            if result.success:
                # Convertir resultado a formato JSON para respuesta
                result_dict = result.to_dict(vms=vms, servers=servers)
                
                # Crear respuesta completa del API para pasarla a la visualización
                api_response = {
                    "status": "success",
                    "message": f"Se realizó el placement del slice completo ({len(result.assignments)} VMs)",
                    "content": result_dict
                }
                
                # Intentar generar visualización PASANDO LA RESPUESTA DEL API
                try:
                    # Usamos la visualización existente PASANDO LA RESPUESTA COMPLETA
                    solver.visualize_placement(result, api_response)
                    Logger.success("Visualización generada correctamente")
                except Exception as viz_error:
                    Logger.warning(f"No se pudo generar la visualización: {str(viz_error)}")
                    Logger.debug(f"Traceback: {traceback.format_exc()}")
                
                Logger.success(f"Slice placement completado exitosamente con {len(result.assignments)} VMs asignadas")
                
                return jsonify(api_response), 200
            else:
                # Respuesta detallada del error
                Logger.failed(f"No se pudo realizar el placement del slice: {result.message}")
                
                return jsonify({
                    "status": "error",
                    "message": "Error al resolver el placement del slice",
                    "details": result.message,
                    "unassigned_details": result.unassigned_details if hasattr(result, 'unassigned_details') else []
                }), 400
            
        except Exception as e:
            Logger.error(f"Error procesando los datos del slice: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Error al procesar los datos del slice",
                "details": str(e)
            }), 400
        
    except Exception as e:
        Logger.error(f"Error en el endpoint de placement: {str(e)}")
        Logger.debug(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "message": "Error interno al realizar el placement",
            "details": str(e)
        }), 500

@app.route('/test-data', methods=['GET'])
def get_test_data():
    """
    Endpoint para obtener datos de prueba para slice placement
    """
    try:
        # Generar datos de prueba con flavors de la BD
        num_vms = random.randint(3, 8)
        
        # Generar datos base
        test_data_json = DataManager.generate_test_data(num_vms=num_vms, num_servers=0, seed=None)
        test_data = json.loads(test_data_json)
        
        # Seleccionar perfil de usuario y workload aleatorios
        user_profiles = [UserProfile.ALUMNO, UserProfile.JP, UserProfile.MAESTRO, UserProfile.INVESTIGADOR]
        workload_types = [WorkloadType.GENERAL, WorkloadType.CPU_INTENSIVE, WorkloadType.MEMORY_INTENSIVE, WorkloadType.IO_INTENSIVE]
        
        slice_id = random.randint(1, 1000)
        slice_name = f"test-slice-{slice_id}"
        user_profile = random.choice(user_profiles)
        workload_type = random.choice(workload_types)
        
        # Crear respuesta
        response_data = {
            "slice_id": slice_id,
            "slice_name": slice_name,
            "user_profile": user_profile,
            "workload_type": workload_type,
            "virtual_machines": test_data["virtual_machines"]
        }
        
        Logger.success(f"Datos de prueba para Slice Placement generados con éxito!")
        Logger.info(f"Slice: {slice_name}, Perfil: {user_profile}, Workload: {workload_type}, VMs: {num_vms}")

        return jsonify({
            "status": "success",
            "message": "Datos de prueba generados correctamente",
            "content": response_data
        }), 200
        
    except Exception as e:
        Logger.error(f"Error generando datos de prueba: {str(e)}")
        return jsonify({
            "status": "error",
            "message": "Error al generar datos de prueba",
            "details": str(e)
        }), 500

# ===================== SERVER =====================
if __name__ == '__main__':
    try:
        Logger.major_section("INICIANDO VM PLACEMENT")
        
        # 1. Inicializar componentes
        Logger.info("Inicializando componentes del sistema...")
        
        # Base de datos
        Logger.debug("Conectando a base de datos...")
        try:
            # Inicializar pool de conexiones
            DatabaseManager.init_pool()
            Logger.success("Conexión a base de datos establecida")
        except Exception as db_error:
            Logger.error(f"Error al conectar con la base de datos: {str(db_error)}")
            Logger.warning("El servicio continuará pero algunas funcionalidades pueden no estar disponibles")
        
        # 2. Mostrar datos de ejemplo (misma semilla uwu)
        try:
            example_data = DataManager.generate_test_data(num_vms=5, num_servers=0, seed=69)
            example = json.loads(example_data)
            vm_data = {"virtual_machines": example["virtual_machines"]}
            
            Logger.info("Ejemplo de payload para el endpoint /placement:")
            Logger.info(json.dumps(vm_data, indent=2))
            
        except Exception as example_error:
            Logger.warning(f"No se pudieron generar datos de ejemplo: {str(example_error)}")
        
        # 3. Iniciar servidor Flask
        Logger.section("INICIANDO SERVIDOR WEB")
        Logger.info("Configuración del servidor:")
        Logger.info(f"- Host: {host}")
        Logger.info(f"- Puerto: {port}")
        Logger.info(f"- Debug: {debug}")
        
        Logger.debug("Iniciando servidor Flask...")
        Logger.success(f"VM Placement listo para recibir conexiones")
        
        # Iniciar servidor Flask
        app.run(
            host=host,
            port=port,
            debug=debug
        )
        
    except Exception as e:
        Logger.error(f"Error iniciando el servidor: {str(e)}")
        Logger.debug(f"Traceback: {traceback.format_exc()}")
        import sys
        sys.exit(1)