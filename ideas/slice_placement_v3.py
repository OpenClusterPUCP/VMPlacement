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
import math

# Acceso a datos:
import mysql.connector
from mysql.connector import pooling
# Acceso a Prometheus
import requests
PROMETHEUS_URL = "http://localhost:9090"  # cambia según tu entorno

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

def consultar_prometheus(query):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
        result = response.json()
        print("DEBUG Prometheus response:", result)
        if result["status"] == "success":
            return result["data"]["result"]
        else:
            print("hola")
            return []
    except Exception as e:
        Logger.error(f"Error al consultar Prometheus: {e}")
        return []

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
    
    def get_by_id(flavor_id: int) -> dict:
        flavors = DatabaseManager.get_active_flavors()
        for flavor in flavors:
            if flavor["id"] == flavor_id:
                return {
                    "id": flavor["id"],
                    "name": flavor["name"],
                    "vcpus": flavor["vcpus"],
                    "ram": flavor["ram"],
                    "disk": float(flavor["disk"])
                }
        raise ValueError(f"No se encontró flavor con ID {flavor_id}")

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
            UserProfile.ALUMNO: 1.3,      # 30% de variabilidad
            UserProfile.JP: 1.4,          # 40% de variabilidad
            UserProfile.MAESTRO: 1.2,     # 20% de variabilidad
            UserProfile.INVESTIGADOR: 1.05 # 5% de variabilidad
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
    
    def validacion_seguridad_cpu_vm(self, servidor, usado: Dict[str, float], profile: str, k: float = 2.0) -> bool:
        """
        Verifica que añadir esta VM al servidor no exceda el 95% de la capacidad CPU física.
        
        Parámetros:
            vm: objeto VirtualMachine
            servidor: {'vcpu': total_vcpu, ...}
            usado: {'vcpu': used_vcpu, ...}
            profile: perfil de usuario ('alumno', 'investigador', etc.)
            k: factor estadístico (default 2.0)

        Retorna:
            True si añadir la VM no sobrepasa el 95% de la capacidad CPU
        """
        # Estadísticas de uso de esta VM
        mu, sigma = self.calcular_estadisticas_de_uso(profile)["cpu"]
        capacidad_efectiva_vm = mu + k * sigma

        # Estado actual del servidor
        total_cpu_fisica = servidor.total_vcpus
        usado_cpu = usado["vcpu"]
        maximo_seguro = total_cpu_fisica * 0.95
        demanda_total = usado_cpu + capacidad_efectiva_vm

        Logger.debug(f"[VALIDACIÓN CPU] VM={self.name} | μ+ksigma={capacidad_efectiva_vm:.2f} | usado={usado_cpu:.2f} | total={total_cpu_fisica} | max seguro={maximo_seguro:.2f}")

        return demanda_total <= maximo_seguro

    
    def cabe_en_servidor_vm(self, servidor, usado: Dict[str, float], user_profile: str, k: float = 2.0) -> bool:
        """
        Verifica si esta VM puede ser asignada al servidor dado,
        considerando sobreaprovisionamiento y uso actual.

        Retorna True si cabe, False si no.
        """
        lambdas = UserProfile.get_overprovisioning_limits(user_profile)

        requerimientos = {
            "vcpu": self.flavor['vcpus'],
            "ram": self.flavor['ram'],
            "disk": self.flavor['disk']
        }
        
        capacidades = {
            "vcpu": servidor.total_vcpus,
            "ram": servidor.total_ram,
            "disk": servidor.total_disk
        }
        

        for recurso in ["vcpu", "ram", "disk"]:
            limite = capacidades[recurso] * lambdas[recurso]
            demanda_total = usado[recurso] + requerimientos[recurso]
            if demanda_total > limite:
                return False  # no cabe en ese recurso

        return True

    
    def calcular_estadisticas_de_uso(self, profile: str) -> Dict[str, Tuple[float, float]]: # 3.4. Cálculo de Uso Promedio y Variabilidad de VM
        """
        Calcula el uso promedio (μ) y la desviación estándar (σ) de CPU, RAM y Disco
        según el perfil de usuario.

        Retorna un diccionario con estructura:
        {
            'cpu': (mu_cpu, sigma_cpu),
            'ram': (mu_ram, sigma_ram),
            'disk': (mu_disk, sigma_disk)
        }
        """
        # Obtener factores α y β desde UserProfile
        factores_uso = UserProfile.get_resource_usage_factors(profile)
        beta = UserProfile.get_workload_variability(profile)

        # Uso promedio (μ) = recurso_nominal * α
        mu_cpu = self.flavor['vcpus'] * factores_uso["vcpu"]
        mu_ram = self.flavor['ram'] * factores_uso["ram"]
        mu_disk = self.flavor['disk'] * factores_uso["disk"]
        

        # Desviación estándar (σ) = μ * (β - 1)
        sigma_cpu = mu_cpu * (beta - 1)
        sigma_ram = mu_ram * (beta - 1)
        sigma_disk = mu_disk * (beta - 1)

        return {
            "cpu": (round(mu_cpu, 4), round(sigma_cpu, 4)), #EVALUAR REDONDEO
            "ram": (round(mu_ram, 4), round(sigma_ram, 4)),
            "disk": (round(mu_disk, 4), round(sigma_disk, 4))
        }

@dataclass
class PhysicalServer:
    """
    Representa un servidor físico con sus capacidades, uso actual y métricas de congestión
    """
    id: int
    name: str
    ip: str
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

    def get_usos_actuales(self) -> Dict[str, float]:
        return {
            "vcpu": self.used_vcpus,
            "ram": self.used_ram,
            "disk": self.used_disk
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
        """
        query = """
            SELECT 
                id, 
                name,
                ip, 
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
            import random #RANDOM
            Logger.info(f"Se obtuvieron {len(servers)} servidores físicos desde la BD")
            return servers
        except Exception as e:
            Logger.error(f"Error al obtener servidores físicos: {str(e)}")
            return []
    
    @classmethod
    def obtenerPhysicalServers(cls):
        """ Esta función nos proporciona los servidores físicos (WORKERS) que se encuentren en DB"""
        
        query = """
                SELECT 
                    id, 
                    name,
                    ip, 
                    total_vcpu as total_vcpus, 
                    total_ram, 
                    total_disk
                FROM physical_server
                WHERE server_type = 'worker' AND status = 'active'
                ORDER BY id
            """
        try:
            servers = cls.execute_query(query)
            
            # IMPORTANTE: USO CPU con prometheus
            import random
            for server in servers:
                instance = f"{server['ip']}:9100"  # Para detectar la instancia
                
                # Obtener RAM usada
                ram_query = (
                    f'(avg_over_time(node_memory_MemTotal_bytes{{instance="{instance}"}}[1h]) - '
                    f'avg_over_time(node_memory_MemAvailable_bytes{{instance="{instance}"}}[1h])) / 1024 / 1024'
                )

                ram_result = consultar_prometheus(ram_query)
                if ram_result:
                    server['used_ram'] = int(float(ram_result[0]['value'][1]))  # en MB
                else:
                    Logger.error(f"No se pudo calcular la RAM usada promedio en la instancia: {server['ip']}")
                    
                
                # Obtener disco usado (root "/")
                disk_query = (
                    f'(avg_over_time(node_filesystem_size_bytes{{instance="{instance}", mountpoint="/"}}[1h]) - '
                    f'avg_over_time(node_filesystem_free_bytes{{instance="{instance}", mountpoint="/"}}[1h])) / 1024 / 1024 / 1024'
                )
                disk_result = consultar_prometheus(disk_query)
                if disk_result:
                    server['used_disk'] = round(float(disk_result[0]['value'][1]), 1)  # en GB
                else:
                    Logger.error(f"No se pudo calcular el DISCO usado promedio en la instancia: {server['ip']}")
                    
                # Obtener CPU usada (aproximado como # cores usados). Primero se calcula el porcentaje de tiempo que está en "idle" y luego se calcula el complemento
                cpu_query = (
                    f'count(count by (cpu) (node_cpu_seconds_total{{instance="{instance}"}})) * '
                    f'(1 - avg(rate(node_cpu_seconds_total{{instance="{instance}", mode="idle"}}[1m])))'
                )

                cpu_result = consultar_prometheus(cpu_query)
                if cpu_result:
                    server['used_vcpus'] = round(float(cpu_result[0]['value'][1]), 1)
                else:
                    Logger.error(f"No se pudo calcular CPU usado promedio en la instancia: {server['ip']}")
                    
                Logger.section(f"===================== {server['name']} =====================")
                Logger.info(f"RAM USADA en promedio (1 hora): {ram_result} MB")
                Logger.info(f"DISCO USADO en promedio (1 hora): {disk_result} GB")
                Logger.info(f"CPU USADO en promedio (1 hora): {cpu_result}")
            Logger.info(f"Se obtuvieron {len(servers)} servidores físicos desde la BD")
            return servers
        except Exception as e:
            Logger.error(f"Error al obtener servidores físicos: {str(e)}")
            return []
    

@dataclass
class Slice:
    """Representa un conjunto de VMs relacionadas a desplegar como unidad"""
    id: int
    name: str
    vms: List[VirtualMachine]
    user_profile: str = UserProfile.ALUMNO
    workload_type: str = WorkloadType.GENERAL
    
    def calcular_capacidad_efectica_slice(self, k: float = 2.0) -> Dict[str, float]: # 3.5.1 Enfoque Independiente Implementado
        """
        Calcula la capacidad esperada del slice usando enfoque independiente
        (media + k * desviación estándar) para CPU, RAM y Disco.
        Usa el Teorema del Límite Central.
        """
        suma_mu = {"cpu": 0.0, "ram": 0.0, "disk": 0.0}
        suma_sigma2 = {"cpu": 0.0, "ram": 0.0, "disk": 0.0}

        for vm in self.vms:
            stats = vm.calcular_estadisticas_de_uso(self.user_profile)
            for recurso in ["cpu", "ram", "disk"]:
                mu, sigma = stats[recurso]
                suma_mu[recurso] += mu
                suma_sigma2[recurso] += sigma**2

        # Calcular capacidad efectiva = μ + σ
        resultados = {}
        for recurso in ["cpu", "ram", "disk"]:
            mu_total = suma_mu[recurso]
            sigma_total = math.sqrt(suma_sigma2[recurso])
            capacidad_efectiva = mu_total + sigma_total
            resultados[recurso] = round(capacidad_efectiva, 2) # EVALUAR REDONDEO

        return resultados #Array de capacidad efectiva por recurso
    
    def calcular_demanda_efectica_slice_correlacionado(self) -> Dict[str, float]: # 3.5.1 Enfoque Correlacionado
        """
        Calcula la demanda esperada del slice usando enfoque correlacionado de VMs
        (media + desviación estándar) para CPU, RAM y Disco.
        Usa el Teorema del Límite Central.
        """
        suma_mu = {"cpu": 0.0, "ram": 0.0, "disk": 0.0}
        suma_sigma = {"cpu": 0.0, "ram": 0.0, "disk": 0.0}

        for vm in self.vms:
            stats = vm.calcular_estadisticas_de_uso(self.user_profile)
            for recurso in ["cpu", "ram", "disk"]:
                mu, sigma = stats[recurso]
                suma_mu[recurso] += mu
                suma_sigma[recurso] += sigma

        # Calcular demanda efectiva = μ + σ
        resultados = {}
        for recurso in ["cpu", "ram", "disk"]:
            mu_total = suma_mu[recurso]
            sigma_total = math.sqrt(suma_sigma[recurso])
            demanda_efectiva = mu_total + sigma_total
            resultados[recurso] = round(demanda_efectiva, 4) # EVALUAR REDONDEO

        return resultados #Array de capacidad efectiva por recurso
    
    from typing import Tuple, Optional

    def cabe_en_servidor(self, servidor: Dict[str, float], usado: Dict[str, float], alfa: float = 0.5) -> Tuple[bool, Optional[str], Dict[str, float]]: #5.2
        """
        Verifica si este slice puede colocarse en un servidor físico dado,
        usando el modelo estadístico independiente con sobreaprovisionamiento SLA.

        Parámetros:
            - servidor: {'vcpu': ..., 'ram': ..., 'disk': ...}  ← total
            - usado:    {'vcpu': ..., 'ram': ..., 'disk': ...}  ← usado
            - k: factor estadístico (confianza)

        Retorna:
            - bool: si el slice cabe
            - str: recurso que falla (si aplica)
            - dict: márgenes disponibles por recurso
        """
        
        # 1. Obtener capacidad efectiva (μ + σ)
        #capacidad_efectiva = self.calcular_capacidad_efectica_slice(k)
        demanda_efectiva = self.calcular_demanda_efectica_slice_correlacionado()
        Logger.info(f"Demanda efectiva del slice: {demanda_efectiva}")
        # 2. Obtener límites de sobreaprovisionamiento por perfil
        lambdas = UserProfile.get_overprovisioning_limits(self.user_profile)
        Logger.info(f"lambdas: {lambdas}")
        # 3. Evaluar para cada recurso
        margenes = {}
        for r in ["vcpu", "ram", "disk"]:
            # Mapeo: en UserProfile los keys son cpu/ram/disk, pero aquí usamos vcpu/ram/disk
            capacidad_total = servidor[r] #Capacidad total del recurso de base de datos del server
            capacidad_usada = usado[r] #Capacidad usada, calculado por Prometheus
            #capacidad_asignada = server
            limite = capacidad_total * lambdas[r]
            if r == "vcpu":
                demanda = capacidad_usada + demanda_efectiva["cpu"]
            else:
                demanda = capacidad_usada + demanda_efectiva[r]
                
            margen = round(limite - demanda, 2)
            if r == "vcpu":
                margenes["cpu"] = margen
            else:
                margenes[r] = margen
            
            if demanda > limite:
                return False, r, margenes # No cabe
            
        return True, None, margenes # Cabe
    
    def capacidad_usada_function(self, recurso, servidor: Dict[str, float]):
        capacidad_usada = 0
        if(recurso == "vcpu"):
            capacidad_usada = servidor.used_vcpu
        elif(recurso=="ram"):
            capacidad_usada = servidor.used_disk
        else:
            capacidad_usada = servidor.used_ram
        
        return capacidad_usada
        
    
    def cabe_en_servidor_nuevo(self, servidor: Dict[str, float], usado: Dict[str, float], alfa: float = 0.5) -> Tuple[bool, Optional[str], Dict[str, float]]: #5.2
        """
        Verifica si este slice puede colocarse en un servidor físico dado,
        usando el modelo estadístico independiente con sobreaprovisionamiento SLA.

        Parámetros:
            - servidor: {'vcpu': ..., 'ram': ..., 'disk': ...}  ← total
            - usado:    {'vcpu': ..., 'ram': ..., 'disk': ...}  ← usado
            - k: factor estadístico (confianza)

        Retorna:
            - bool: si el slice cabe
            - str: recurso que falla (si aplica)
            - dict: márgenes disponibles por recurso
        """
        
        # 1. Obtener capacidad efectiva (μ + σ)
        #capacidad_efectiva = self.calcular_capacidad_efectica_slice(k)
        demanda_efectiva = self.calcular_demanda_efectica_slice_correlacionado()
        Logger.info(f"Demanda efectiva del slice: {demanda_efectiva}")
        # 2. Obtener límites de sobreaprovisionamiento por perfil
        lambdas = UserProfile.get_overprovisioning_limits(self.user_profile)
        Logger.info(f"lambdas: {lambdas}")
        # 3. Evaluar para cada recurso
        margenes = {}
        for r in ["vcpu", "ram", "disk"]:
            # Mapeo: en UserProfile los keys son cpu/ram/disk, pero aquí usamos vcpu/ram/disk
            capacidad_total = servidor[r] #Capacidad total del recurso de base de datos del server
            capacidad_usada = usado[r] #Capacidad usada, calculado por Prometheus
            capacidad_asignada = self.capacidad_usada_function(r, servidor) #Capacidad Asignada del server a otros slices. CORREGIR con modelo INDEPENDIENTE
            capacidad_disponible_server = lambdas[r]*(capacidad_total-alfa*capacidad_asignada-(1-alfa)*capacidad_usada)
            if r == "vcpu":
                demanda = demanda_efectiva["cpu"] #demanda en cpu del slice
            else:
                demanda = demanda_efectiva[r] #demanda en ram o disk del slice
                
            margen = round(capacidad_disponible_server - demanda, 4) #margen entre demanda la capacidad disponible del server y la demanda del slice
            if r == "vcpu":
                margenes["cpu"] = margen
            else:
                margenes[r] = margen
            
            if demanda > capacidad_disponible_server:
                return False, r, margenes # No cabe
        return True, None, margenes # Cabe
    
    def validacion_seguridad_cpu(self, servidor: Dict[str, float], usado: Dict[str, float], k: float = 2.0) -> bool: # OS, etc.
        """
        Verifica que el uso total estimado de CPU no exceda el 95% de la capacidad física total,
        independientemente del sobreaprovisionamiento (seguridad del sistema).

        Parámetros:
            servidor: {'vcpu': total_vcpu, ...}
            usado: {'vcpu': used_vcpu, ...}
            k: factor estadístico de confianza (default 2.0 para 95%)

        Retorna:
            True si pasa la validación de seguridad, False si se excede el 95%
        """
        # 1. Calcular μ_s^cpu y σ_s^cpu
        suma_mu = 0.0
        suma_sigma2 = 0.0
        for vm in self.vms:
            mu, sigma = vm.calcular_estadisticas_de_uso(self.user_profile)["cpu"]
            suma_mu += mu
            suma_sigma2 += sigma**2

        sigma_total = math.sqrt(suma_sigma2)
        capacidad_efectiva_cpu = suma_mu + k * sigma_total

        # 2. Capacidad física real límite (95%)
        total_cpu_fisica = servidor["vcpu"]
        usado_cpu = usado["vcpu"]
        maximo_seguro = total_cpu_fisica * 0.95 # Se setea en 95%
        demanda_total = usado_cpu + capacidad_efectiva_cpu
        print(f"capacidad_efectiva cpu: {capacidad_efectiva_cpu}")
        print(f"demanda total: {demanda_total}")
        print(f"maximo seguro: {maximo_seguro}")
        return demanda_total <= maximo_seguro

    
    def calcular_congestion_y_Q(self, servidor: Dict[str, float], usado: Dict[str, float]) -> Tuple[float, float]: # 6.2.1,  6.3.2, 6.4.1 
        """
        Calcula:
        - Congestión ponderada del slice s en el servidor i
        - Factor de espera en cola Q basado en la congestión de CPU
        Parámetros:
            servidor: {'vcpu': total_vcpu, 'ram': ..., 'disk': ...}
            usado:    {'vcpu': used_vcpus, 'ram': ..., 'disk': ...}
        Retorna:
            (congestion_ponderada, Q)
        """
        # 1. Obtener demanda estimada μ + kσ para cada recurso
        demanda_total = self.calcular_capacidad_efectica_slice(k)
        # 2. Obtener límites de sobreaprovisionamiento
        lambdas = UserProfile.get_overprovisioning_limits(self.user_profile)

        # 3. Calcular congestión por recurso
        congestion = {}
        for r in ["vcpu", "ram", "disk"]:
            total = servidor[r]
            usado_actual = usado[r]
            limite_aprovisionado = total * lambdas[r]
            if r == "vcpu":
                numerador = usado_actual + demanda_total["cpu"]
            else:
                numerador = usado_actual + demanda_total[r]
            if r == "vcpu":
                congestion["cpu"] = round(numerador / limite_aprovisionado, 4)
            else:
                congestion[r] = round(numerador / limite_aprovisionado, 4)
        # 4. Obtener pesos del tipo de workload
        pesos = WorkloadType.get_resource_weights(self.workload_type)
        # 5. Congestión ponderada
        rho_weighted = (
            pesos["vcpu"] * congestion["cpu"] +
            pesos["ram"] * congestion["ram"] +
            pesos["disk"] * congestion["disk"]
        )
        print(f"rho_weighted: {rho_weighted}")
        # 6. Cálculo de Q en base a congestión CPU (sigmoide)
        a = 12
        b = 0.7
        rho_cpu = congestion["cpu"]
        Q = 1 / (1 + math.exp(-a * (rho_cpu - b)))
        return round(rho_weighted, 4), round(Q, 4)
    
    def calcular_score_ponderado(self, servidor: Dict[str, float], usado: Dict[str, float]) -> float:
        """
        Calcula el score ponderado de asignación del slice en un servidor físico.

        Parámetros:
        - servidor: {'vcpu': total_vcpu, 'ram': ..., 'disk': ...} se extrae CAPACIDAD TOTAL DEL SERVER y CAPACIDAD ASIGNADA A LOS SLICES QUE CORRE
        - usado: {'vcpu': used_vcpus, 'ram': ..., 'disk': ...} se extrae la CAPACIDAD USADA (en un promedio de 1 hora)
        - phi: performance relativa del servidor (por defecto 1.0)
        - num_vms_mismo_sitio: número de VMs del slice ya ubicadas en el mismo servidor o zona
        - k: factor de confianza estadística (default 1.0), no usaremos

        Retorna:
        - score final ponderado (float)
        """
        
        # 1. Verificar si cabe (restricción)
        cabe, _, margenes = self.cabe_en_servidor_nuevo(servidor, usado, alfa=0.5) #Se setea alfa en 0.5, CORREGIR
        if not cabe:
            Logger.error("No cabe")
            return 0.0
        else:
            Logger.info(f"Cabe slice en servidor y estos son los márgenes: {margenes}")
        # 2. Calcular congestión ponderada y Q (usa función ya implementada)
        rho_weighted, Q = self.calcular_congestion_y_Q(servidor, usado)
        Logger.info(f"rho: {rho_weighted}")
        Logger.info(f"Q: {Q}")
        # 3. Factores individuales
        f_cong = 1 - rho_weighted
        f_queue = 1 - Q
        # 4. Score ponderado final
        score = 0.4 * f_cong + 0.3 * f_queue
        return round(score, 4)
    
    def seleccionar_mejor_servidor(self, servidores: List[PhysicalServer]) -> Tuple[Optional[PhysicalServer], float]:

        """
        Selecciona el mejor servidor (máximo score ponderado) para un slice.
        Parámetros:
        - servidores: lista de servidores con claves 'vcpu', 'used_vcpus', etc.
        Retorna:
        - Lista de tuplas [tupla:(mejor_servidor_dict, score)]
        """
        lista_tupla_servidor_score = [] #Lista de tuplas (server,score)

        for server in servidores:
            servidor = {
                "vcpu": server.total_vcpus,
                "ram": server.total_ram,
                "disk": server.total_disk,
                "used_disk": server.used_disk,
                "used_ram": server.used_ram,
                "used_vcpus": server.used_vcpus
                
            }
            #Obtenido con prometheus
            usado = {
                "vcpu": server.used_vcpus,
                "ram": server.used_ram,
                "disk": server.used_disk
            }

            # Calculamos el score del slice en el server
            score = self.calcular_score_ponderado(servidor, usado) # No usamos k
            Logger.info(f"{server.name}, score: {score}")
            tupla_server_score = (server, score)
            lista_tupla_servidor_score.append(tupla_server_score)
        return lista_tupla_servidor_score

    
    def asignar_vms_distribuidas(self, servidores: List[PhysicalServer], k: float = 2.0) -> Tuple[Optional[PhysicalServer], float]:
        """
        Selecciona el mejor servidor (máximo score ponderado) para este slice.

        Parámetros:
        - servidores: lista de servidores con claves 'vcpu', 'used_vcpus', etc.
        - k: nivel de confianza estadística

        Retorna:
        - { servidor_id: [vm1, vm2, ...] }
        """
        
        Logger.section("VMs instanciadas")
        Logger.info(self.vms)
        
        ga = 0
        servidores_scores = []
        # 1. Calcular score en cada servidor
        for server in servidores:
            servidor = {
                "vcpu": server.total_vcpus,
                "ram": server.total_ram,
                "disk": server.total_disk
            }
            #Obtenido con prometheus
            usado = {
                "vcpu": server.used_vcpus,
                "ram": server.used_ram,
                "disk": server.used_disk
            }
            # Aquí puedes incluir lógica para contar VMs en mismo sitio si aplica
            num_vms_mismo_sitio = 0
            phi = 1.0  # o 1.0 por defecto (puede variar)
            Logger.section(f"Resultados de {server.name}")
            Logger.info(f"Recursos nominales: {servidor}")
            Logger.info(f"Recursos usados: {usado}")
            score = self.calcular_score_ponderado(servidor, usado, phi, num_vms_mismo_sitio, k)
            Logger.success(f"score server {ga + 1}: {score}")
            servidores_scores.append((server, score))
            ga += 1
        # 2. Ordenar servidores por score descendente
        servidores_ordenados = sorted(servidores_scores, key=lambda x: x[1], reverse=True)   
        # 3. Asignar VMs
        asignaciones = {srv.id: [] for srv, _ in servidores_ordenados}
        recursos_usados = {
                            srv.id: {
                                "vcpu": srv.used_vcpus,
                                "ram": srv.used_ram,
                                "disk": srv.used_disk
                            }
                            for srv, _ in servidores_ordenados
                        }
        for vm in self.vms:
            asignada = False
            for servidor, _ in servidores_ordenados:
                usado = recursos_usados[servidor.id]
                cabe = vm.cabe_en_servidor_vm(servidor, usado, self.user_profile, k = 2.0)  # usa perfil
                seguridad_95 = vm.validacion_seguridad_cpu_vm(servidor, usado, self.user_profile, k = 2.0)
                if cabe or seguridad_95:
                    asignaciones[servidor.id].append(vm)
                    # actualizar recursos usados
                    usado['vcpu'] += vm.flavor['vcpus']
                    usado['ram'] += vm.flavor['ram']
                    usado['disk'] += vm.flavor['disk']
                    asignada = True
                    break
            if not asignada:
                Logger.warning(f"⚠️ VM {vm.name} no pudo ser asignada a ningún servidor.")
       
        return asignaciones
    
    def get_resource_weights(self) -> Dict[str, float]:
        """Obtiene pesos relativos para cada tipo de recurso según el workload"""
        return WorkloadType.get_resource_weights(self.workload_type)
    


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

# ===================== SUBMÓDULOS =====================

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
            db_servers = DatabaseManager.obtenerPhysicalServers()
            # Convertir servidores a objetos
            servers = []
            for server_data in db_servers:
                server = PhysicalServer(
                    id=server_data['id'],
                    name=server_data['name'],
                    ip = server_data['ip'],
                    total_vcpus=server_data['total_vcpus'],
                    total_ram=server_data['total_ram'],
                    total_disk=float(server_data['total_disk']),
                    used_vcpus=server_data['used_vcpus'], # Por defecto en db es cero ya que se saca de Prometheus
                    used_ram=server_data['used_ram'],
                    used_disk=float(server_data['used_disk'])
                )
                
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

# Implementamos slice_placement
@app.route('/slice_placement_v2', methods=['POST']) # Aún no terminada
def slice_placement_v2():
    try:
        data = request.get_json()
        # 1. Validar entrada mínima
        required_fields = ["slice_id", "slice_name", "user_profile", "workload_type", "virtual_machines"]
        if not all(field in data for field in required_fields):
            return jsonify({"status": "error", "message": "Faltan campos requeridos"}), 400
        # 2. Crear VMs
        vms = []
        for vm_data in data["virtual_machines"]:
            if "id" not in vm_data or "name" not in vm_data or "flavor_id" not in vm_data:
                return jsonify({"status": "error", "message": "Formato de VM incorrecto"}), 400

            # Buscar el flavor correspondiente (esto debería venir de BD en entorno real)
            flavor = Flavor.get_by_id(vm_data["flavor_id"])  # Debes tener esta función implementada
            if not flavor:
                return jsonify({"status": "error", "message": f"Flavor ID {vm_data['flavor_id']} no encontrado"}), 404

            vm = VirtualMachine(id=vm_data["id"], name=vm_data["name"], flavor=flavor)
            vms.append(vm)
        # 3. Construir el Slice
        slice_obj = Slice(
            id=data["slice_id"],
            name=data["slice_name"],
            vms=vms,
            user_profile=data["user_profile"].lower(),
            workload_type=data["workload_type"].lower()
        )
        # 4. Obtener servidores físicos
        _, servidores = DataManager.load_from_database()
        if not servidores:
            return jsonify({"status": "error", "message": "No hay servidores disponibles"}), 500
        # 5. Resolver placement
        asignaciones = slice_obj.asignar_vms_distribuidas(servidores, k = 2.0)
        
        servidores_asignados = {sid: vms for sid, vms in asignaciones.items() if vms}
        
        # Formatear respuesta
        respuesta = {
            "status": "success",
            "slice_id": data["slice_id"],
            "slice_name": data["slice_name"],
            "asignaciones": []
        }
        
        if not servidores_asignados:
            return jsonify({"status": "fail", "message": "Ningún servidor puede alojar las VMs del slice"}), 200

        for servidor_id, vms in servidores_asignados.items():
            servidor = next((s for s in servidores if s.id == servidor_id), None)
            if not servidor:
                continue
            respuesta["asignaciones"].append({
                "servidor_id": servidor.id,
                "servidor_name": servidor.name,
                "servidor_ip": servidor.ip,
                "vms_asignadas": [vm.name for vm in vms]
            })

        Logger.success(f"Slice {data['slice_name']} asignado parcialmente a {len(servidores_asignados)} servidor(es)")
        return jsonify(respuesta), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/slice_placement_v1', methods=['POST'])
def slice_placement_v1():
    try:
        data = request.get_json()
        # 1. Validar entrada mínima
        required_fields = ["slice_id", "slice_name", "user_profile", "workload_type", "virtual_machines"]
        if not all(field in data for field in required_fields):
            return jsonify({"status": "error", "message": "Faltan campos requeridos"}), 400
        # 2. Crear VMs
        vms = []
        for vm_data in data["virtual_machines"]:
            if "id" not in vm_data or "name" not in vm_data or "flavor_id" not in vm_data:
                return jsonify({"status": "error", "message": "Formato de VM incorrecto"}), 400

            # Buscar el flavor correspondiente (esto debería venir de BD en entorno real)
            flavor = Flavor.get_by_id(vm_data["flavor_id"])  # Debes tener esta función implementada
            if not flavor:
                return jsonify({"status": "error", "message": f"Flavor ID {vm_data['flavor_id']} no encontrado"}), 404

            vm = VirtualMachine(id=vm_data["id"], name=vm_data["name"], flavor=flavor)
            vms.append(vm)
        # 3. Construir el Slice
        slice_obj = Slice(
            id=data["slice_id"],
            name=data["slice_name"],
            vms=vms,
            user_profile=data["user_profile"].lower(),
            workload_type=data["workload_type"].lower()
        )
        # 4. Obtener servidores físicos
        _, servidores = DataManager.load_from_database()
        if not servidores:
            return jsonify({"status": "error", "message": "No hay servidores disponibles"}), 500
        # 5. Resolver placement
        #mejor_servidor, mejor_score = slice_obj.seleccionar_mejor_servidor(servidores, k = 2.0)
        
        
        
        #Solo valida que no se puede ingresar el slice a 1 servidor, falta reubicarlos
        #LOGICA DE REUBICACION DE VMS EN SERVIDORES
        
        if not mejor_servidor:
            return jsonify({"status": "fail", "message": "Ningún servidor puede alojar el slice"}), 200

        return jsonify({
            "status": "success",
            "slice_id": data["slice_id"],
            "slice_name": data["slice_name"],
            "asignado_a": mejor_servidor.name,
            "server_ip": mejor_servidor.ip,
            "score": mejor_score
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/slice_placement_v3', methods=['POST'])
def slice_placement_v3():
    try:
        data = request.get_json()
        # 1. Validar entrada mínima, se agrega ZA: LOW TIER, PREMIUM, VIP, parámetros de slice básicos: ID y NAME, USUARIO
        required_fields = ["slice_id", "slice_name", "user_profile", "availability_zone", "workload_type", "virtual_machines"] 
        if not all(field in data for field in required_fields):
            return jsonify({"status": "error", "message": "Faltan campos requeridos"}), 400
        # 2. Crear VMs
        vms = []
        for vm_data in data["virtual_machines"]:
            if "id" not in vm_data or "name" not in vm_data or "flavor_id" not in vm_data:
                return jsonify({"status": "error", "message": "Formato de VM incorrecto"}), 400
            # Buscar el flavor correspondiente (esto debería venir de BD en entorno real)
            flavor = Flavor.get_by_id(vm_data["flavor_id"])  # Debes tener esta función implementada
            if not flavor:
                return jsonify({"status": "error", "message": f"Flavor ID {vm_data['flavor_id']} no encontrado"}), 404

            vm = VirtualMachine(id=vm_data["id"], name=vm_data["name"], flavor=flavor)
            vms.append(vm)
        # 3. Construir el Slice
        slice_obj = Slice(
            id=data["slice_id"], 
            name=data["slice_name"], 
            vms=vms, #Lista de VMs
            user_profile=data["user_profile"].lower(),
            workload_type=data["workload_type"].lower()
        )
        
            #user_profile: Define los alfas (uso promedio)
            #workload_type: Define los lambdas (factores de sobreaprovisionamiento)
        
        # 4. Obtener servidores físicos
        _, servidores = DataManager.load_from_database()
        if not servidores:
            return jsonify({"status": "error", "message": "No hay servidores disponibles"}), 500
        print(servidores)
        
        # 5. Resolver placement
        lista_tupla_servidor_score = slice_obj.seleccionar_mejor_servidor(servidores)
        for tupla_servidor_score in lista_tupla_servidor_score:
            print(f"Servidor: {tupla_servidor_score[0].name}, score: {tupla_servidor_score[1]}")
        
        # Validar si la lista está vacía o si todos los scores son 0.0
        if not lista_tupla_servidor_score or all(score == 0.0 for _, score in lista_tupla_servidor_score):
            return jsonify({"status": "fail", "message": "Ningún servidor puede alojar el slice"}), 200

        """return jsonify({
            "status": "success",
            "slice_id": data["slice_id"],
            "slice_name": data["slice_name"],
            "asignado_a": mejor_servidor.name,
            "server_ip": mejor_servidor.ip,
            "score": mejor_score
        }), 200"""
        return jsonify({"status": "ok", "message": "Slice procesado"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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