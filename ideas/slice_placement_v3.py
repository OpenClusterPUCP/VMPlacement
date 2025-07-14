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
from typing import List, Dict, Tuple, Optional, Union, Any

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
PROMETHEUS_URL = "http://10.20.12.98:9090"  # cambia según tu entorno
# eureka
from py_eureka_client import eureka_client
from itertools import combinations

# ===================== CONFIGURACIÓN DE FLASK =====================
app = Flask(__name__)
host = '0.0.0.0'
port = 6001
debug = False

# ===================== CONFIGURACIÓN DE EUREKA =====================
eureka_server = "http://10.20.12.214:8080"

# Configuración de Eureka
eureka_client.init(
    eureka_server=eureka_server,
    app_name="vm-placement",
    instance_port=port,
    instance_host="localhost",
    renewal_interval_in_secs=30,
    duration_in_secs=90,
)

def get_service_instance(service_name: str) -> dict:
    """
    Obtiene información de la instancia de un servicio registrado en Eureka.
    """
    try:
        Logger.debug(f"Buscando instancia de servicio: {service_name}")
        instance = eureka_client.get_client().applications.get_application(service_name)
        if not instance or not instance.up_instances:
            Logger.error(f"Servicio {service_name} no encontrado en Eureka")
            return None

        instance = instance.up_instances[0]
        service_info = {
            'ipAddr': instance.ipAddr,
            'port': instance.port.port,
            'hostName': instance.hostName
        }

        Logger.debug(f"Instancia encontrada: {json.dumps(service_info, indent=2)}")
        return service_info

    except Exception as e:
        Logger.error(f"Error obteniendo instancia de {service_name}: {str(e)}")
        return None

# ===================== CONFIGURACIÓN BD =====================
DB_CONFIG = {
    "host": "10.20.12.214",
    "user": "root",
    "password": "Branko",
    "port": 4000,
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
    mu_vcpu_used: float = 0.0
    mu_ram_used: float = 0.0
    mu_disk_used: float = 0.0
    desv_vcpu_used: float = 0.0
    desv_ram_used: float = 0.0
    desv_disk_used: float = 0.0
    
    def __str__(self):
        return f"VM {self.id}: {self.name} - {self.flavor.name} (Utilidad: {self.utility:.2f})"
    
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
            "vm_name": self.name,
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
    availability_zone: int = 0 #ID de ZA
    
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
    def obtenerPhysicalServers(cls):
        """ Esta función nos proporciona los servidores físicos (WORKERS) que se encuentren en DB"""
        
        query = """
                SELECT 
                    id, 
                    name,
                    ip, 
                    total_vcpu as total_vcpus, 
                    total_ram, 
                    total_disk,
                    availability_zone
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
    
    @classmethod
    def calcular_capacidad_asignada_con_modelo_compuesto(cls, servidor_id: int) -> Dict[str, float]:
        """
        Calcula la capacidad asignada (μ y σ) por recurso en un servidor físico.
        Aplica:
        - Enfoque correlacionado dentro de cada slice (μ_slice = suma(μ_vm), σ_slice = suma(σ_vm))
        - Enfoque independiente entre slices (σ_total = sqrt(suma(σ_slice^2)), μ_total = suma(μ_slice))
        """
        query = """
            SELECT slice,
                SUM(COALESCE(mu_vcpu_used, 0)) AS mu_cpu_slice,
                SUM(COALESCE(mu_ram_used, 0)) AS mu_ram_slice,
                SUM(COALESCE(mu_disk_used, 0)) AS mu_disk_slice,
                SUM(COALESCE(desv_vcpu_used, 0)) AS sigma_cpu_slice,
                SUM(COALESCE(desv_ram_used, 0)) AS sigma_ram_slice,
                SUM(COALESCE(desv_disk_used, 0)) AS sigma_disk_slice
            FROM virtual_machine
            WHERE physical_server = %s
            AND status IN ('running', 'preparing')
            GROUP BY slice;
        """

        try:
            resultados = cls.execute_query(query, (servidor_id,))
            #Si no hay resultados significa que el server no tiene instancia alguna
            if not resultados:
                return {
                    "mu_cpu": 0.0, "sigma_cpu": 0.0,
                    "mu_ram": 0.0, "sigma_ram": 0.0,
                    "mu_disk": 0.0, "sigma_disk": 0.0
                }

            mu_total = {"cpu": 0.0, "ram": 0.0, "disk": 0.0}
            sigma_total_cuad = {"cpu": 0.0, "ram": 0.0, "disk": 0.0}

            for row in resultados:
                # μ_total ← suma directa
                mu_total["cpu"] += row["mu_cpu_slice"] or 0
                mu_total["ram"] += row["mu_ram_slice"] or 0
                mu_total["disk"] += row["mu_disk_slice"] or 0

                # σ_total ← sqrt(suma de σ_slice²)
                sigma_total_cuad["cpu"] += (row["sigma_cpu_slice"] or 0) ** 2
                sigma_total_cuad["ram"] += (row["sigma_ram_slice"] or 0) ** 2
                sigma_total_cuad["disk"] += (row["sigma_disk_slice"] or 0) ** 2

            # σ final: raíz cuadrada de suma de cuadrados
            sigma_total = {
                "cpu": math.sqrt(sigma_total_cuad["cpu"]),
                "ram": math.sqrt(sigma_total_cuad["ram"]),
                "disk": math.sqrt(sigma_total_cuad["disk"])
            }

            return {
                "mu_cpu": round(mu_total["cpu"], 4),
                "sigma_cpu": round(sigma_total["cpu"], 4),
                "mu_ram": round(mu_total["ram"], 4),
                "sigma_ram": round(sigma_total["ram"], 4),
                "mu_disk": round(mu_total["disk"], 4),
                "sigma_disk": round(sigma_total["disk"], 4)
            }

        except Exception as e:
            Logger.error(f"Error al calcular capacidad asignada compuesta: {str(e)}")
            return {
                "mu_cpu": 0.0, "sigma_cpu": 0.0,
                "mu_ram": 0.0, "sigma_ram": 0.0,
                "mu_disk": 0.0, "sigma_disk": 0.0
            }


    
    

@dataclass
class Slice:
    """Representa un conjunto de VMs relacionadas a desplegar como unidad"""
    id: int
    name: str
    vms: List[VirtualMachine]
    user_profile: str = UserProfile.ALUMNO
    workload_type: str = WorkloadType.GENERAL
    
    def calcular_demanda_efectica_slice_correlacionado(self) -> Dict[str, float]: # 3.5.1 Enfoque Correlacionado
        """
        Calcula la demanda esperada del slice usando enfoque correlacionado de VMs
        (media + desviación estándar) para CPU, RAM y Disco.
        Usa el Teorema del Límite Central.
        """
        suma_mu = {"cpu": 0.0, "ram": 0.0, "disk": 0.0}
        suma_sigma = {"cpu": 0.0, "ram": 0.0, "disk": 0.0}
        lista_stats_vms = [] #Lista para capturar los stats de cada VM (media de uso y desv estándar)

        for vm in self.vms:
            stats = vm.calcular_estadisticas_de_uso(self.user_profile)
            for recurso in ["cpu", "ram", "disk"]:
                mu, sigma = stats[recurso]
                suma_mu[recurso] += mu
                suma_sigma[recurso] += sigma
            lista_stats_vms.append(stats)

        # Calcular demanda efectiva = μ + σ
        resultados = {}
        for recurso in ["cpu", "ram", "disk"]:
            mu_total = suma_mu[recurso]
            sigma_total = math.sqrt(suma_sigma[recurso])
            demanda_efectiva = mu_total + sigma_total
            resultados[recurso] = round(demanda_efectiva, 4) # EVALUAR REDONDEO
            
        self.lista_stats_vms = lista_stats_vms #GUARDA COMO ATRIBUTO DE INSTANCIA
        return resultados, lista_stats_vms #Array de capacidad efectiva por recurso
    
    from typing import Tuple, Optional
    
    def capacidades_function(self, recurso, servidor: PhysicalServer):
        """
            Función que calcula las capacidades de los servidores
        """
        #FALTA CORREGIR LA CAPACIDAD ASIGNADA
        capacidades_asignadas = DatabaseManager.calcular_capacidad_asignada_con_modelo_compuesto(servidor.id)
        if(recurso == "vcpu"):
            capacidad_usada = servidor.used_vcpus
            capacidad_total = servidor.total_vcpus
            capacidad_asignada = capacidades_asignadas["mu_cpu"] + capacidades_asignadas["sigma_cpu"]
        elif(recurso=="ram"):
            capacidad_usada = servidor.used_ram
            capacidad_total = servidor.total_ram
            capacidad_asignada = capacidades_asignadas["mu_ram"] + capacidades_asignadas["sigma_ram"]
        else:
            capacidad_usada = servidor.used_disk
            capacidad_total = servidor.total_disk
            capacidad_asignada = capacidades_asignadas["mu_disk"] + capacidades_asignadas["sigma_disk"]
        return capacidad_total, capacidad_usada, capacidad_asignada
    
    def obtener_demanda_slice(self, r: str, demanda_efectiva: Dict[str, float]):
        if r == "vcpu":
            demanda_slice = demanda_efectiva["cpu"] #demanda en cpu del slice
        else:
            demanda_slice = demanda_efectiva[r] #demanda en ram o disk del slice
        return demanda_slice
        
    
    def cabe_en_servidor(self, servidor: PhysicalServer, alfa: float = 0.5) -> Tuple[bool, Optional[str], Dict[str, float]]: #5.2
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
        congestion_dicc = {"congestion_cpu": 0.0, "congestion_ram": 0.0, "congestion_disk": 0.0}
        
        # 1. Obtener demanda efectiva (μ + σ) y lista_stats_vms para guardar en db si se instancia el slice
        demanda_efectiva, lista_stats_vms = self.calcular_demanda_efectica_slice_correlacionado()
        Logger.info(f"Demanda efectiva del slice: {demanda_efectiva}")
        Logger.info(f"Lista stats: {lista_stats_vms}")
        # 2. Obtener límites de sobreaprovisionamiento por perfil
        lambdas = UserProfile.get_overprovisioning_limits(self.user_profile)
        Logger.info(f"lambdas: {lambdas}")
        # 3. Evaluar para cada recurso
        margenes = {}
        for r in ["vcpu", "ram", "disk"]:
            #Capacidades calculadas del server
            capacidad_total, capacidad_usada, capacidad_asignada = self.capacidades_function(r, servidor)
            Logger.info(f"[{r}] Total: {capacidad_total}, Usada: {capacidad_usada}, Asignada: {capacidad_asignada}")
            #Capacidad disponible del server
            capacidad_disponible_server = lambdas[r]*(capacidad_total-alfa*capacidad_asignada-(1-alfa)*capacidad_usada)
            #Demanda del slice en base al recurso
            demanda_slice = self.obtener_demanda_slice(r, demanda_efectiva)
            #margen del recurso analizado (cpu, ram, disk)
            margen = round(capacidad_disponible_server - demanda_slice, 4) #margen entre demanda la capacidad disponible del server y la demanda del slice
            if r == "vcpu":
                margenes["cpu"] = margen
            else:
                margenes[r] = margen
            #Verificamos restricción
            if demanda_slice > capacidad_disponible_server:
                return False, r, margenes, None # No cabe
            
            #Calculamos congestión del recurso
            congestion = (lambdas[r]*(alfa*capacidad_asignada+(1-alfa)*capacidad_usada) + demanda_slice)/(capacidad_total*lambdas[r])
            if r == "vcpu":
                congestion_dicc["congestion_cpu"] = round(congestion, 4)
            elif r == "ram":
                congestion_dicc["congestion_ram"] = round(congestion, 4)
            else:
                congestion_dicc["congestion_disk"] = round(congestion, 4)
            
        return True, None, margenes, congestion_dicc # Cabe

    
    def calcular_congestion_y_Q(self, congestion_dicc: Dict[str, float]) -> Tuple[float, float]: # 6.2.1,  6.3.2, 6.4.1 
        
        # 1. Obtener pesos del tipo de workload
        pesos = WorkloadType.get_resource_weights(self.workload_type)
        # 2. Congestión ponderada
        rho_weighted = (
            pesos["vcpu"] * congestion_dicc["congestion_cpu"] +
            pesos["ram"] * congestion_dicc["congestion_ram"] +
            pesos["disk"] * congestion_dicc["congestion_disk"]
        )
        print(f"rho_weighted: {rho_weighted}")
        # 3. Cálculo de Q en base a congestión CPU (sigmoide)
        a = 12
        b = 0.7
        rho_cpu = congestion_dicc["congestion_cpu"]
        Q = 1 / (1 + math.exp(-a * (rho_cpu - b)))
        return round(rho_weighted, 4), round(Q, 4)
    
    def calcular_score_ponderado(self, servidor: PhysicalServer) -> float:
        """
        Calcula el score ponderado de asignación del slice en un servidor físico.
        Parámetros:
        - servidor: PhysicalServer
        Retorna:
        - score final ponderado (float)
        """
        
        # 1. Verificar si cabe (restricción)
        cabe, _, margenes, congestion_dicc = self.cabe_en_servidor(servidor, alfa=0.5) #Se setea alfa en 0.5
        if not cabe:
            Logger.error("No cabe")
            return 0.0
        else:
            Logger.info(f"Cabe slice en servidor y estos son los márgenes: {margenes}")
        # 2. Calcular congestión ponderada y Q (usa función ya implementada)
        rho_weighted, Q = self.calcular_congestion_y_Q(congestion_dicc)
        Logger.info(f"rho: {rho_weighted}")
        Logger.info(f"Q: {Q}")
        # 3. Factores individuales
        f_cong = 1 - rho_weighted #factor de congestión
        f_queue = 1 - Q #factor de cola
        # 4. Score ponderado final
        score = 0.6 * f_cong + 0.4 * f_queue
        return round(score, 4)

    def score_servidores(self, servidores: List[PhysicalServer]) -> List[Tuple[PhysicalServer, float]]:
        """
        Evalúa todos los servidores y devuelve una lista de tuplas (servidor, score).

        Parámetros:
        - servidores: lista de servidores físicos

        Retorna:
        - Lista de tuplas (servidor, score)
        """
        lista_tupla_servidor_score = []

        for server in servidores:
            # Calculamos el score del slice en el server
            score = self.calcular_score_ponderado(server)
            Logger.info(f"{server.name}, score: {score}")
            tupla_server_score = (server, score)
            lista_tupla_servidor_score.append(tupla_server_score)

        return lista_tupla_servidor_score

    def get_resource_weights(self) -> Dict[str, float]:
        """Obtiene pesos relativos para cada tipo de recurso según el workload"""
        return WorkloadType.get_resource_weights(self.workload_type)

    def aplicar_fase_2_greedy(self, servidores: List[PhysicalServer]) -> Dict[str, Any]:
        """
        FASE 2: Algoritmo greedy para distribución de slice en múltiples servidores
        Empieza desde tamaño N-1 porque sabemos que N VMs juntas no caben (FASE 1 falló)
        """
        Logger.major_section("INICIANDO FASE 2 - ALGORITMO GREEDY")
        Logger.info(f"FASE 1 falló con {len(self.vms)} VMs. Iniciando fragmentación...")
        Logger.info(f"Slice: {self.name}")
        Logger.info(f"Perfil: {self.user_profile}, Workload: {self.workload_type}")
        Logger.info(f"Servidores disponibles: {len(servidores)}")

        # Inicialización
        vms_pendientes = self.vms.copy()
        asignaciones_finales = {}
        servidores_usados = []
        iteracion = 1

        # Bucle principal
        while vms_pendientes:
            Logger.section(f"=== ITERACIÓN {iteracion} ===")
            Logger.info(f"VMs pendientes: {len(vms_pendientes)} - {[vm.name for vm in vms_pendientes]}")

            # Servidores disponibles (excluyendo los ya usados)
            servidores_disponibles = [s for s in servidores if s.id not in servidores_usados]
            Logger.info(f"Servidores disponibles: {[s.name for s in servidores_disponibles]}")

            if not servidores_disponibles:
                Logger.error("No hay más servidores disponibles")
                return self._generar_respuesta_fallo("No hay servidores disponibles para VMs restantes")

            # CLAVE: Determinar tamaño máximo para esta iteración
            tamaño_maximo = len(vms_pendientes)
            if iteracion == 1:
                # En primera iteración, saltar tamaño completo (ya probado en FASE 1)
                tamaño_maximo -= 1
                Logger.info(f"Primera iteración: saltando tamaño {len(vms_pendientes)} (ya probado en FASE 1)")

            if tamaño_maximo < 1:
                Logger.error("No es posible fragmentar más - tamaño máximo < 1")
                return self._generar_respuesta_fallo("Fragmentación agotada")

            # Encontrar el mejor grupo para esta iteración
            mejor_asignacion = self._encontrar_mejor_grupo_acotado(
                vms_pendientes, servidores_disponibles, tamaño_maximo
            )

            if not mejor_asignacion:
                Logger.error("No se encontró asignación viable para las VMs restantes")
                return self._generar_respuesta_fallo("No se encontró asignación viable")

            # Procesar la asignación
            vms_asignadas, servidor_seleccionado, score_grupo = mejor_asignacion

            Logger.success(f"✅ Asignación encontrada:")
            Logger.info(f"   Servidor: {servidor_seleccionado.name}")
            Logger.info(f"   VMs: {[vm.name for vm in vms_asignadas]} ({len(vms_asignadas)} VMs)")
            Logger.info(f"   Score: {score_grupo}")

            # Guardar asignación
            asignaciones_finales[servidor_seleccionado.id] = {
                "servidor": servidor_seleccionado,
                "vms": vms_asignadas,
                "score": score_grupo
            }

            # Actualizar estado
            vms_pendientes = [vm for vm in vms_pendientes if vm not in vms_asignadas]
            servidores_usados.append(servidor_seleccionado.id)
            iteracion += 1

        Logger.major_section("FASE 2 COMPLETADA EXITOSAMENTE")
        return self._generar_respuesta_exito(asignaciones_finales)

    def _encontrar_mejor_grupo_acotado(self, vms_pendientes: List[VirtualMachine],
                                       servidores_disponibles: List[PhysicalServer],
                                       tamaño_maximo: int) -> Optional[Tuple]:
        """
        Encuentra el mejor grupo de VMs con límite de tamaño máximo
        """
        Logger.subsection(f"Búsqueda del mejor grupo (tamaño máximo: {tamaño_maximo})")

        # Calcular límite para evitar redundancia
        n_vms = len(vms_pendientes)
        tamaño_minimo = max(1, n_vms // 2)  # No evaluar más allá de la mitad

        if tamaño_maximo > tamaño_minimo:
            Logger.info(f"Evaluando tamaños {tamaño_maximo} hasta {tamaño_minimo} (evitando redundancia)")
            rango_evaluacion = range(tamaño_maximo, tamaño_minimo - 1, -1)
        else:
            Logger.info(f"Evaluando tamaños pequeños: {tamaño_maximo} hasta 1")
            rango_evaluacion = range(tamaño_maximo, 0, -1)

        # Búsqueda descendente por tamaño
        for tamaño in rango_evaluacion:
            Logger.debug(f"Evaluando grupos de tamaño {tamaño}")

            mejores_de_este_tamaño = []
            num_combinaciones = math.comb(len(vms_pendientes), tamaño)
            Logger.debug(f"  Combinaciones a evaluar: {num_combinaciones}")

            # Generar y evaluar todas las combinaciones de este tamaño
            for i, combinacion_vms in enumerate(combinations(vms_pendientes, tamaño)):
                if i % 10 == 0 and num_combinaciones > 20:  # Log de progreso
                    Logger.debug(f"  Progreso: {i + 1}/{num_combinaciones}")

                # Evaluar esta combinación en todos los servidores disponibles
                for servidor in servidores_disponibles:
                    resultado = self._evaluar_combinacion_fase2(list(combinacion_vms), servidor)

                    if resultado["cabe"]:
                        mejores_de_este_tamaño.append((
                            list(combinacion_vms),
                            servidor,
                            resultado["score"]
                        ))

            # Si encontré soluciones de este tamaño, retorno la mejor por score
            if mejores_de_este_tamaño:
                mejor_solucion = max(mejores_de_este_tamaño, key=lambda x: x[2])  # Por score puro
                vms_mejor, servidor_mejor, score_mejor = mejor_solucion

                Logger.info(f"✅ Mejor solución tamaño {tamaño}:")
                Logger.info(f"   Score: {score_mejor}")
                Logger.info(f"   Servidor: {servidor_mejor.name}")
                Logger.info(f"   VMs: {[vm.name for vm in vms_mejor]}")

                return mejor_solucion
            else:
                Logger.debug(f"❌ No hay soluciones viables para tamaño {tamaño}")

        Logger.warning("No se encontró ninguna solución viable")
        return None

    def _evaluar_combinacion_fase2(self, vms_combinacion: List[VirtualMachine],
                                   servidor: PhysicalServer) -> Dict[str, Any]:
        """
        Evalúa una combinación específica de VMs en un servidor específico
        """
        try:
            # Crear slice temporal con esta combinación
            slice_temporal = Slice(
                id=f"temp-{hash(tuple(vm.id for vm in vms_combinacion))}",
                name=f"temporal-{len(vms_combinacion)}vms",
                vms=vms_combinacion,
                user_profile=self.user_profile,
                workload_type=self.workload_type
            )

            # Usar la función de scoring existente
            score = slice_temporal.calcular_score_ponderado(servidor)

            return {
                "score": score,
                "cabe": score > 0,
                "vms": vms_combinacion,
                "servidor": servidor,
                "tamaño": len(vms_combinacion)
            }

        except Exception as e:
            Logger.error(f"Error evaluando combinación: {str(e)}")
            return {
                "score": 0.0,
                "cabe": False,
                "vms": vms_combinacion,
                "servidor": servidor,
                "tamaño": len(vms_combinacion)
            }

    def _generar_respuesta_exito(self, asignaciones_finales: Dict) -> Dict[str, Any]:
        """
        Genera respuesta de éxito para FASE 2
        """
        # Calcular estadísticas
        total_vms = len(self.vms)
        servidores_utilizados = len(asignaciones_finales)
        grupo_mas_grande = max(len(asig["vms"]) for asig in asignaciones_finales.values())
        localidad_maxima = grupo_mas_grande / total_vms

        # Calcular score promedio ponderado
        score_total_ponderado = 0
        for asignacion in asignaciones_finales.values():
            peso = len(asignacion["vms"]) / total_vms
            score_total_ponderado += asignacion["score"] * peso

        # Construir lista de asignaciones para respuesta
        asignaciones_detalle = []
        for server_id, asignacion in asignaciones_finales.items():
            servidor = asignacion["servidor"]
            vms = asignacion["vms"]

            asignaciones_detalle.append({
                "server_id": servidor.id,
                "server_name": servidor.name,
                "server_ip": servidor.ip,
                "vms": [{"id": vm.id, "name": vm.name} for vm in vms],
                "vms_count": len(vms),
                "score": asignacion["score"]
            })

        return {
            "status": "success",
            "slice_id": self.id,
            "slice_name": self.name,
            "fase": 2,
            "user_profile": self.user_profile,
            "workload_type": self.workload_type,
            "estadisticas": {
                "total_vms": total_vms,
                "servidores_utilizados": servidores_utilizados,
                "grupo_mas_grande": grupo_mas_grande,
                "localidad_maxima": round(localidad_maxima * 100, 2),  # Porcentaje
                "score_promedio_ponderado": round(score_total_ponderado, 4)
            },
            "asignaciones": asignaciones_detalle,
            "message": f"Slice distribuido exitosamente en {servidores_utilizados} servidores con localidad máxima de {grupo_mas_grande} VMs"
        }

    def _generar_respuesta_fallo(self, razon: str) -> Dict[str, Any]:
        """
        Genera respuesta de fallo para FASE 2
        """
        return {
            "status": "fail",
            "slice_id": self.id,
            "slice_name": self.name,
            "fase": 2,
            "message": f"FASE 2 falló: {razon}",
            "razon": razon
        }
    

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
                    used_disk=float(server_data['used_disk']),
                    availability_zone=server_data['availability_zone'] #Zona de Disponibilidad (ID)
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


@app.route('/slice_placement_v3', methods=['POST'])
def slice_placement_v3():
    try:
        Logger.major_section("API: VM PLACEMENT REQUEST")
        data = request.get_json()

        # 1. Validar entrada
        required_fields = ["slice_id", "slice_name", "user_profile", "availability_zone", "workload_type",
                           "virtual_machines"]
        if not all(field in data for field in required_fields):
            return jsonify({
                "status": "fail",
                "message": "Faltan campos requeridos",
                "slice_id": data.get("slice_id"),
                "slice_name": data.get("slice_name"),
                "servers_info": []
            }), 400

        # 2. Crear VMs
        vms = []
        for vm_data in data["virtual_machines"]:
            if "id" not in vm_data or "name" not in vm_data or "flavor_id" not in vm_data:
                return jsonify({
                    "status": "fail",
                    "message": "Formato de VM incorrecto",
                    "slice_id": data.get("slice_id"),
                    "slice_name": data.get("slice_name"),
                    "servers_info": []
                }), 400

            # Buscar el flavor correspondiente
            flavor = Flavor.get_by_id(vm_data["flavor_id"])
            if not flavor:
                return jsonify({
                    "status": "fail",
                    "message": f"Flavor ID {vm_data['flavor_id']} no encontrado",
                    "slice_id": data.get("slice_id"),
                    "slice_name": data.get("slice_name"),
                    "servers_info": []
                }), 400

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
            return jsonify({
                "status": "fail",
                "message": "No hay servidores disponibles",
                "slice_id": data["slice_id"],
                "slice_name": data["slice_name"],
                "servers_info": []
            }), 500

        # 4.1 Filtrar servidores por zona de disponibilidad
        zona_requerida_id = int(data["availability_zone"])
        Logger.section(f"Filtrando servidores en la zona: {zona_requerida_id}")
        servidores_zona_requerida = [srv for srv in servidores if srv.availability_zone == zona_requerida_id]

        if not servidores_zona_requerida:
            return jsonify({
                "status": "fail",
                "message": f"No hay servidores disponibles en la zona ID {zona_requerida_id}",
                "slice_id": data["slice_id"],
                "slice_name": data["slice_name"],
                "availability_zone": zona_requerida_id,
                "servers_info": []
            }), 200

        # 5. FASE 1: Intentar colocar slice completo en un servidor
        Logger.section("FASE 1: Intentando colocación en servidor único")
        lista_tupla_servidor_score = slice_obj.score_servidores(servidores_zona_requerida)

        # Encontrar el mejor servidor
        best_server = None
        best_score = 0.0

        for servidor, score in lista_tupla_servidor_score:
            Logger.debug(f"Servidor: {servidor.name}, score: {score}")
            if score > best_score:
                best_score = score
                best_server = servidor

        # 6. Generar respuesta según resultado de FASE 1
        if best_server and best_score > 0:
            Logger.success(f"FASE 1 exitosa: Slice asignado a {best_server.name}")

            # Obtener estadísticas de las VMs
            stats_por_vm = slice_obj.lista_stats_vms

            # Obtener capacidades del servidor
            capacidades_asignadas = DatabaseManager.calcular_capacidad_asignada_con_modelo_compuesto(best_server.id)

            # Formato de respuesta exitosa
            vms_response = []
            for i, vm in enumerate(slice_obj.vms):
                vm_stats = stats_por_vm[i]
                vms_response.append({
                    "vm_id": vm.id,
                    "vm_name": vm.name,
                    "mu_vcpu_used": vm_stats["cpu"][0],
                    "mu_ram_used": vm_stats["ram"][0],
                    "mu_disk_used": vm_stats["disk"][0],
                    "desv_vcpu_used": vm_stats["cpu"][1],
                    "desv_ram_used": vm_stats["ram"][1],
                    "desv_disk_used": vm_stats["disk"][1]
                })

            return jsonify({
                "status": "success",
                "assignments": [
                    {
                        "server_id": best_server.id,
                        "server_name": best_server.name,
                        "server_ip": best_server.ip,
                        "current_usage": {
                            "vcpus": best_server.used_vcpus,
                            "ram": best_server.used_ram,
                            "disk": best_server.used_disk
                        },
                        "current_assigned_capacity": {
                            "mu_cpu": capacidades_asignadas["mu_cpu"],
                            "sigma_cpu": capacidades_asignadas["sigma_cpu"],
                            "mu_ram": capacidades_asignadas["mu_ram"],
                            "sigma_ram": capacidades_asignadas["sigma_ram"],
                            "mu_disk": capacidades_asignadas["mu_disk"],
                            "sigma_disk": capacidades_asignadas["sigma_disk"]
                        },
                        "vms": vms_response
                    }
                ]
            }), 200

        else:
            Logger.warning("FASE 1 falló: Intentando FASE 2...")

            # FASE 2: Distribución en múltiples servidores
            resultado_fase2 = slice_obj.aplicar_fase_2_greedy(servidores_zona_requerida)

            if resultado_fase2["status"] == "success":
                # Convertir resultado de FASE 2 al formato unificado
                assignments = []
                for asignacion in resultado_fase2["asignaciones"]:
                    servidor_id = asignacion["server_id"]
                    servidor = next(s for s in servidores_zona_requerida if s.id == servidor_id)

                    # Obtener capacidades del servidor
                    capacidades_asignadas = DatabaseManager.calcular_capacidad_asignada_con_modelo_compuesto(
                        servidor.id)

                    # Preparar VMs response
                    vms_response = []
                    for vm_info in asignacion["vms"]:
                        # Encontrar la VM original para obtener sus stats
                        vm_original = next(vm for vm in slice_obj.vms if vm.id == vm_info["id"])
                        vm_stats = vm_original.calcular_estadisticas_de_uso(slice_obj.user_profile)

                        vms_response.append({
                            "vm_id": vm_info["id"],
                            "vm_name": vm_info["name"],
                            "mu_vcpu_used": vm_stats["cpu"][0],
                            "mu_ram_used": vm_stats["ram"][0],
                            "mu_disk_used": vm_stats["disk"][0],
                            "desv_vcpu_used": vm_stats["cpu"][1],
                            "desv_ram_used": vm_stats["ram"][1],
                            "desv_disk_used": vm_stats["disk"][1]
                        })

                    assignments.append({
                        "server_id": servidor.id,
                        "server_name": servidor.name,
                        "server_ip": servidor.ip,
                        "current_usage": {
                            "vcpus": servidor.used_vcpus,
                            "ram": servidor.used_ram,
                            "disk": servidor.used_disk
                        },
                        "current_assigned_capacity": {
                            "mu_cpu": capacidades_asignadas["mu_cpu"],
                            "sigma_cpu": capacidades_asignadas["sigma_cpu"],
                            "mu_ram": capacidades_asignadas["mu_ram"],
                            "sigma_ram": capacidades_asignadas["sigma_ram"],
                            "mu_disk": capacidades_asignadas["mu_disk"],
                            "sigma_disk": capacidades_asignadas["sigma_disk"]
                        },
                        "vms": vms_response
                    })

                return jsonify({
                    "status": "success",
                    "assignments": assignments
                }), 200

            else:
                # Tanto FASE 1 como FASE 2 fallaron
                Logger.error("Ambas fases fallaron")

                # Obtener información de todos los servidores evaluados
                servers_info = []
                for servidor in servidores_zona_requerida:
                    capacidades_asignadas = DatabaseManager.calcular_capacidad_asignada_con_modelo_compuesto(
                        servidor.id)
                    servers_info.append({
                        "server_id": servidor.id,
                        "server_name": servidor.name,
                        "server_ip": servidor.ip,
                        "current_usage": {
                            "vcpus": servidor.used_vcpus,
                            "ram": servidor.used_ram,
                            "disk": servidor.used_disk
                        },
                        "current_assigned_capacity": {
                            "mu_cpu": capacidades_asignadas["mu_cpu"],
                            "sigma_cpu": capacidades_asignadas["sigma_cpu"],
                            "mu_ram": capacidades_asignadas["mu_ram"],
                            "sigma_ram": capacidades_asignadas["sigma_ram"],
                            "mu_disk": capacidades_asignadas["mu_disk"],
                            "sigma_disk": capacidades_asignadas["sigma_disk"]
                        }
                    })

                return jsonify({
                    "status": "fail",
                    "message": "No se puede colocar el slice en ningún servidor disponible en la zona de disponibilidad",
                    "slice_id": data["slice_id"],
                    "slice_name": data["slice_name"],
                    "availability_zone": zona_requerida_id,
                    "servers_info": servers_info
                }), 200

    except Exception as e:
        Logger.error(f"Error en VM Placement: {str(e)}")
        Logger.debug(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "fail",
            "message": f"Error interno: {str(e)}",
            "slice_id": data.get("slice_id") if 'data' in locals() else None,
            "slice_name": data.get("slice_name") if 'data' in locals() else None,
            "servers_info": []
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
        Logger.success(f"VM Placement listo para recibir conexiones en Eureka")

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