# ==============================================================================
# | ARCHIVO: vm_placement.py
# ==============================================================================
# | DESCRIPCIÓN:
# | Módulo API REST que implementa un algoritmo de asignación de máquinas virtuales
# | a servidores físicos (VM Placement) utilizando técnicas de optimización MILP
# | (Mixed-Integer Linear Programming). Gestiona la asignación óptima maximizando
# | la utilidad mientras cumple con restricciones de recursos.
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
# |    - VirtualMachine: Representa una máquina virtual con su flavor
# |    - PhysicalServer: Representa un servidor físico con capacidades
# |    - PlacementResult: Resultado de la colocación de VMs
# |
# | 3. SUBMÓDULOS PRINCIPALES
# |    - DatabaseManager: Gestiona conexiones y consultas a la BD
# |    - UtilityCalculator: Calcula la utilidad de asignar una VM
# |    - VMPlacementSolver: Resuelve el problema de placement con MILP
# |    - DataManager: Conversión entre formatos de datos
# |
# | 4. ALGORITMO DE PLACEMENT
# |    - Formulación MILP con restricciones de recursos
# |    - Cálculo de utilidad para optimización
# |    - Validación de recursos disponibles
# |    - Análisis de asignaciones y restricciones
# |
# | 5. VISUALIZACIÓN DE RESULTADOS
# |    - Generación de gráficas de uso de recursos
# |    - Visualización de asignaciones VM-servidor
# |    - Exportación a imágenes en carpeta 'resultados'
# |
# | 6. API ENDPOINTS
# |    - /health: Verificación del servicio
# |    - /placement: Endpoint para resolver el problema del placement uwu
# |    - /test-data: Generación de datos de prueba
# ==============================================================================

# ===================== IMPORTACIONES =====================
from flask import Flask, request, jsonify

# Networking y sistema:
import os
import traceback

# Utilidades:
import json
import random
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

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
            # para los campos que tienen 0 o 0.0 (máximo 60% del total)
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
                
                # Validación extra: verificar que los valores asignados no excedan el 60%
                # Si por alguna razón los valores en la BD exceden el 60%, los ajustamos
                if server['used_vcpus'] > int(server['total_vcpus'] * 0.6):
                    server['used_vcpus'] = int(server['total_vcpus'] * 0.6)
                    Logger.warning(f"Server {server['name']}: used_vcpus ajustado a {server['used_vcpus']} (60% del total)")
                    
                if server['used_ram'] > int(server['total_ram'] * 0.6):
                    server['used_ram'] = int(server['total_ram'] * 0.6)
                    Logger.warning(f"Server {server['name']}: used_ram ajustado a {server['used_ram']} (60% del total)")
                    
                if float(server['used_disk']) > float(server['total_disk']) * 0.6:
                    server['used_disk'] = round(float(server['total_disk']) * 0.6, 1)
                    Logger.warning(f"Server {server['name']}: used_disk ajustado a {server['used_disk']} (60% del total)")
            
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
class PhysicalServer:
    """
    Representa un servidor físico con sus capacidades y uso actual
    """
    id: int
    name: str
    total_vcpus: int
    total_ram: int  # En MB
    total_disk: float  # En GB
    used_vcpus: int = 0
    used_ram: int = 0
    used_disk: float = 0.0
    
    @property
    def available_vcpus(self) -> int:
        """Retorna la cantidad de vCPUs disponibles"""
        return self.total_vcpus - self.used_vcpus
    
    @property
    def available_ram(self) -> int:
        """Retorna la cantidad de RAM disponible en MB"""
        return self.total_ram - self.used_ram
    
    @property
    def available_disk(self) -> float:
        """Retorna la cantidad de espacio en disco disponible en GB"""
        return self.total_disk - self.used_disk
    
    def __str__(self):
        # Calcular porcentajes de uso
        vcpu_pct = (self.used_vcpus / self.total_vcpus * 100) if self.total_vcpus > 0 else 0
        ram_pct = (self.used_ram / self.total_ram * 100) if self.total_ram > 0 else 0
        disk_pct = (self.used_disk / self.total_disk * 100) if self.total_disk > 0 else 0
        
        return (f"Servidor {self.id}: {self.name} - "
                f"CPU: {self.used_vcpus}/{self.total_vcpus} ({vcpu_pct:.1f}%), "
                f"RAM: {self.used_ram}/{self.total_ram} MB ({ram_pct:.1f}%), "
                f"Disco: {self.used_disk:.1f}/{self.total_disk:.1f} GB ({disk_pct:.1f}%)")

@dataclass
class PlacementResult:
    """
    Almacena el resultado de la colocación de VMs
    """
    success: bool
    assignments: Dict[int, int] = field(default_factory=dict)  # VM_id -> Server_id
    message: str = ""
    objective_value: float = 0.0
    
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
                        total_available_vcpus += server.available_vcpus
                        total_available_ram += server.available_ram
                        total_available_disk += server.available_disk
                        
                        # Calcular porcentajes
                        vcpu_percent = (vcpus_used / server.available_vcpus * 100) if server.available_vcpus > 0 else 0
                        ram_percent = (ram_used / server.available_ram * 100) if server.available_ram > 0 else 0
                        disk_percent = (disk_used / server.available_disk * 100) if server.available_disk > 0 else 0
                        
                        servers_usage.append({
                            "id": server.id,
                            "name": server.name,
                            "vms_count": len(vm_ids),
                            "resources": {
                                "vcpus": {
                                    "used": vcpus_used,
                                    "total": server.available_vcpus,
                                    "percent": round(vcpu_percent, 2)
                                },
                                "ram": {
                                    "used": ram_used,
                                    "total": server.available_ram,
                                    "percent": round(ram_percent, 2)
                                },
                                "disk": {
                                    "used": round(disk_used, 1),
                                    "total": round(server.available_disk, 1),
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
class UtilityCalculator:
    """
    Calcula la utilidad de asignar una VM a un servidor físico
    
    Esta clase proporciona métodos para calcular la utilidad o valor
    de colocar una VM en un servidor específico.
    """
    
    @staticmethod
    def calculate_basic(vm: VirtualMachine) -> float:
        """
        Cálculo básico de utilidad basado en el flavor de la VM
        
        Esta función implementa un cálculo básico de utilidad basado en los recursos
        del flavor.
        """
        # Fórmula básica: utilidad proporcional a los recursos/flavors
        # Podemos ajustar los pesos según la importancia relativa de cada recurso
        cpu_weight = 0.7
        ram_weight = 0.8
        disk_weight = 0.7
        
        utility = (
            cpu_weight * vm.flavor.vcpus + 
            ram_weight * (vm.flavor.ram / 1024) +
            disk_weight * vm.flavor.disk
        )
        
        return utility
    
    @staticmethod
    def calculate_advanced(vm: VirtualMachine, **kwargs) -> float:
        """
        Cálculo avanzado de utilidad que puede considerar factores adicionales
        
        Args:
            vm: La máquina virtual
            **kwargs: Factores adicionales como rol de usuario, prioridad, etc.
        """
        # Primero calculamos la utilidad básica
        utility = UtilityCalculator.calculate_basic(vm)
        
        # Factores adicionales que pueden implementarse en el futuro
        # Por ejemplo: rol de usuario
        if 'user_role' in kwargs:
            role_factor = {
                'investigador': 2.0,
                'jp': 1.5,
                'alumno': 1.0
            }.get(kwargs['user_role'], 1.0)
            utility *= role_factor
        
        # Factor de prioridad
        if 'priority' in kwargs:
            priority = kwargs['priority']
            utility *= (1 + priority / 10)  # Prioridad en escala 0-10
            
        return utility

class VMPlacementSolver:
    """
    Resuelve el problema de la colocación de VMs utilizando MILP de SciPy
    """
    
    def __init__(self, vms: List[VirtualMachine], servers: List[PhysicalServer]):
        """
        Inicializa el solucionador con las VMs y servidores disponibles
        
        Args:
            vms: Lista de máquinas virtuales a colocar
            servers: Lista de servidores físicos disponibles
        """
        self.vms = vms
        self.servers = servers
        
        # Calcular utilidad para cada VM si no está definida
        for vm in self.vms:
            if vm.utility <= 0:
                vm.utility = UtilityCalculator.calculate_basic(vm)
    
    def solve(self) -> PlacementResult:
        """
        Resuelve el problema de placement usando MILP con SciPy
        
        Returns:
            PlacementResult con los resultados de la colocación
        """
        Logger.section("Iniciando resolución del problema de VM Placement con MILP (SciPy)")
        
        if not self.vms or not self.servers:
            Logger.error("No hay VMs o servidores disponibles para resolver el problema")
            return PlacementResult(success=False, message="No hay VMs o servidores disponibles")
        
        Logger.info(f"Resolviendo para {len(self.vms)} VMs y {len(self.servers)} servidores")
        
        # Verificar capacidad total de los servidores antes de resolver
        total_vcpus_required = sum(vm.flavor.vcpus for vm in self.vms)
        total_ram_required = sum(vm.flavor.ram for vm in self.vms)
        total_disk_required = sum(vm.flavor.disk for vm in self.vms)
        
        total_vcpus_available = sum(server.available_vcpus for server in self.servers)
        total_ram_available = sum(server.available_ram for server in self.servers)
        total_disk_available = sum(server.available_disk for server in self.servers)
        
        Logger.info(f"Recursos requeridos - vCPUs: {total_vcpus_required}, RAM: {total_ram_required} MB, Disco: {total_disk_required:.1f} GB")
        Logger.info(f"Recursos disponibles - vCPUs: {total_vcpus_available}, RAM: {total_ram_available} MB, Disco: {total_disk_available:.1f} GB")
        
        # Verificar si hay suficientes recursos totales
        insufficient_resources = []
        if total_vcpus_required > total_vcpus_available:
            insufficient_resources.append(f"vCPUs (Requerido: {total_vcpus_required}, Disponible: {total_vcpus_available})")
        if total_ram_required > total_ram_available:
            insufficient_resources.append(f"RAM (Requerido: {total_ram_required} MB, Disponible: {total_ram_available} MB)")
        if total_disk_required > total_disk_available:
            insufficient_resources.append(f"Disco (Requerido: {total_disk_required:.1f} GB, Disponible: {total_disk_available:.1f} GB)")
        
        if insufficient_resources:
            error_msg = f"No hay suficientes recursos en total para todas las VMs: {', '.join(insufficient_resources)}"
            Logger.error(error_msg)
            return PlacementResult(success=False, message=error_msg)
        
        try:
            # Definir dimensiones del problema
            n_vms = len(self.vms)
            n_servers = len(self.servers)
            n_vars = n_vms * n_servers  # Total de variables de decisión
            
            # Vector de coeficientes de la función objetivo (utilidades, negativas porque milp minimiza)
            # Tenemos que hacer -p porque queremos maximizar la utilidad, pero milp minimiza
            c = np.zeros(n_vars)
            for i in range(n_vms):
                for j in range(n_servers):
                    idx = i * n_servers + j
                    c[idx] = -self.vms[i].utility
            
            # Todas las variables son binarias
            integrality = np.ones(n_vars, dtype=bool)  # True para todas las variables (binarias)
            
            # Límites de las variables: entre 0 y 1 (variables binarias)
            bounds = Bounds(0, 1)
            
            # Lista para almacenar todas las restricciones
            all_constraints = []
            
            # Restricción 1: Cada VM debe estar asignada a máximo un servidor
            for i in range(n_vms):
                # Para cada VM, creamos una matriz de coeficientes donde solo las variables
                # correspondientes a esa VM tienen coeficiente 1
                A_vm = np.zeros((1, n_vars))
                for j in range(n_servers):
                    idx = i * n_servers + j
                    A_vm[0, idx] = 1
                
                # La suma debe ser <= 1 (una VM puede estar en a lo sumo un servidor)
                constraint_vm = LinearConstraint(A_vm, -np.inf, 1)
                all_constraints.append(constraint_vm)
            
            # Restricciones de capacidad para cada tipo de recurso en cada servidor
            
            # Restricción 2: No exceder capacidad de vCPUs en cada servidor
            for j in range(n_servers):
                A_cpu = np.zeros((1, n_vars))
                for i in range(n_vms):
                    idx = i * n_servers + j
                    A_cpu[0, idx] = self.vms[i].flavor.vcpus
                
                # La suma ponderada de vCPUs no debe exceder la capacidad del servidor
                constraint_cpu = LinearConstraint(A_cpu, -np.inf, self.servers[j].available_vcpus)
                all_constraints.append(constraint_cpu)
            
            # Restricción 3: No exceder capacidad de RAM en cada servidor
            for j in range(n_servers):
                A_ram = np.zeros((1, n_vars))
                for i in range(n_vms):
                    idx = i * n_servers + j
                    A_ram[0, idx] = self.vms[i].flavor.ram
                
                # La suma ponderada de RAM no debe exceder la capacidad del servidor
                constraint_ram = LinearConstraint(A_ram, -np.inf, self.servers[j].available_ram)
                all_constraints.append(constraint_ram)
            
            # Restricción 4: No exceder capacidad de disco en cada servidor
            for j in range(n_servers):
                A_disk = np.zeros((1, n_vars))
                for i in range(n_vms):
                    idx = i * n_servers + j
                    A_disk[0, idx] = self.vms[i].flavor.disk
                
                # La suma ponderada de disco no debe exceder la capacidad del servidor
                constraint_disk = LinearConstraint(A_disk, -np.inf, self.servers[j].available_disk)
                all_constraints.append(constraint_disk)
            
            Logger.section("Resolviendo el modelo MILP con SciPy")
            
            # Resolver el problema usando milp de scipy.optimize
            result = milp(
                c=c,
                constraints=all_constraints,
                integrality=integrality,
                bounds=bounds,
                options={'disp': True}
            )
            
            Logger.info(f"Estado de la solución: {result.status}, {result.message}")
            
            if result.success:
                # Redondear para asegurar valores binarios exactos (por si acaso)
                x_solution = np.round(result.x).astype(int)
                
                # Extraer las asignaciones y registrar VMs no asignadas
                assignments = {}
                unassigned_vms = []
                
                for i in range(n_vms):
                    assigned = False
                    for j in range(n_servers):
                        idx = i * n_servers + j
                        if x_solution[idx] == 1:
                            assignments[self.vms[i].id] = self.servers[j].id
                            assigned = True
                            Logger.success(f"VM {self.vms[i].name} (ID: {self.vms[i].id}) asignada al servidor {self.servers[j].name} (ID: {self.servers[j].id})")
                            break
                    
                    if not assigned:
                        unassigned_vms.append(self.vms[i])
                        Logger.warning(f"VM {self.vms[i].name} (ID: {self.vms[i].id}) no fue asignada a ningún servidor")
                
                # El valor objetivo es negativo de la función objetivo (porque minimizamos -utilidad)
                objective_value = -result.fun
                Logger.success(f"Solución encontrada con valor objetivo: {objective_value:.2f}")
                
                # Estadísticas de la solución
                vm_count = len(assignments)
                server_count = len(set(assignments.values()))
                Logger.info(f"Se asignaron {vm_count} de {n_vms} VMs a {server_count} servidores")
                
                # Analizar por qué las VMs no pudieron ser asignadas
                if unassigned_vms:
                    Logger.section("Análisis de VMs no asignadas")
                    
                    for vm in unassigned_vms:
                        Logger.warning(f"Analizando por qué VM {vm.name} (ID: {vm.id}) no pudo ser asignada:")
                        
                        # Verificar si puede caber en algún servidor con respecto a cada recurso
                        can_fit_by_vcpu = False
                        can_fit_by_ram = False
                        can_fit_by_disk = False
                        
                        for server in self.servers:
                            if vm.flavor.vcpus <= server.available_vcpus:
                                can_fit_by_vcpu = True
                            if vm.flavor.ram <= server.available_ram:
                                can_fit_by_ram = True
                            if vm.flavor.disk <= server.available_disk:
                                can_fit_by_disk = True
                        
                        if not can_fit_by_vcpu:
                            Logger.error(f"  - VM requiere {vm.flavor.vcpus} vCPUs pero ningún servidor tiene suficiente capacidad disponible")
                        if not can_fit_by_ram:
                            Logger.error(f"  - VM requiere {vm.flavor.ram} MB de RAM pero ningún servidor tiene suficiente capacidad disponible")
                        if not can_fit_by_disk:
                            Logger.error(f"  - VM requiere {vm.flavor.disk:.1f} GB de disco pero ningún servidor tiene suficiente capacidad disponible")
                        
                        # Si la VM puede caber en servidores individuales pero no fue asignada
                        if can_fit_by_vcpu and can_fit_by_ram and can_fit_by_disk:
                            Logger.error("  - La VM podría caber individualmente, pero la solución óptima encontrada por el algoritmo no la incluye " +
                                        "debido a restricciones globales o para maximizar la utilidad total")
                    
                    # Generar mensaje de error apropiado
                    message = f"No se pudieron asignar todas las VMs. {len(unassigned_vms)} de {n_vms} VMs no fueron asignadas debido a restricciones de capacidad o para mantener la solución óptima."
                    
                    # Decidir si la solución es parcialmente exitosa o fallida
                    if vm_count > 0:
                        Logger.warning(message)
                        return PlacementResult(
                            success=True,  # Consideramos éxito parcial
                            assignments=assignments,
                            message=message,
                            objective_value=objective_value
                        )
                    else:
                        Logger.failed("No se pudo asignar ninguna VM")
                        return PlacementResult(success=False, message="No se pudo asignar ninguna VM")
                
                # Si todas las VMs fueron asignadas
                return PlacementResult(
                    success=True,
                    assignments=assignments,
                    message=f"Solución óptima encontrada. Todas las VMs ({vm_count}) asignadas a {server_count} servidores.",
                    objective_value=objective_value
                )
            else:
                Logger.failed(f"No se encontró una solución: {result.message}")
                return PlacementResult(success=False, message=f"No se pudo encontrar solución: {result.message}")
            
        except Exception as e:
            Logger.error(f"Error al resolver el problema: {str(e)}")
            return PlacementResult(success=False, message=f"Error: {str(e)}")

    def visualize_placement(self, result: PlacementResult):
        """
        Visualiza gráficamente los resultados del placement y guarda en carpeta 'resultados'
        
        Args:
            result: Resultado del placement
        """
        if not result.success or not result.assignments:
            Logger.error("No hay una solución válida para visualizar")
            return
        
        Logger.section("Visualización del Placement")
        
        # Crear un mapa de servidores a VMs asignadas
        server_to_vms = {}
        for vm_id, server_id in result.assignments.items():
            if server_id not in server_to_vms:
                server_to_vms[server_id] = []
            server_to_vms[server_id].append(vm_id)
        
        # Calcular uso de recursos por servidor
        server_usage = {}
        # Información detallada de las VMs para gráfica adicional
        vm_details = []
        
        for server_id, vm_ids in server_to_vms.items():
            # Buscar el objeto servidor
            server = next((s for s in self.servers if s.id == server_id), None)
            if not server:
                continue
                
            vcpus_used = 0
            ram_used = 0
            disk_used = 0.0
            
            for vm_id in vm_ids:
                # Buscar el objeto VM
                vm = next((v for v in self.vms if v.id == vm_id), None)
                if not vm:
                    continue
                    
                vcpus_used += vm.flavor.vcpus
                ram_used += vm.flavor.ram
                disk_used += vm.flavor.disk
                
                # Guardar información detallada de la VM para la gráfica adicional
                vm_details.append({
                    "vm_id": vm_id,
                    "vm_name": vm.name,
                    "server_id": server_id,
                    "server_name": server.name,
                    "vcpus": vm.flavor.vcpus,
                    "ram": vm.flavor.ram,
                    "disk": vm.flavor.disk
                })
            
            server_usage[server_id] = {
                "server_name": server.name,
                "vcpus": {
                    "used": vcpus_used,
                    "total": server.available_vcpus,
                    "percent": (vcpus_used / server.available_vcpus * 100) if server.available_vcpus > 0 else 0
                },
                "ram": {
                    "used": ram_used,
                    "total": server.available_ram,
                    "percent": (ram_used / server.available_ram * 100) if server.available_ram > 0 else 0
                },
                "disk": {
                    "used": disk_used,
                    "total": server.available_disk,
                    "percent": (disk_used / server.available_disk * 100) if server.available_disk > 0 else 0
                },
                "vms": len(vm_ids)
            }
        
        # Crear gráficos para visualizar la asignación y guardarlos como imagen
        try:
            
            import matplotlib
            matplotlib.use('Agg') 

            # Crear carpeta 'resultados' si no existe
            results_dir = 'resultados'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                Logger.info(f"Creada carpeta '{results_dir}' para guardar visualizaciones")
            
            # Generar nombre de archivo único con timestamp
            timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
            timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = os.path.join(results_dir, f'resultado_{timestamp}.png')
            
            # Crear figura más grande para incluir 4 gráficos (3 de uso de recursos + 1 tabla de VMs)
            fig = plt.figure(figsize=(16, 12))
            fig.text(0.5, 0.02, f"Generado: {timestamp_text}", 
                           ha='center', fontsize=8, style='italic', color='gray')

            # Configurar subplots: 2 filas, 2 columnas
            ax1 = plt.subplot2grid((2, 2), (0, 0))  # vCPUs
            ax2 = plt.subplot2grid((2, 2), (0, 1))  # RAM
            ax3 = plt.subplot2grid((2, 2), (1, 0))  # Disco
            ax4 = plt.subplot2grid((2, 2), (1, 1))  # Tabla de VMs
            
            # Ordenar servidores por ID
            server_ids = sorted(server_usage.keys())
            server_names = [server_usage[sid]["server_name"] for sid in server_ids]
            
            # Graficar uso de vCPUs
            cpu_used = [server_usage[sid]["vcpus"]["used"] for sid in server_ids]
            cpu_total = [server_usage[sid]["vcpus"]["total"] for sid in server_ids]
            
            ax1.bar(server_names, cpu_total, color='lightblue', alpha=0.6, label='Disponible')
            ax1.bar(server_names, cpu_used, color='blue', label='Usado')
            ax1.set_title('Uso de vCPUs por Servidor')
            ax1.set_ylabel('vCPUs')
            ax1.set_xlabel('Servidores')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Graficar uso de RAM
            ram_used = [server_usage[sid]["ram"]["used"] / 1024 for sid in server_ids]  # Convertir a GB
            ram_total = [server_usage[sid]["ram"]["total"] / 1024 for sid in server_ids]  # Convertir a GB
            
            ax2.bar(server_names, ram_total, color='lightgreen', alpha=0.6, label='Disponible')
            ax2.bar(server_names, ram_used, color='green', label='Usado')
            ax2.set_title('Uso de RAM por Servidor')
            ax2.set_ylabel('RAM (GB)')
            ax2.set_xlabel('Servidores')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            # Graficar uso de Disco
            disk_used = [server_usage[sid]["disk"]["used"] for sid in server_ids]
            disk_total = [server_usage[sid]["disk"]["total"] for sid in server_ids]
            
            ax3.bar(server_names, disk_total, color='lightcoral', alpha=0.6, label='Disponible')
            ax3.bar(server_names, disk_used, color='red', label='Usado')
            ax3.set_title('Uso de Disco por Servidor')
            ax3.set_ylabel('Disco (GB)')
            ax3.set_xlabel('Servidores')
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # Crear visualización de asignaciones de VMs por servidor
            ax4.set_title('Asignación de VMs a Servidores', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Servidores', fontsize=12)
            ax4.set_ylabel('Recursos Asignados (%)', fontsize=12)
            ax4.grid(True, linestyle='--', alpha=0.7)

            # Organizar los datos por servidor
            servers_data = {}
            for server_id in server_ids:
                server_name = server_usage[server_id]["server_name"]
                servers_data[server_name] = []

            for vm in vm_details:
                servers_data[vm["server_name"]].append(vm)

            # Ordenar servidores para la visualización
            server_names = list(servers_data.keys())
            server_names.sort()

            # Definimos colores específicos para cada tipo de recurso
            color_vcpus = '#4287f5'  
            color_ram = '#42f584'  
            color_disk = '#f54242'   

            # Ancho de las barras y posiciones
            bar_width = 0.25
            positions = np.arange(len(server_names))

            # Crear tres barras por servidor (una para cada tipo de recurso: vCPUs, RAM, Disk)
            resource_types = ['vCPUs', 'RAM', 'Disco']
            resource_colors = [color_vcpus, color_ram, color_disk]
            resource_positions = [-1, 0, 1]  # Posiciones relativas para las tres barras

            # Crear leyenda más visible y mejorada para tipos de recursos
            legend_elements = [
                plt.Rectangle((0,0), 1, 1, color=color_vcpus, label='vCPUs'),
                plt.Rectangle((0,0), 1, 1, color=color_ram, label='RAM'),
                plt.Rectangle((0,0), 1, 1, color=color_disk, label='Disk')
            ]
            ax4.legend(handles=legend_elements, loc='upper right', title='Tipos de Recursos',
                       fontsize=10, title_fontsize=11, framealpha=0.9, facecolor='white',
                       edgecolor='gray', fancybox=True, shadow=True)

            # Para cada tipo de recurso, crear barras para cada servidor
            for i, (resource_type, color, position) in enumerate(zip(resource_types, resource_colors, resource_positions)):
                bottom_values = np.zeros(len(server_names))  # Reiniciar para cada tipo de recurso
                
                # Para cada servidor, crear barras apiladas de VMs
                for server_idx, server_name in enumerate(server_names):
                    server_vms = servers_data[server_name]
                    
                    # Ordenar VMs por tamaño de recursos para mejor visualización
                    if resource_type == 'vCPUs':
                        server_vms.sort(key=lambda x: x["vcpus"])
                        resource_totals = [server_usage[sid]["vcpus"]["total"] for sid in server_ids]
                        vm_values = [vm["vcpus"] for vm in server_vms]
                    elif resource_type == 'RAM':
                        server_vms.sort(key=lambda x: x["ram"])
                        resource_totals = [server_usage[sid]["ram"]["total"] for sid in server_ids]
                        vm_values = [vm["ram"] for vm in server_vms]
                    else:
                        server_vms.sort(key=lambda x: x["disk"])
                        resource_totals = [server_usage[sid]["disk"]["total"] for sid in server_ids]
                        vm_values = [vm["disk"] for vm in server_vms]
                    
                    # Calcular porcentajes para cada VM en este servidor y tipo de recurso
                    total_resource = resource_totals[server_idx]
                    vm_percentages = [value / total_resource * 100 for value in vm_values]
                    
                    # Dibujar cada VM como un segmento en la barra apilada
                    for j, (vm, percentage) in enumerate(zip(server_vms, vm_percentages)):
                        x_pos = positions[server_idx] + position * bar_width
                        
                        # Definir un color ligeramente diferente para cada VM para distinguirlas
                        # pero manteniendo la coherencia del color base según el tipo de recurso
                        vm_color = color
                        if j % 2:  # Para VMs alternas, usar un tono ligeramente más oscuro
                            # Oscurecer el color un poco
                            r, g, b = matplotlib.colors.to_rgb(color)
                            vm_color = (r*0.8, g*0.8, b*0.8)  # 20% más oscuro
                        
                        # Dibujar el segmento de la VM
                        bar = ax4.bar(x_pos, percentage, bar_width, bottom=bottom_values[server_idx], 
                                     color=vm_color, edgecolor='white', linewidth=0.7)
                        
                        # Añadir etiqueta del nombre de la VM si hay espacio suficiente
                        if percentage > 5:  # Solo etiquetar segmentos que ocupen más del 5%
                            y_pos = bottom_values[server_idx] + percentage / 2
                            ax4.text(x_pos, y_pos, vm["vm_name"], ha='center', va='center', 
                                    color='white', fontsize=9, fontweight='bold')
                        
                        # Actualizar el valor acumulado para la siguiente VM
                        bottom_values[server_idx] += percentage

            # Añadir línea de referencia al 100%
            ax4.axhline(y=100, color='black', linestyle='--', alpha=0.5)

            # Configurar ejes
            ax4.set_xticks(positions)
            ax4.set_xticklabels(server_names, fontsize=11, fontweight='bold')
            ax4.set_ylim(0, 110)  # Dar espacio para etiquetas
            ax4.set_yticks([0, 25, 50, 75, 100])
            ax4.grid(axis='y', linestyle='--', alpha=0.3)

            # Crear una tabla detallada de las VMs y guardarla como imagen separada
            table_filename = os.path.join(results_dir, f'resultado_tabla_{timestamp}.png')

            # Preparar datos para la tabla con formato mejorado
            table_data = []
            for vm in sorted(vm_details, key=lambda x: x["vm_id"]):
                table_data.append([
                    vm["vm_name"],
                    f"{vm['vcpus']}",
                    f"{vm['ram']/1000:.3f} GB",  # 3 decimales para RAM
                    f"{vm['disk']:.2f} GB",      # 2 decimales para Disco
                    vm["server_name"]
                ])

            # Crear figura para la tabla más atractiva
            fig_table = plt.figure(figsize=(10, len(table_data) * 0.5 + 1.5), facecolor='#f9f9f9')
            ax_table = fig_table.add_subplot(111)

            # Ocultar ejes
            ax_table.axis('off')
            ax_table.axis('tight')

            # Definir colores para la tabla
            header_color = '#4472C4'  # Azul para encabezados
            row_colors = ['#E6F0FF', '#FFFFFF']  # Alternar colores de fila
            text_color = 'black'
            border_color = '#BFBFBF'

            # Crear la tabla con estilos mejorados
            table = ax_table.table(
                cellText=table_data,
                colLabels=['Máquina Virtual', 'vCPUs', 'RAM', 'Disco', 'Servidor'],
                loc='center',
                cellLoc='center'
            )

            # Mejorar estilo de la tabla
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)

            # Aplicar colores y estilos a las celdas
            for k, cell in table.get_celld().items():
                cell.set_edgecolor(border_color)
                
                # Encabezados de columnas (fila 0)
                if k[0] == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor(header_color)
                else:
                    # Alternar colores para filas de datos
                    cell.set_facecolor(row_colors[(k[0]) % 2])
                    
                    # Alinear columnas numéricas a la derecha
                    if k[1] in [1, 2, 3]:  # vCPUs, RAM, Disco
                        cell.get_text().set_horizontalalignment('right')

            # Agregar título con estilo
            ax_table.set_title("Detalle de VMs Asignadas", 
                               fontsize=14, 
                               fontweight='bold',
                               color='#333333',
                               pad=20)

            fig_table.text(0.5, 0.02, f"Generado: {timestamp_text}", 
                           ha='center', fontsize=8, style='italic', color='gray')

            # Guardar tabla con mayor calidad
            plt.tight_layout()
            plt.savefig(table_filename, dpi=300, bbox_inches='tight', facecolor=fig_table.get_facecolor())
            plt.close(fig_table)
            
            # Ajustar diseño y guardar la figura principal
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            Logger.success(f"Visualizaciones guardadas en carpeta '{results_dir}':")
            Logger.success(f"- Gráficos: {os.path.basename(filename)}")
            Logger.success(f"- Tabla detallada: {os.path.basename(table_filename)}")
            
        except Exception as e:
            Logger.error(f"Error al generar visualización: {str(e)}")
            Logger.debug(f"Traceback: {traceback.format_exc()}")

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
            
            Logger.success(f"Datos cargados correctamente: {len(vms)} VMs y {len(servers)} servidores")
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
    Endpoint para resolver el problema de placement de VMs
    
    Cuerpo del request:
    {
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
            200: Placement exitoso
            400: Datos de entrada inválidos
            500: Error interno
    """
    try:
        Logger.major_section("API: VM PLACEMENT")
        
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
        
        # Crear un JSON estructurado con los flavors de la BD y las VMs del request
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
            
            # Mostrar resumen de datos
            Logger.section("Resumen de datos para placement")
            Logger.info(f"Total de VMs: {len(vms)}")
            Logger.info(f"Total de servidores: {len(servers)}")
            
            # Resolver el problema
            solver = VMPlacementSolver(vms, servers)
            result = solver.solve()
            
            # Generar respuesta según el resultado
            if result.success:
                # Convertir resultado a formato JSON para respuesta
                result_dict = result.to_dict(vms=vms, servers=servers)
                
                # Intentar generar visualización y guardarla
                try:
                    solver.visualize_placement(result)
                    Logger.success("Visualización generada correctamente")
                    Logger.success(f"Placement realizado exitosamente: {len(result.assignments)} VMs asignadas")
                except Exception as viz_error:
                    Logger.warning(f"No se pudo generar la visualización: {str(viz_error)}")
                
                return jsonify({
                    "status": "success",
                    "message": f"Se realizó el placement de {len(result.assignments)} VMs",
                    "content": result_dict
                }), 200
            else:
                Logger.error(f"No se pudo resolver el problema de VM Placement")
                return jsonify({
                    "status": "error",
                    "message": "Error al resolver el placement",
                    "details": result.message
                }), 400
            
        except Exception as e:
            Logger.error(f"Error procesando los datos: {str(e)}")
            return jsonify({
                "status": "error",
                "message": "Error al procesar los datos",
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
    Endpoint para obtener datos de prueba que pueden usarse con el endpoint de placement
    """
    try:
        # Generar datos de prueba con flavors de la BD
        num_vms = np.random.randint(1,8)
        test_data_json = DataManager.generate_test_data(num_vms=num_vms, num_servers=0, seed=None)
        test_data = json.loads(test_data_json)
        
        response_data = {
            "virtual_machines": test_data["virtual_machines"]
        }
        
        Logger.success(f"Datos de prueba para VM Placement generados con éxito!")

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