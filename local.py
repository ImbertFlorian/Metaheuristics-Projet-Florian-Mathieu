import os
import glob
import time
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional

# ============================================================
# DATA CLASSES & PARSER
# ============================================================

@dataclass
class CALBPInstance:
    tasks: List[int]                        
    stations: List[int]                     
    modes: List[int]                        
    t: Dict[Tuple[int, int], float]         
    E: Dict[Tuple[int, int], float]         
    precedences: List[Tuple[int, int]]      
    T: float                                
    Cs: float                               
    Cw: float                               
    Cc: float                               
    Re: float                               

    @property
    def MH(self) -> Set[int]:
        return {1, 3}   # HI, SU

    @property
    def MC(self) -> Set[int]:
        return {2, 3}   # CI, SU

def read_instance(filepath: str) -> CALBPInstance:
    tasks = []
    stations = []
    modes = []
    t = {}
    E = {}
    precedences = []
    T = Cs = Cw = Cc = Re = 0.0
    current_section = None

    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line: continue
            if line.startswith("#"):
                line_lower = line.lower()
                if "tasks" in line_lower: current_section = "tasks"
                elif "workstations" in line_lower: current_section = "workstations"
                elif "modes" in line_lower: current_section = "modes"
                elif "processing" in line_lower: current_section = "processing"
                elif "energy" in line_lower: current_section = "energy"
                elif "precedence" in line_lower: current_section = "precedence"
                elif "parameters" in line_lower: current_section = "parameters"
                continue

            if current_section == "tasks": tasks.extend([int(x) for x in line.split()])
            elif current_section == "workstations": stations.extend([int(x) for x in line.split()])
            elif current_section == "modes": modes.extend([int(x) for x in line.split()])
            elif current_section == "processing":
                parts = line.split()
                if len(parts) == 3: t[(int(parts[0]), int(parts[1]))] = float(parts[2])
            elif current_section == "energy":
                parts = line.split()
                if len(parts) == 3: E[(int(parts[0]), int(parts[1]))] = float(parts[2])
            elif current_section == "precedence":
                parts = line.split(',')
                if len(parts) == 2: precedences.append((int(parts[0]), int(parts[1])))
            elif current_section == "parameters":
                parts = line.split('=')
                if len(parts) == 2:
                    key = parts[0].strip().upper()
                    val = float(parts[1].strip())
                    if key == "R_E": Re = val
                    elif key == "T": T = val
                    elif key == "C_S": Cs = val
                    elif key == "C_W": Cw = val
                    elif key == "C_C": Cc = val

    return CALBPInstance(
        tasks=tasks, stations=stations, modes=modes,
        t=t, E=E, precedences=precedences,
        T=T, Cs=Cs, Cw=Cw, Cc=Cc, Re=Re
    )

# ============================================================
# SOLUTION CLASS
# ============================================================

class Solution:
    def __init__(self, inst: CALBPInstance, task_seq: list, task_modes: dict):
        self.inst = inst
        self.task_seq = list(task_seq)
        self.task_modes = task_modes.copy()
        self.assignments = {} 
        self.cost = float('inf')
        self.energy = float('inf')
        self.station_tasks = {}
        self.worker_stations = set()
        self.cobot_stations = set()
        
    def decode_and_evaluate(self):
        current_station = 1
        current_time = 0.0
        self.assignments = {}
        self.station_tasks = {}
        
        for j in self.task_seq:
            mode = self.task_modes[j]
            t_jm = self.inst.t[(j, mode)]
            if current_time + t_jm > self.inst.T:
                current_station += 1
                current_time = 0.0
                
            self.assignments[j] = (current_station, mode)
            if current_station not in self.station_tasks:
                self.station_tasks[current_station] = []
            self.station_tasks[current_station].append((j, mode))
            current_time += t_jm

        task_energy_total = 0.0
        idle_energy_total = 0.0
        self.worker_stations = set()
        self.cobot_stations = set()

        for k, tasks_in_k in self.station_tasks.items():
            needs_worker = False
            needs_cobot = False
            cobot_active_time = 0.0
            for j, mode in tasks_in_k:
                task_energy_total += self.inst.E[(j, mode)]
                if mode in self.inst.MH: needs_worker = True
                if mode in self.inst.MC: 
                    needs_cobot = True
                    cobot_active_time += self.inst.t[(j, mode)]

            if needs_worker: self.worker_stations.add(k)
            if needs_cobot:
                self.cobot_stations.add(k)
                idle_time = self.inst.T - cobot_active_time
                idle_energy_total += self.inst.Re * idle_time

        num_stations = len(self.station_tasks)
        num_workers = len(self.worker_stations)
        num_cobots = len(self.cobot_stations)
        
        self.cost = (num_stations * self.inst.Cs) + (num_workers * self.inst.Cw) + (num_cobots * self.inst.Cc)
        self.energy = task_energy_total + idle_energy_total

# ============================================================
# HEURISTIQUES
# ============================================================

def constructive_heuristic_cost(inst: CALBPInstance) -> Solution:
    task_seq = sorted(inst.tasks)
    task_modes = {}
    for j in task_seq:
        mode = 2
        if inst.t[(j, mode)] > inst.T:
            mode = min(inst.modes, key=lambda m: inst.t[(j, m)])
        task_modes[j] = mode
    sol = Solution(inst, task_seq, task_modes)
    sol.decode_and_evaluate()
    return sol

def constructive_heuristic_energy(inst: CALBPInstance) -> Solution:
    task_seq = sorted(inst.tasks)
    task_modes = {}
    for j in task_seq:
        mode = 1
        if inst.t[(j, mode)] > inst.T:
            mode = min(inst.modes, key=lambda m: inst.t[(j, m)])
        task_modes[j] = mode
    sol = Solution(inst, task_seq, task_modes)
    sol.decode_and_evaluate()
    return sol

# ============================================================
# OUTILS PARETO & VOISINAGE (OPTIMISÉS)
# ============================================================

def dominates_sol(s1: Solution, s2: Solution, tol=1e-6) -> bool:
    return (s1.cost <= s2.cost + tol and s1.energy <= s2.energy + tol and 
            (s1.cost < s2.cost - tol or s1.energy < s2.energy - tol))

def filter_nondominated_sols(population: list) -> list:
    nd = []
    for s in population:
        if not any(dominates_sol(t, s) for t in population if s is not t):
            nd.append(s)
    unique = []
    seen = set()
    for s in nd:
        key = (round(s.cost, 6), round(s.energy, 6))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    unique.sort(key=lambda z: (z.energy, z.cost))
    return unique

def build_predecessors_dict(inst: CALBPInstance) -> dict:
    preds = {j: set() for j in inst.tasks}
    for (p, s) in inst.precedences:
        preds[s].add(p)
    changed = True
    while changed:
        changed = False
        for j in inst.tasks:
            for p in list(preds[j]):
                for pp in preds[p]:
                    if pp not in preds[j]:
                        preds[j].add(pp)
                        changed = True
    return preds

def generate_random_valid_solution(inst: CALBPInstance, all_preds: dict) -> Solution:
    seq = list(sorted(inst.tasks))
    for _ in range(20): 
        i = random.randint(0, len(seq) - 2)
        if seq[i] not in all_preds[seq[i+1]]:
            seq[i], seq[i+1] = seq[i+1], seq[i]
    modes = {j: random.choice(inst.modes) for j in seq}
    sol = Solution(inst, seq, modes)
    sol.decode_and_evaluate()
    return sol

def generate_neighbors(sol: Solution, all_preds: dict) -> list:
    neighbors = []
    inst = sol.inst
    for j in inst.tasks:
        current_mode = sol.task_modes[j]
        for m in inst.modes:
            if m != current_mode:
                new_modes = sol.task_modes.copy()
                new_modes[j] = m
                new_sol = Solution(inst, sol.task_seq, new_modes)
                new_sol.decode_and_evaluate()
                neighbors.append(new_sol)
    for i in range(len(sol.task_seq) - 1):
        task_1 = sol.task_seq[i]
        task_2 = sol.task_seq[i+1]
        if task_1 not in all_preds[task_2]:
            new_seq = list(sol.task_seq)
            new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i]
            new_sol = Solution(inst, new_seq, sol.task_modes)
            new_sol.decode_and_evaluate()
            neighbors.append(new_sol)
    return neighbors

# ============================================================
# ALGORITHME PLS 
# ============================================================

def pareto_local_search(inst: CALBPInstance, time_limit: float = 30.0) -> list:
    print(f"Lancement de la Recherche Locale (Temps limite: {time_limit}s)...")
    all_preds = build_predecessors_dict(inst)
    
    initial_sols = [constructive_heuristic_cost(inst), constructive_heuristic_energy(inst)]
    for _ in range(10): # Paramètre optimisé
        initial_sols.append(generate_random_valid_solution(inst, all_preds))
        
    archive = filter_nondominated_sols(initial_sols)
    explored_signatures = set()
    start_time = time.time()
    iterations = 0
    
    while time.time() - start_time < time_limit:
        current_sol = None
        for sol in archive:
            sig = (tuple(sol.task_seq), tuple(sorted(sol.task_modes.items())))
            if sig not in explored_signatures:
                current_sol = sol
                explored_signatures.add(sig)
                break
                
        if current_sol is None:
            current_sol = generate_random_valid_solution(inst, all_preds)
            explored_signatures.add((tuple(current_sol.task_seq), tuple(sorted(current_sol.task_modes.items()))))
            
        neighbors = generate_neighbors(current_sol, all_preds)
        new_valid_neighbors = []
        for n in neighbors:
            if not any(dominates_sol(a, n) for a in archive):
                new_valid_neighbors.append(n)
                
        if new_valid_neighbors:
            archive.extend(new_valid_neighbors)
            archive = filter_nondominated_sols(archive)
                
        iterations += 1

    temps_ecoule = time.time() - start_time
    print(f"Fin de la PLS : {len(archive)} solutions trouvées en {temps_ecoule:.2f}s ({iterations} itérations).")
    return archive

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    inst = read_instance(r"C:\Users\floim\Desktop\Florian CI2\Semestre 2\P2\MORE TP Amine\Projet Final\instances\instances\CALBP_OTTO_Mitchell_n21.txt")
    
    front_meta = pareto_local_search(inst, time_limit=30.0)
    
    print("\n--- FRONT DE PARETO (MÉTAHEURISTIQUE - SCHOLL n=100) ---")
    for i, sol in enumerate(front_meta, start=1):
        print(f"Point {i}: Coût = {sol.cost:.2f}, Énergie = {sol.energy:.2f}")
        
    meta_costs = [sol.cost for sol in front_meta]
    meta_energies = [sol.energy for sol in front_meta]
    
    plt.figure(figsize=(9, 6))
    plt.plot(meta_costs, meta_energies, marker='o', linestyle='-', color='red', 
             linewidth=2, markersize=8, label='Recherche Locale (PLS)')

    # Annotation de quelques points (pas tous pour ne pas surcharger)
    step = max(1, len(meta_costs) // 10)
    for i in range(0, len(meta_costs), step):
        plt.annotate(f"({meta_costs[i]:.0f}, {meta_energies[i]:.1f})", 
                     (meta_costs[i], meta_energies[i]), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontsize=8)

    plt.xlabel("Coût d'exploitation ($z_c$)", fontsize=12)
    plt.ylabel("Consommation énergétique ($z_e$)", fontsize=12)
    plt.title("Front de Pareto Métaheuristique (SCHOLL n=100)", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    plt.savefig("pareto_scholl_100.png", dpi=300)
    print("\nGraphe sauvegardé sous 'pareto_scholl_100.png'")
    plt.show()
    
    print("\nNote : Impossible de comparer avec l'hypervolume exact car la méthode MIP classique échoue sur 100 tâches.")
