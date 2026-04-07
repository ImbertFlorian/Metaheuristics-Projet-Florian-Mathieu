import os
import glob
import time
import random
import matplotlib.pyplot as plt
from typing import Tuple



from Local_Search_V2 import (
    read_instance, Solution, constructive_heuristic_cost, 
    constructive_heuristic_energy, filter_nondominated_sols, 
    generate_neighbors, dominates_sol, build_predecessors_dict
)

# ============================================================
# FONCTIONS ADAPTÉES POUR LE TUNING
# ============================================================

def calculate_reference_point(inst) -> Tuple[float, float]:
    """
    Calcule le point de référence (tau) pour l'hypervolume selon la formule du sujet.
    """
    num_k = len(inst.stations)
    ref_cost = num_k * (inst.Cs + inst.Cw + inst.Cc) + 1
    max_energy_sum = sum(max(inst.E[(j, m)] for m in inst.modes) for j in inst.tasks)
    ref_energy = max_energy_sum + (inst.Re * num_k * inst.T) + 1
    return (ref_cost, ref_energy)

def calculate_hypervolume(pareto_front, ref_point) -> float:
    if not pareto_front: return 0.0
    if isinstance(pareto_front[0], dict): points = [(p['cost'], p['energy']) for p in pareto_front]
    elif hasattr(pareto_front[0], 'cost'): points = [(p.cost, p.energy) for p in pareto_front]
    else: points = pareto_front
        
    points.sort(key=lambda x: x[0])
    ref_c, ref_e = ref_point
    hv = 0.0
    current_ref_e = ref_e
    
    for cost, energy in points:
        if cost > ref_c or energy > ref_e: continue
        width = ref_c - cost
        height = current_ref_e - energy
        hv += width * height
        current_ref_e = energy
    return hv

def generate_random_valid_solution_tuned(inst, all_preds, nb_swaps) -> Solution:
    seq = list(sorted(inst.tasks))
    for _ in range(nb_swaps): 
        i = random.randint(0, len(seq) - 2)
        if seq[i] not in all_preds[seq[i+1]]:
            seq[i], seq[i+1] = seq[i+1], seq[i]
            
    modes = {j: random.choice(inst.modes) for j in seq}
    sol = Solution(inst, seq, modes)
    sol.decode_and_evaluate()
    return sol

def pareto_local_search_tuned(inst, time_limit, nb_initial_sols, nb_swaps) -> list:
    all_preds = build_predecessors_dict(inst)
    initial_sols = [constructive_heuristic_cost(inst), constructive_heuristic_energy(inst)]
    
    for _ in range(nb_initial_sols):
        initial_sols.append(generate_random_valid_solution_tuned(inst, all_preds, nb_swaps))
        
    archive = filter_nondominated_sols(initial_sols)
    explored_signatures = set()
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        current_sol = None
        for sol in archive:
            sig = (tuple(sol.task_seq), tuple(sorted(sol.task_modes.items())))
            if sig not in explored_signatures:
                current_sol = sol
                explored_signatures.add(sig)
                break
                
        if current_sol is None:
            # Si on a convergé avant la fin du temps, on relance !
            new_sol = generate_random_valid_solution_tuned(inst, all_preds, nb_swaps)
            archive.append(new_sol)
            archive = filter_nondominated_sols(archive)
            continue
            
        neighbors = generate_neighbors(current_sol, all_preds)
        for n in neighbors:
            is_dominated = any(dominates_sol(a, n) for a in archive)
            if not is_dominated:
                archive.append(n)
                archive = filter_nondominated_sols(archive)
                
    return archive

# ============================================================
# SCRIPT DE PARAMETER TUNING (GRID SEARCH)
# ============================================================

if __name__ == "__main__":
    print("Démarrage du Parameter Tuning...\n")
    
    
    filepath = r"C:\Users\floim\Desktop\Florian CI2\Semestre 2\P2\MORE TP Amine\Projet Final\instances\instances\CALBP_SCHOLL_instance_n=20_17.txt"
    inst = read_instance(filepath)
    instance_name = os.path.splitext(os.path.basename(filepath))[0]
    
    print("============================================================")
    print(f"=== TUNING DE L'INSTANCE : {instance_name} ===")
    print("============================================================")
    
    # Calcul automatique du point de référence pour l'instance
    ref_point = calculate_reference_point(inst)
    print(f"Point de référence (tau) pour cette instance : {ref_point}\n")
    
    # 1. Définition de la grille de paramètres à tester 
    list_nb_initial_sols = [5, 10, 20, 30]  # 4 valeurs
    list_nb_swaps = [5, 10, 20, 30]         # 4 valeurs
    
    # Paramètres de l'expérience
    runs_per_config = 3   # On lance 3 fois pour faire une moyenne (statistique)
    time_limit_tuning = 5.0 # 5 secondes max par configuration (ça suffit pour comparer)
    
    print(f"Configurations à tester : {len(list_nb_initial_sols) * len(list_nb_swaps)}")
    print(f"Total des runs prévus  : {len(list_nb_initial_sols) * len(list_nb_swaps) * runs_per_config}")
    print(f"Temps max estimé       : {len(list_nb_initial_sols) * len(list_nb_swaps) * runs_per_config * time_limit_tuning} secondes\n")
    
    print(f"{'Init Sols':<12} | {'Swaps':<8} | {'HV Moyen':<10} | {'HV Max':<10}")
    print("-" * 50)
    
    best_config = None
    best_avg_hv = -1
    
    # 2. Boucle de test (Grid Search)
    for initial_sols in list_nb_initial_sols:
        for swaps in list_nb_swaps:
            
            hvs = []
            for r in range(runs_per_config):
                # On lance l'algorithme avec les paramètres de la boucle actuelle
                front = pareto_local_search_tuned(inst, time_limit_tuning, initial_sols, swaps)
                hv = calculate_hypervolume(front, ref_point)
                hvs.append(hv)
                
            avg_hv = sum(hvs) / len(hvs)
            max_hv = max(hvs)
            
            # Affichage de la ligne du tableau
            print(f"{initial_sols:<12} | {swaps:<8} | {avg_hv:<10.2f} | {max_hv:<10.2f}")
            
            # Sauvegarde du meilleur
            if avg_hv > best_avg_hv:
                best_avg_hv = avg_hv
                best_config = (initial_sols, swaps)
                
    print("-" * 50)
    print(f"\n MEILLEURE CONFIGURATION TROUVÉE :")
    print(f"-> nb_initial_sols = {best_config[0]} (Taille de l'archive de départ)")
    print(f"-> nb_swaps = {best_config[1]} (Force de la perturbation)")
    print(f"-> Hypervolume Moyen = {best_avg_hv:.2f}")
    print("\nNote pour le rapport : Vous devez intégrer ces paramètres gagnants dans votre code final !")
