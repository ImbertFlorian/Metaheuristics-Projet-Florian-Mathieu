import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ============================================================
# 1. STRUCTURES
# ============================================================

@dataclass
class Individual:
    perm: List[int]                  # permutation des tâches
    modes: Dict[int, int]            # mode choisi pour chaque tâche
    cost: float = math.inf
    energy: float = math.inf
    energy_raw: int = 10**18
    rank: Optional[int] = None
    crowding: float = 0.0
    assignment: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # j -> (station, mode)
    stations: Dict[int, dict] = field(default_factory=dict)
    feasible: bool = True


# ============================================================
# 2. OUTILS PRECEDENCE
# ============================================================

def build_precedence_helpers(data):
    J = data["J"]
    P = data["P"]

    preds = {j: set() for j in J}
    succs = {j: set() for j in J}

    for i, j in P:
        preds[j].add(i)
        succs[i].add(j)

    return preds, succs


def randomized_topological_order(J, preds, succs, rng):
    indeg = {j: len(preds[j]) for j in J}
    avail = [j for j in J if indeg[j] == 0]
    order = []

    while avail:
        rng.shuffle(avail)
        j = avail.pop()
        order.append(j)
        for s in succs[j]:
            indeg[s] -= 1
            if indeg[s] == 0:
                avail.append(s)

    if len(order) != len(J):
        raise ValueError("Le graphe de précédence contient probablement un cycle.")
    return order


def repair_permutation_topological(perm, preds, succs):
    """
    Répare une permutation quelconque pour en faire un ordre topologique
    en respectant au mieux les priorités de 'perm'.
    """
    priority = {task: idx for idx, task in enumerate(perm)}
    indeg = {j: len(preds[j]) for j in preds}
    avail = [j for j in preds if indeg[j] == 0]
    order = []

    while avail:
        avail.sort(key=lambda x: priority.get(x, 10**9))
        j = avail.pop(0)
        order.append(j)
        for s in succs[j]:
            indeg[s] -= 1
            if indeg[s] == 0:
                avail.append(s)

    if len(order) != len(perm):
        raise ValueError("Impossible de réparer la permutation : cycle probable.")
    return order


# ============================================================
# 3. MODES FAISABLES / REPARATION DES MODES
# ============================================================

def feasible_modes_for_task(data, j):
    T = data["T"]
    return [m for m in data["M"] if data["t"][(j, m)] <= T]


def repair_modes(data, modes):
    """
    Si un mode choisi est impossible (temps > T), on le remplace
    par le mode faisable de plus petit temps.
    """
    repaired = {}
    for j in data["J"]:
        m = modes[j]
        feasible = feasible_modes_for_task(data, j)
        if not feasible:
            raise ValueError(f"Aucun mode faisable pour la tâche {j}.")
        if m not in feasible:
            repaired[j] = min(feasible, key=lambda mm: data["t"][(j, mm)])
        else:
            repaired[j] = m
    return repaired


# ============================================================
# 4. DECODER : PERM + MODES -> SOLUTION (stations, coût, énergie)
# ============================================================

def decode_individual(data, perm, modes):
    """
    Décode un individu :
    - répare l'ordre pour respecter les précédences
    - répare les modes impossibles
    - affecte chaque tâche à la première station faisable
      respectant la précédence et le temps de cycle
    - calcule coût et énergie
    """
    J = data["J"]
    K = data["K"]
    MH = set(data["MH"])
    MC = set(data["MC"])
    T = data["T"]
    C_s = data["C_s"]
    C_w = data["C_w"]
    C_c = data["C_c"]
    R_e = data["R_e"]
    t = data["t"]
    E = data["E"]
    scale = data.get("energy_scale", 1)

    preds, succs = build_precedence_helpers(data)

    perm = repair_permutation_topological(perm, preds, succs)
    modes = repair_modes(data, modes)

    station_load = {}          # k -> somme des temps
    station_worker = {}        # k -> 0/1
    station_cobot = {}         # k -> 0/1
    station_tasks = {}         # k -> [(j,m)]
    station_cobot_active = {}  # k -> somme des temps des tâches utilisant cobot

    assignment = {}            # j -> (k,m)

    open_stations = []

    for j in perm:
        m = modes[j]
        proc = t[(j, m)]

        pred_station = 1
        if preds[j]:
            pred_station = max(assignment[p][0] for p in preds[j])

        chosen_k = None
        for k in open_stations:
            if k < pred_station:
                continue
            if station_load[k] + proc <= T:
                chosen_k = k
                break

        if chosen_k is None:
            if len(open_stations) >= len(K):
                # pénalisation si on dépasse le nombre de stations candidates
                return {
                    "feasible": False,
                    "cost": 1e12,
                    "energy": 1e12,
                    "energy_raw": 10**18,
                    "assignment": {},
                    "stations": {}
                }
            chosen_k = K[len(open_stations)]
            open_stations.append(chosen_k)
            station_load[chosen_k] = 0.0
            station_worker[chosen_k] = 0
            station_cobot[chosen_k] = 0
            station_tasks[chosen_k] = []
            station_cobot_active[chosen_k] = 0.0

        station_load[chosen_k] += proc
        station_tasks[chosen_k].append((j, m))
        assignment[j] = (chosen_k, m)

        if m in MH:
            station_worker[chosen_k] = 1
        if m in MC:
            station_cobot[chosen_k] = 1
            station_cobot_active[chosen_k] += proc

    cost = 0.0
    energy_tasks_raw = 0
    energy_idle_raw = 0
    stations = {}

    for k in open_stations:
        wk = station_worker[k]
        yk = station_cobot[k]
        idle_time = T - station_cobot_active[k] if yk == 1 else 0.0

        cost += C_s + C_w * wk + C_c * yk

        for (j, m) in station_tasks[k]:
            energy_tasks_raw += E[(j, m)]

        energy_idle_raw += int(round(R_e * idle_time))

        stations[k] = {
            "worker": wk,
            "cobot": yk,
            "idle_time": idle_time,
            "tasks": sorted(station_tasks[k], key=lambda x: x[0]),
        }

    energy_raw = energy_tasks_raw + energy_idle_raw
    energy = energy_raw / scale

    return {
        "feasible": True,
        "cost": cost,
        "energy": energy,
        "energy_raw": energy_raw,
        "assignment": assignment,
        "stations": stations
    }


def evaluate_individual(data, ind):
    sol = decode_individual(data, ind.perm, ind.modes)
    ind.feasible = sol["feasible"]
    ind.cost = sol["cost"]
    ind.energy = sol["energy"]
    ind.energy_raw = sol["energy_raw"]
    ind.assignment = sol["assignment"]
    ind.stations = sol["stations"]


# ============================================================
# 5. DOMINANCE / NSGA-II
# ============================================================

def dominates(a, b):
    return (
        a.cost <= b.cost and
        a.energy_raw <= b.energy_raw and
        (a.cost < b.cost or a.energy_raw < b.energy_raw)
    )


def fast_non_dominated_sort(pop):
    fronts = []
    S = [[] for _ in range(len(pop))]
    n = [0 for _ in range(len(pop))]
    first_front = []

    for i, p in enumerate(pop):
        for j, q in enumerate(pop):
            if i == j:
                continue
            if dominates(p, q):
                S[i].append(j)
            elif dominates(q, p):
                n[i] += 1

        if n[i] == 0:
            p.rank = 0
            first_front.append(i)

    fronts_idx = [first_front]
    current = 0

    while fronts_idx[current]:
        next_front = []
        for i in fronts_idx[current]:
            for j in S[i]:
                n[j] -= 1
                if n[j] == 0:
                    pop[j].rank = current + 1
                    next_front.append(j)
        current += 1
        fronts_idx.append(next_front)

    if not fronts_idx[-1]:
        fronts_idx.pop()

    fronts = [[pop[i] for i in front] for front in fronts_idx]
    return fronts


def crowding_distance(front):
    if not front:
        return

    for ind in front:
        ind.crowding = 0.0

    if len(front) <= 2:
        for ind in front:
            ind.crowding = float("inf")
        return

    # objectif coût
    front.sort(key=lambda x: x.cost)
    front[0].crowding = front[-1].crowding = float("inf")
    cmin, cmax = front[0].cost, front[-1].cost
    if cmax > cmin:
        for i in range(1, len(front) - 1):
            front[i].crowding += (front[i + 1].cost - front[i - 1].cost) / (cmax - cmin)

    # objectif énergie
    front.sort(key=lambda x: x.energy_raw)
    front[0].crowding = front[-1].crowding = float("inf")
    emin, emax = front[0].energy_raw, front[-1].energy_raw
    if emax > emin:
        for i in range(1, len(front) - 1):
            front[i].crowding += (front[i + 1].energy_raw - front[i - 1].energy_raw) / (emax - emin)


def nsga2_select(pop, pop_size):
    fronts = fast_non_dominated_sort(pop)
    new_pop = []

    for front in fronts:
        crowding_distance(front)
        if len(new_pop) + len(front) <= pop_size:
            new_pop.extend(front)
        else:
            front.sort(key=lambda x: x.crowding, reverse=True)
            remaining = pop_size - len(new_pop)
            new_pop.extend(front[:remaining])
            break

    return new_pop


def tournament_selection(pop, rng):
    a = rng.choice(pop)
    b = rng.choice(pop)

    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    return a if a.crowding > b.crowding else b


# ============================================================
# 6. OPERATEURS GENETIQUES
# ============================================================

def order_crossover(p1, p2, rng):
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n

    child[a:b+1] = p1[a:b+1]

    fill = [x for x in p2 if x not in child]
    ptr = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[ptr]
            ptr += 1
    return child


def uniform_mode_crossover(m1, m2, tasks, rng):
    child = {}
    for j in tasks:
        child[j] = m1[j] if rng.random() < 0.5 else m2[j]
    return child


def mutate_permutation(perm, rng, p_swap=0.2):
    perm = perm[:]
    if rng.random() < p_swap:
        i, j = sorted(rng.sample(range(len(perm)), 2))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def mutate_modes(data, modes, rng, p_mode=0.15):
    modes = dict(modes)
    for j in data["J"]:
        if rng.random() < p_mode:
            feasible = feasible_modes_for_task(data, j)
            modes[j] = rng.choice(feasible)
    return modes


# ============================================================
# 7. HEURISTIQUES CONSTRUCTIVES INITIALES
# ============================================================

def cheapest_mode_for_task(data, j):
    feasible = feasible_modes_for_task(data, j)

    # proxy simple fondé sur le sujet :
    # CI (2) souvent moins cher, HI (1) ensuite, SU (3) plus cher
    pref = {2: 0, 1: 1, 3: 2}
    return min(feasible, key=lambda m: pref.get(m, 99))


def lowest_energy_mode_for_task(data, j):
    feasible = feasible_modes_for_task(data, j)
    return min(feasible, key=lambda m: data["E"][(j, m)])


def constructive_individual_cost(data, rng):
    preds, succs = build_precedence_helpers(data)
    perm = randomized_topological_order(data["J"], preds, succs, rng)
    modes = {j: cheapest_mode_for_task(data, j) for j in data["J"]}
    return Individual(perm=perm, modes=modes)


def constructive_individual_energy(data, rng):
    preds, succs = build_precedence_helpers(data)
    perm = randomized_topological_order(data["J"], preds, succs, rng)
    modes = {j: lowest_energy_mode_for_task(data, j) for j in data["J"]}
    return Individual(perm=perm, modes=modes)


def random_individual(data, rng):
    preds, succs = build_precedence_helpers(data)
    perm = randomized_topological_order(data["J"], preds, succs, rng)
    modes = {j: rng.choice(feasible_modes_for_task(data, j)) for j in data["J"]}
    return Individual(perm=perm, modes=modes)


# ============================================================
# 8. ALGORITHME GENETIQUE MULTI-OBJECTIF (NSGA-II)
# ============================================================

import time

def run_nsga2_calbp(
    data,
    pop_size=60,
    generations=100,
    crossover_rate=0.9,
    mutation_rate=0.3,
    seed=42,
    verbose=True,
    time_limit_seconds=None
):
    rng = random.Random(seed)

    # -------------------------
    # Initialisation
    # -------------------------
    population = []

    # deux heuristiques constructives imposées par le sujet
    population.append(constructive_individual_cost(data, rng))
    population.append(constructive_individual_energy(data, rng))

    while len(population) < pop_size:
        population.append(random_individual(data, rng))

    for ind in population:
        evaluate_individual(data, ind)

    population = nsga2_select(population, pop_size)

    # -------------------------
    # Boucle principale
    # -------------------------
    import time

    start_time = time.time()
    gen = 0

    while True:
        # arrêt si temps dépassé
        if time_limit_seconds is not None and (time.time() - start_time >= time_limit_seconds):
            print("Arrêt : limite de temps atteinte")
            break

        # arrêt si générations atteintes
        if gen >= generations:
            print("Arrêt : nombre de générations atteint")
            break

        gen += 1

        # ======== ton code GA ========
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            crowding_distance(front)

        offspring = []

        while len(offspring) < pop_size:
            p1 = tournament_selection(population, rng)
            p2 = tournament_selection(population, rng)

            if rng.random() < crossover_rate:
                child_perm = order_crossover(p1.perm, p2.perm, rng)
                child_modes = uniform_mode_crossover(p1.modes, p2.modes, data["J"], rng)
            else:
                child_perm = p1.perm[:]
                child_modes = dict(p1.modes)

            if rng.random() < mutation_rate:
                child_perm = mutate_permutation(child_perm, rng)
                child_modes = mutate_modes(data, child_modes, rng)

            child = Individual(perm=child_perm, modes=child_modes)
            evaluate_individual(data, child)
            offspring.append(child)

        population = nsga2_select(population + offspring, pop_size)

        if verbose and gen % 10 == 0:
            f0 = fast_non_dominated_sort(population)[0]
            print(f"Génération {gen} | front = {len(f0)}")
    total_time = time.time() - start_time
    print(f"Temps total GA : {total_time:.2f} secondes")

    # -------------------------
    # Front final
    # -------------------------
    fronts = fast_non_dominated_sort(population)
    pareto = fronts[0]

    # dédoublonnage
    seen = set()
    unique = []
    for ind in pareto:
        key = (round(ind.cost, 6), ind.energy_raw)
        if key not in seen:
            seen.add(key)
            unique.append(ind)

    unique.sort(key=lambda x: (x.energy_raw, x.cost))

    results = []
    for ind in unique:
        results.append({
            "cost": ind.cost,
            "energy": ind.energy,
            "energy_raw": ind.energy_raw,
            "stations": ind.stations,
            "assignment": ind.assignment,
            "perm": ind.perm,
            "modes": ind.modes,
        })

    return results, population

import matplotlib.pyplot as plt

def plot_pareto_front_ga(pareto, data, title="Front de Pareto GA", save_path=None, show_plot=True):
    if not pareto:
        print("Aucune solution à afficher.")
        return

    # tri pour avoir une belle courbe
    pareto_sorted = sorted(pareto, key=lambda s: (s["cost"], s["energy"]))

    costs = [s["cost"] for s in pareto_sorted]
    energies = [s["energy"] for s in pareto_sorted]
    
    ref_point = compute_reference_point(data)
    hv = compute_hypervolume(pareto_sorted, ref_point)
    hv_norm = compute_hv_norm(hv, ref_point)

    plt.figure(figsize=(8, 5))

    # ligne + points
    plt.plot(costs, energies, marker="o")
    
    plt.xlabel("Coût total")
    plt.ylabel("Consommation énergétique")
    plt.title(f"{title}\nHV = {hv:.2f} | HVnorm = {hv_norm:.4f}")
    plt.grid(True)

    # annotation
    for i, s in enumerate(pareto_sorted, start=1):
        plt.annotate(
            f"S{i}\n({s['cost']:.1f}, {s['energy']:.1f})",
            (s["cost"], s["energy"]),
            textcoords="offset points",
            xytext=(5, 5)
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Graphique sauvegardé : {save_path}")

    if show_plot:
        plt.show()

    plt.close()
    

def compute_reference_point(data):
    K = data["K"]
    J = data["J"]
    M = data["M"]

    C_s = data["C_s"]
    C_w = data["C_w"]
    C_c = data["C_c"]
    E = data["E"]
    R_e = data["R_e"]
    T = data["T"]
    scale = data.get("energy_scale", 1)

    cost_ref = len(K) * (C_s + C_w + C_c) + 1

    # conversion en unités réelles
    energy_tasks_ref = sum(max(E[(j, m)] for m in M) for j in J) / scale
    energy_idle_ref = (R_e / scale) * len(K) * T

    energy_ref = energy_tasks_ref + energy_idle_ref + 1

    return (float(cost_ref), float(energy_ref))

def compute_hypervolume(pareto, ref_point):
    if not pareto:
        return 0.0

    pareto_sorted = sorted(pareto, key=lambda s: s["cost"])

    hv = 0.0
    prev_energy = ref_point[1]

    for s in pareto_sorted:
        width = ref_point[0] - s["cost"]
        height = prev_energy - s["energy"]
        hv += width * height
        prev_energy = s["energy"]

    return hv

def compute_hv_norm(hv, ref_point):
    hv_max = ref_point[0] * ref_point[1]
    return hv / hv_max


# ============================================================
# 9. AFFICHAGE
# ============================================================

def print_ga_pareto_front(pareto):
    print("\nFront de Pareto approché (GA)")
    print("-" * 60)
    for idx, s in enumerate(pareto, 1):
        print(f"Solution {idx}: coût = {s['cost']:.4f}, énergie = {s['energy']:.4f}")
        for k, info in sorted(s["stations"].items()):
            print(
                f"  Station {k}: worker={info['worker']}, cobot={info['cobot']}, "
                f"idle={info['idle_time']:.4f}, tasks={info['tasks']}"
            )
        print("-" * 60)


# ============================================================
# 10. EXEMPLE D'UTILISATION
# ============================================================

if __name__ == "__main__":
    # IMPORTANT :
    # ce module suppose que tu as déjà une fonction read_instance(filepath, energy_scale)
    # identique ou compatible avec celle que tu utilises dans ton code exact.

    import os

    # adapte cet import selon ton projet
    # from ton_fichier_exact import read_instance
    from e_constraint import read_instance

    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "instances", "CALBP_OTTO_Jackson_n11.txt")

    data = read_instance(filepath, energy_scale=100)

    pareto_ga, _ = run_nsga2_calbp(
        data,
        pop_size=80,
        crossover_rate=0.9,
        mutation_rate=0.35,
        seed=42,
        verbose=True,
        time_limit_seconds=None
    ) 
    print_ga_pareto_front(pareto_ga)
        
    # Nom du fichier image
    instance_name = os.path.splitext(os.path.basename(filepath))[0]
    save_graph = os.path.join(base_dir, f"pareto_ga_{instance_name}.png")

    # Tracé du front GA
    plot_pareto_front_ga(
        pareto_ga,
        data,
        title="Front de Pareto GA",
        save_path="pareto_ga.png"
    )

    
    
