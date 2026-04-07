from docplex.mp.model import Model
import os
import matplotlib.pyplot as plt

# ============================================================
# 1. LECTURE D'INSTANCE
# ============================================================

def read_instance(filepath, energy_scale=100):
    """
    Lit une instance texte et convertit les énergies en entiers
    via energy_scale pour permettre une epsilon-constraint exacte.

    Exemple :
        0.66 -> 66 si energy_scale = 100
    """

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    data = {
        "J": [],
        "K": [],
        "M": [],
        "t": {},
        "E": {},
        "P": [],
        "R_e": None,
        "T": None,
        "C_s": None,
        "C_w": None,
        "C_c": None,
        "energy_scale": energy_scale,
    }

    section = None

    for line in lines:
        if line.startswith("#"):
            title = line.lower().strip()

            if "tasks" in title:
                section = "tasks"
            elif "workstations" in title:
                section = "workstations"
            elif "modes" in title:
                section = "modes"
            elif "processing times" in title:
                section = "processing"
            elif "energy" in title:
                section = "energy"
            elif "precedence relations" in title:
                section = "precedence"
            elif "parameters" in title:
                section = "parameters"
            else:
                section = None
            continue

        if section == "tasks":
            data["J"] = [int(x) for x in line.split()]
            section = None

        elif section == "workstations":
            data["K"] = [int(x) for x in line.split()]
            section = None

        elif section == "modes":
            data["M"] = [int(x) for x in line.split()]
            section = None

        elif section == "processing":
            parts = line.split()
            if len(parts) == 3:
                j, m, val = int(parts[0]), int(parts[1]), float(parts[2])
                data["t"][(j, m)] = val

        elif section == "energy":
            parts = line.split()
            if len(parts) == 3:
                j, m, val = int(parts[0]), int(parts[1]), float(parts[2])
                data["E"][(j, m)] = int(round(val * energy_scale))

        elif section == "precedence":
            if "," in line:
                i, j = line.split(",")
                data["P"].append((int(i.strip()), int(j.strip())))

        elif section == "parameters":
            if "=" in line:
                key, value = line.split("=")
                key = key.strip()
                value = float(value.strip())

                if key == "R_e":
                    data[key] = int(round(value * energy_scale))
                else:
                    data[key] = value

    # Vérifications et compléments
    for j in data["J"]:
        for m in data["M"]:
            if (j, m) not in data["t"]:
                raise ValueError(f"Temps manquant pour (j={j}, m={m})")
            if (j, m) not in data["E"]:
                if m == 1:
                    data["E"][(j, m)] = 0
                else:
                    raise ValueError(f"Energie manquante pour (j={j}, m={m})")

    # Pré-calculs utiles
    data["MH"] = [1, 3]   # modes nécessitant un humain
    data["MC"] = [2, 3]   # modes nécessitant un cobot
    data["JKM"] = [(j, k, m) for j in data["J"] for k in data["K"] for m in data["M"]]

    return data


# ============================================================
# 2. CONSTRUCTION DU MODELE MILP
# ============================================================
def build_calbp_model(data, epsilon=None, primary="cost"):
    """
    Construit le modèle MILP CALBP.

    primary = 'cost'   -> min z_c
    primary = 'energy' -> min z_e

    Si epsilon n'est pas None, ajoute la contrainte z_e <= epsilon.
    """

    J = data["J"]
    K = data["K"]
    M = data["M"]
    MH = data["MH"]
    MC = data["MC"]
    JKM = data["JKM"]

    t = data["t"]
    E = data["E"]
    P = data["P"]

    T = data["T"]
    C_s = data["C_s"]
    C_w = data["C_w"]
    C_c = data["C_c"]
    R_e = data["R_e"]

    mdl = Model(name="CALBP_epsilon_constraint")

    # Variables
    x = mdl.binary_var_dict(JKM, name="x")
    o = mdl.binary_var_dict(K, name="o")
    w = mdl.binary_var_dict(K, name="w")
    y = mdl.binary_var_dict(K, name="y")
    l = mdl.continuous_var_dict(K, lb=0, name="l")

    # Objectifs
    z_c = mdl.sum(C_s * o[k] + C_w * w[k] + C_c * y[k] for k in K)
    z_e_tasks = mdl.sum(E[(j, m)] * x[j, k, m] for (j, k, m) in JKM)
    z_e_idle = mdl.sum(R_e * l[k] for k in K)
    z_e = z_e_tasks + z_e_idle

    # 1) Affectation unique
    for j in J:
        mdl.add_constraint(mdl.sum(x[j, k, m] for k in K for m in M) == 1)

    # 2) Temps de cycle
    for k in K:
        mdl.add_constraint(
            mdl.sum(t[(j, m)] * x[j, k, m] for j in J for m in M) <= T * o[k]
        )

    # 3) Précédence
    for (i, j) in P:
        mdl.add_constraint(
            mdl.sum(k * x[i, k, m] for k in K for m in M)
            <=
            mdl.sum(k * x[j, k, m] for k in K for m in M)
        )

    # 4) Liaison modes -> worker
    for j in J:
        for k in K:
            for m in MH:
                mdl.add_constraint(x[j, k, m] <= w[k])

    # 5) Liaison modes -> cobot
    for j in J:
        for k in K:
            for m in MC:
                mdl.add_constraint(x[j, k, m] <= y[k])

    # 6) Ressources seulement si station ouverte
    for k in K:
        mdl.add_constraint(w[k] <= o[k])
        mdl.add_constraint(y[k] <= o[k])

    # 7) Temps idle du cobot
    for k in K:
        mdl.add_constraint(
            l[k] == T * y[k] - mdl.sum(t[(j, m)] * x[j, k, m] for j in J for m in MC)
        )

    # 8) Casser la symétrie sur les stations ouvertes
    for idx in range(len(K) - 1):
        mdl.add_constraint(o[K[idx]] >= o[K[idx + 1]])

    # Epsilon-constraint
    if epsilon is not None:
        mdl.add_constraint(z_e <= epsilon)

    # Objectif principal
    if primary == "cost":
        mdl.minimize(z_c)
    elif primary == "energy":
        mdl.minimize(z_e)
    else:
        raise ValueError("primary doit être 'cost' ou 'energy'.")

    return mdl, x, o, w, y, l, z_c, z_e


# ============================================================
# 3. PARAMETRAGE SOLVEUR
# ============================================================

def configure_solver(mdl, time_limit=240, mip_gap=0.0, threads=0):
    """
    Configure CPLEX/DOcplex.
    threads=0 : laisse CPLEX décider / utiliser les coeurs disponibles.
    """
    if time_limit is not None:
        mdl.parameters.timelimit = time_limit

    if mip_gap is not None:
        mdl.parameters.mip.tolerances.mipgap = mip_gap

    mdl.parameters.threads = threads


# ============================================================
# 4. EXTRACTION DE SOLUTION
# ============================================================

def extract_solution(data, mdl, x, o, w, y, l, z_c, z_e):
    """
    Extrait la solution complète.
    """
    if mdl.solution is None:
        return None

    J = data["J"]
    K = data["K"]
    M = data["M"]
    scale = data.get("energy_scale", 1)

    assignment = {}
    for j in J:
        found = False
        for k in K:
            for m in M:
                if x[j, k, m].solution_value > 0.5:
                    assignment[j] = (k, m)
                    found = True
                    break
            if found:
                break

    stations = {}
    for k in K:
        if o[k].solution_value > 0.5:
            stations[k] = {
                "worker": int(round(w[k].solution_value)),
                "cobot": int(round(y[k].solution_value)),
                "idle_time": l[k].solution_value,
                "tasks": []
            }

    for j, (k, m) in assignment.items():
        if k in stations:
            stations[k]["tasks"].append((j, m))

    return {
        "cost": float(z_c.solution_value),
        "energy": float(z_e.solution_value) / scale,
        "energy_raw": int(round(z_e.solution_value)),
        "stations": stations,
        "assignment": assignment
    }


# ============================================================
# 5. BORNES
# ============================================================

def compute_energy_bounds(data, time_limit=None, mip_gap=0.0, threads=0):
    """
    Calcule :
    - min_energy : solution qui minimise l'énergie
    - max_energy : énergie de la solution qui minimise le coût
    - min_cost   : coût minimum
    - cost_at_min_energy : coût de la solution à énergie minimale
    """

    # Min énergie
    mdl_e, x, o, w, y, l, z_c, z_e = build_calbp_model(data, epsilon=None, primary="energy")
    configure_solver(mdl_e, time_limit=time_limit, mip_gap=mip_gap, threads=threads)
    sol_e = mdl_e.solve(log_output=False)
    
    print("=== Min énergie ===")
    print("Status:", mdl_e.solve_details.status)
    print("Gap:", mdl_e.solve_details.mip_relative_gap)
    if sol_e is None:
        raise RuntimeError("Impossible de résoudre le modèle de minimisation énergie.")

    min_energy = int(round(z_e.solution_value))
    cost_at_min_energy = float(z_c.solution_value)

    # Min coût
    mdl_c, x, o, w, y, l, z_c, z_e = build_calbp_model(data, epsilon=None, primary="cost")
    configure_solver(mdl_c, time_limit=time_limit, mip_gap=mip_gap, threads=threads)
    sol_c = mdl_c.solve(log_output=False)
    print("=== Min coût ===")
    print("Status:", mdl_c.solve_details.status)
    print("Gap:", mdl_c.solve_details.mip_relative_gap)
    if sol_c is None:
        raise RuntimeError("Impossible de résoudre le modèle de minimisation coût.")

    max_energy = int(round(z_e.solution_value))
    min_cost = float(z_c.solution_value)

    return {
        "min_energy": min_energy,
        "max_energy": max_energy,
        "min_cost": min_cost,
        "cost_at_min_energy": cost_at_min_energy
    }


# ============================================================
# 6. FILTRAGE NON DOMINE
# ============================================================

def is_dominated(sol, others):
    for other in others:
        if other is sol:
            continue
        if (other["cost"] <= sol["cost"] and other["energy_raw"] <= sol["energy_raw"] and
            (other["cost"] < sol["cost"] or other["energy_raw"] < sol["energy_raw"])):
            return True
    return False


def filter_nondominated(solutions):
    unique = []
    seen = set()

    for s in solutions:
        key = (round(s["cost"], 6), s["energy_raw"])
        if key not in seen:
            seen.add(key)
            unique.append(s)

    nd = [s for s in unique if not is_dominated(s, unique)]
    nd.sort(key=lambda s: (s["energy_raw"], s["cost"]))
    return nd


# ============================================================
# 7. EPSILON-CONSTRAINT EXACT ITÉRATIF
# ============================================================

def exact_pareto_front_epsilon_constraint(data, time_limit=None, mip_gap=0.0, threads=0):
    """
    Génère le front de Pareto exact par epsilon-constraint itératif.

    Idée :
    - on part de eps = max_energy
    - on résout min cost s.c. energy <= eps
    - si une solution a énergie e*, on saute directement à eps = e* - 1

    C'est exact tant que :
    - z_e est entier (grâce à energy_scale)
    - chaque sous-problème est résolu optimalement
    """

    bounds = compute_energy_bounds(
        data,
        time_limit=time_limit,
        mip_gap=mip_gap,
        threads=threads
    )

    min_energy = int(bounds["min_energy"])
    eps = int(bounds["max_energy"])

    solutions = []
    seen = set()

    while eps >= min_energy:
        mdl, x, o, w, y, l, z_c, z_e = build_calbp_model(data, epsilon=eps, primary="cost")
        configure_solver(mdl, time_limit=time_limit, mip_gap=mip_gap, threads=threads)

        sol = mdl.solve(log_output=False)
        print(f"=== Epsilon = {eps} ===")
        print("Status:", mdl.solve_details.status)
        print("Gap:", mdl.solve_details.mip_relative_gap)

        if sol is None:
            eps -= 1
            continue

        s = extract_solution(data, mdl, x, o, w, y, l, z_c, z_e)
        if s is None:
            eps -= 1
            continue

        key = (round(s["cost"], 6), s["energy_raw"])
        if key not in seen:
            seen.add(key)
            s["epsilon"] = eps
            solutions.append(s)

        # saut direct à l'énergie strictement inférieure
        eps = s["energy_raw"] - 1

    pareto = filter_nondominated(solutions)
    return pareto, bounds


# ============================================================
# 8. AFFICHAGE TEXTE
# ============================================================

def print_bounds(bounds, scale=1):
    print("Bornes trouvées :")
    print({
        "min_energy": bounds["min_energy"] / scale,
        "max_energy": bounds["max_energy"] / scale,
        "min_cost": bounds["min_cost"],
        "cost_at_min_energy": bounds["cost_at_min_energy"]
    })


def print_pareto_front(pareto):
    print("\nFront de Pareto exact")
    print("-" * 50)
    for idx, s in enumerate(pareto, 1):
        print(f"Solution {idx}: coût = {s['cost']:.4f}, énergie = {s['energy']:.4f}")
        for k, info in sorted(s["stations"].items()):
            print(
                f"  Station {k}: worker={info['worker']}, cobot={info['cobot']}, "
                f"idle={info['idle_time']:.4f}, tasks={info['tasks']}"
            )
        print("-" * 50)


# ============================================================
# 9. GRAPHIQUE DU FRONT DE PARETO
# ============================================================

def plot_pareto_front(pareto, title="Front de Pareto", save_path=None, show_plot=False):
    """
    Trace et sauvegarde le front de Pareto.
    """
    if not pareto:
        print("Aucune solution à afficher.")
        return

    pareto_sorted = sorted(pareto, key=lambda s: (s["cost"], s["energy"]))
    costs = [s["cost"] for s in pareto_sorted]
    energies = [s["energy"] for s in pareto_sorted]

    plt.figure(figsize=(8, 5))
    plt.plot(costs, energies, marker="o")
    plt.xlabel("Coût total")
    plt.ylabel("Consommation énergétique")
    plt.title(title)
    plt.grid(True)

    for i, s in enumerate(pareto_sorted, start=1):
        plt.annotate(
            f"S{i}\n({s['cost']:.2f}, {s['energy']:.2f})",
            (s["cost"], s["energy"]),
            textcoords="offset points",
            xytext=(5, 5)
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Graphique sauvegardé dans : {save_path}")

    if show_plot:
        plt.show()

    plt.close()


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Choisir ici l'instance à traiter
    filepath = os.path.join(base_dir, "instances", "CALBP_SCHOLL_instance_n=100_473.txt")

    # Lecture unique de l'instance
    data = read_instance(filepath, energy_scale=100)

    # Calcul du front exact
    pareto, bounds = exact_pareto_front_epsilon_constraint(
        data,
        time_limit=240,   # None si tu ne veux pas de limite
        mip_gap=0.0,     # 0.0 = exact
        threads=0        # utilise les coeurs disponibles
    )

    scale = data["energy_scale"]

    print_bounds(bounds, scale=scale)
    print_pareto_front(pareto)

    # Nom du fichier image
    instance_name = os.path.splitext(os.path.basename(filepath))[0]
    save_graph = os.path.join(base_dir, f"pareto_{instance_name}.png")

    plot_pareto_front(
        pareto,
        title=f"Front de Pareto exact - {instance_name}",
        save_path=save_graph,
        show_plot=True
    )
