import os
import time
import math
import csv
import matplotlib.pyplot as plt


# ============================================================
# IMPORTS DES DEUX CODES
# ============================================================

from e_constraint import (
    read_instance,
    exact_pareto_front_epsilon_constraint,
)

from genetic import (
    run_nsga2_calbp,
)


# ============================================================
# 1. OUTILS GENERAUX
# ============================================================

def extract_points(front):
    """
    Transforme un front en liste de points (cost, energy).
    """
    return [(float(s["cost"]), float(s["energy"])) for s in front]


def sort_front(front):
    """
    Trie un front par coût croissant puis énergie croissante.
    """
    return sorted(front, key=lambda s: (s["cost"], s["energy"]))


def filter_nondominated_points(points):
    """
    Filtre les points non dominés dans une liste de tuples (cost, energy).
    Minimisation des deux objectifs.
    """
    unique = sorted(set((round(c, 10), round(e, 10)) for c, e in points))
    nondom = []

    for p in unique:
        dominated = False
        for q in unique:
            if q == p:
                continue
            if (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]):
                dominated = True
                break
        if not dominated:
            nondom.append(p)

    nondom.sort(key=lambda x: (x[0], x[1]))
    return nondom


# ============================================================
# 2. POINT DE REFERENCE
# ============================================================

def compute_reference_point(data):
    """
    D'après le sujet :
    r = ( |K|(Cs + Cw + Cc) + 1,
          sum_j max_m E_jm + R_e |K| T + 1 )
    """
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

    # E est stocké en entier scaled dans ton code exact
    energy_tasks_ref = sum(max(E[(j, m)] for m in M) for j in J) / scale
    energy_idle_ref = R_e * len(K) * T / scale

    energy_ref = energy_tasks_ref + energy_idle_ref + 1

    return (float(cost_ref), float(energy_ref))


# ============================================================
# 3. HYPERVOLUME 2D
# ============================================================

def compute_hypervolume_2d(front, ref_point):
    """
    Hypervolume en 2D pour un problème de minimisation.
    """
    points = extract_points(front) if front and isinstance(front[0], dict) else front
    points = filter_nondominated_points(points)

    if not points:
        return 0.0

    points = sorted(points, key=lambda x: x[0])

    hv = 0.0
    prev_energy = ref_point[1]

    for cost, energy in points:
        width = ref_point[0] - cost
        height = prev_energy - energy

        if width > 0 and height > 0:
            hv += width * height

        prev_energy = min(prev_energy, energy)

    return hv


def compute_hv_norm(front, ref_point):
    """
    HV normalisé par l'aire totale du rectangle [0, ref_point].
    """
    hv = compute_hypervolume_2d(front, ref_point)
    total_area = ref_point[0] * ref_point[1]

    if total_area <= 0:
        return 0.0

    return hv / total_area


# ============================================================
# 4. DISTANCES / COVERAGE MAP
# ============================================================

def normalize_points(points, ref_point):
    """
    Normalisation par le point de référence.
    """
    norm = []
    for c, e in points:
        nc = c / ref_point[0] if ref_point[0] != 0 else 0.0
        ne = e / ref_point[1] if ref_point[1] != 0 else 0.0
        norm.append((nc, ne))
    return norm


def nearest_distance(point, others):
    if not others:
        return float("inf")
    return min(math.dist(point, q) for q in others)


def compute_coverage_distances(exact_front, meta_front, ref_point):
    """
    Pour chaque point exact, distance au point GA le plus proche.
    Distance calculée sur objectifs normalisés.
    """
    exact_points = extract_points(exact_front)
    meta_points = extract_points(meta_front)

    exact_norm = normalize_points(exact_points, ref_point)
    meta_norm = normalize_points(meta_points, ref_point)

    distances = []
    for p_raw, p_norm in zip(exact_points, exact_norm):
        d = nearest_distance(p_norm, meta_norm)
        distances.append((p_raw[0], p_raw[1], d))

    return distances


# ============================================================
# 5. INTERPOLATION POUR LES REGIONS MANQUEES
# ============================================================

def interpolate_front_y(points, x):
    """
    Interpolation linéaire simple sur un front trié par coût croissant.
    Retourne y(x).
    """
    if not points:
        return None

    points = sorted(points, key=lambda p: p[0])

    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        if x1 <= x <= x2:
            if x2 == x1:
                return min(y1, y2)
            alpha = (x - x1) / (x2 - x1)
            return y1 + alpha * (y2 - y1)

    return points[-1][1]


# ============================================================
# 6. GRAPHE COMBINE
# ============================================================

def plot_combined_analysis(
    exact_front,
    ga_front,
    ref_point,
    exact_time,
    ga_time,
    exact_hv,
    exact_hv_norm,
    ga_hv,
    ga_hv_norm,
    title,
    save_path=None,
    show_plot=True
):
    """
    Graphe unique combinant :
    - front exact
    - front GA
    - coverage map (points exacts colorés par distance au GA le plus proche)
    - régions manquées par le GA
    - temps + HV + HVnorm
    """

    exact_points = sorted(extract_points(exact_front), key=lambda x: x[0])
    ga_points = sorted(extract_points(ga_front), key=lambda x: x[0])

    if not exact_points or not ga_points:
        print("Impossible de tracer l'analyse combinée sans les deux fronts.")
        return

    # Distances coverage map
    coverage = compute_coverage_distances(exact_front, ga_front, ref_point)
    costs_cov = [x[0] for x in coverage]
    energies_cov = [x[1] for x in coverage]
    distances_cov = [x[2] for x in coverage]

    # Zone manquée
    x_values = sorted(set([p[0] for p in exact_points] + [p[0] for p in ga_points]))
    y_exact = [interpolate_front_y(exact_points, x) for x in x_values]
    y_ga = [interpolate_front_y(ga_points, x) for x in x_values]

    plt.figure(figsize=(11, 7))

    # Région manquée par le GA
    plt.fill_between(
        x_values,
        y_exact,
        y_ga,
        where=[yg > ye for yg, ye in zip(y_ga, y_exact)],
        alpha=0.20,
        interpolate=True,
        label="Région manquée par le GA"
    )

    # Front exact
    plt.plot(
        [p[0] for p in exact_points],
        [p[1] for p in exact_points],
        linewidth=2,
        alpha=0.9,
        label="Front exact"
    )

    # Points exacts colorés
    sc = plt.scatter(
        costs_cov,
        energies_cov,
        c=distances_cov,
        s=110,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
        label="Points exacts (distance au GA)"
    )

    # Front GA
    plt.plot(
        [p[0] for p in ga_points],
        [p[1] for p in ga_points],
        linestyle="--",
        alpha=0.8,
        label="Front GA"
    )

    plt.scatter(
        [p[0] for p in ga_points],
        [p[1] for p in ga_points],
        marker="x",
        s=90,
        linewidths=2,
        zorder=4,
        label="Points GA"
    )

    speedup = exact_time / ga_time if ga_time > 0 else float("inf")

    info_text = (
        f"Temps exact = {exact_time:.2f} s\n"
        f"Temps GA = {ga_time:.2f} s\n"
        f"Speed-up = {speedup:.2f}x\n"
        f"HV exact = {exact_hv:.4f}\n"
        f"HVnorm exact = {exact_hv_norm:.4f}\n"
        f"HV GA = {ga_hv:.4f}\n"
        f"HVnorm GA = {ga_hv_norm:.4f}"
    )

    plt.gcf().text(
        0.73, 0.53,
        info_text,
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.xlabel("Coût total")
    plt.ylabel("Consommation énergétique")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.colorbar(sc, label="Distance normalisée au point GA le plus proche")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Graphe combiné sauvegardé : {save_path}")

    if show_plot:
        plt.show()

    plt.close()


# ============================================================
# 7. EXPORT CSV
# ============================================================

def save_summary_csv(filepath, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Résumé CSV sauvegardé : {filepath}")


# ============================================================
# 8. COMPARAISON COMPLETE SUR UNE INSTANCE
# ============================================================

def compare_instance(
    instance_path,
    energy_scale=100,
    exact_time_limit=240,
    exact_mip_gap=0.0,
    exact_threads=0,
    ga_pop_size=80,
    ga_generations=200,
    ga_crossover_rate=0.9,
    ga_mutation_rate=0.35,
    ga_seed=42,
    show_plots=True,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    instance_name = os.path.splitext(os.path.basename(instance_path))[0]

    print("=" * 80)
    print(f"INSTANCE : {instance_name}")
    print("=" * 80)

    # --------------------------------------------------------
    # Lecture de l'instance
    # --------------------------------------------------------
    data = read_instance(instance_path, energy_scale=energy_scale)
    ref_point = compute_reference_point(data)

    print(f"Point de référence : {ref_point}")

    # --------------------------------------------------------
    # Exact
    # --------------------------------------------------------
    t0 = time.time()
    exact_front, exact_bounds = exact_pareto_front_epsilon_constraint(
        data,
        time_limit=exact_time_limit,
        mip_gap=exact_mip_gap,
        threads=exact_threads
    )
    exact_time = time.time() - t0

    exact_hv = compute_hypervolume_2d(exact_front, ref_point)
    exact_hv_norm = compute_hv_norm(exact_front, ref_point)

    print(f"Temps exact : {exact_time:.2f} s")
    print(f"HV exact : {exact_hv:.6f}")
    print(f"HVnorm exact : {exact_hv_norm:.6f}")

    # --------------------------------------------------------
    # GA avec budget = moitié du temps exact
    # --------------------------------------------------------
    ga_time_budget = exact_time / 2.0

    t1 = time.time()
    ga_front, final_pop = run_nsga2_calbp(
        data,
        pop_size=ga_pop_size,
        generations=ga_generations,
        crossover_rate=ga_crossover_rate,
        mutation_rate=ga_mutation_rate,
        seed=ga_seed,
        verbose=True,
        time_limit_seconds=ga_time_budget
    )
    ga_time = time.time() - t1

    ga_hv = compute_hypervolume_2d(ga_front, ref_point)
    ga_hv_norm = compute_hv_norm(ga_front, ref_point)

    print(f"Budget GA : {ga_time_budget:.2f} s")
    print(f"Temps GA : {ga_time:.2f} s")
    print(f"HV GA : {ga_hv:.6f}")
    print(f"HVnorm GA : {ga_hv_norm:.6f}")

    # --------------------------------------------------------
    # Graphe combiné
    # --------------------------------------------------------
    combined_plot = os.path.join(base_dir, f"combined_analysis_{instance_name}.png")

    plot_combined_analysis(
        exact_front,
        ga_front,
        ref_point,
        exact_time=exact_time,
        ga_time=ga_time,
        exact_hv=exact_hv,
        exact_hv_norm=exact_hv_norm,
        ga_hv=ga_hv,
        ga_hv_norm=ga_hv_norm,
        title=f"Analyse combinée Exact vs GA - {instance_name}",
        save_path=combined_plot,
        show_plot=show_plots
    )

    # --------------------------------------------------------
    # Résumé
    # --------------------------------------------------------
    summary = {
        "instance": instance_name,
        "exact_time_sec": round(exact_time, 6),
        "ga_budget_sec": round(ga_time_budget, 6),
        "ga_time_sec": round(ga_time, 6),
        "n_exact_points": len(exact_front),
        "n_ga_points": len(ga_front),
        "hv_exact": exact_hv,
        "hvnorm_exact": exact_hv_norm,
        "hv_ga": ga_hv,
        "hvnorm_ga": ga_hv_norm,
        "ref_cost": ref_point[0],
        "ref_energy": ref_point[1],
    }

    return summary


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Mets ici l'instance à comparer
    instance_path = os.path.join(base_dir, "instances", "CALBP_OTTO_Roszieg_n25.txt")

    summary = compare_instance(
        instance_path=instance_path,
        energy_scale=100,
        exact_time_limit=240,
        exact_mip_gap=0.0,
        exact_threads=0,
        ga_pop_size=80,
        ga_generations=200,
        ga_crossover_rate=0.9,
        ga_mutation_rate=0.35,
        ga_seed=42,
        show_plots=True,
    )

    summary_csv = os.path.join(base_dir, "comparison_summary.csv")
    save_summary_csv(summary_csv, [summary])

    print("\nRésumé final :")
    for k, v in summary.items():
        print(f"{k}: {v}")
