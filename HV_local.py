import matplotlib.pyplot as plt

# ============================================================
# CALCULATEUR D'HYPERVOLUME NORMALISÉ (OUTIL POUR LE RAPPORT)
# ============================================================

def calculate_hypervolume(points, ref_point):
    """Calcule l'Hypervolume à partir d'une liste de tuples (cost, energy)"""
    if not points:
        return 0.0
        
    # 1. On trie les points du moins cher au plus cher
    points.sort(key=lambda x: x[0]) 
    
    ref_c, ref_e = ref_point
    hv = 0.0
    current_ref_e = ref_e
    
    # 2. On additionne l'aire des rectangles
    for cost, energy in points:
        # On ignore les points qui sont pires que le point de référence
        if cost > ref_c or energy > ref_e:
            continue
            
        width = ref_c - cost
        height = current_ref_e - energy
        hv += width * height
        current_ref_e = energy
        
    return hv

if __name__ == "__main__":
    # 1. point de référence tau
    tau = (316.0, 13.11) 

    # 2. POINTS EXACTS du epsilon-contrainte
    front_exact = [
        (200.00, 0.00),
        (125.00, 0.57),
        (94.00, 1.88),
        (67.00, 3.84),
        (45.00, 6.11)
    ]

    # 3.POINTS MÉTAHEURISTIQUE 
    front_meta = [
        (200.00, 0.00),
        (125.00, 0.57),
        (94.00, 1.88),
        (67.00, 3.84),
        (45.00, 6.11)
    ]

    #  4. CALCULS
    hv_exact = calculate_hypervolume(front_exact, tau)
    hv_meta = calculate_hypervolume(front_meta, tau)
    
    if hv_exact > 0:
        hv_norm = (hv_meta / hv_exact) * 100
    else:
        hv_norm = 0.0

    #  5. AFFICHAGE 
    print(f"--- RÉSULTATS POUR L'INSTANCE BOWMAN ---")
    print(f"Point de référence tau : {tau}")
    print(f"Hypervolume Exact (HV_exact) : {hv_exact:.2f}")
    print(f"Hypervolume Méta  (HV_meta)  : {hv_meta:.2f}")
    print("-" * 40)
    print(f"HV_norm = {hv_norm:.2f} %")
    print("-" * 40)

    # ============================================================
    # 6. CRÉATION DU GRAPHIQUE COMPARATIF
    # ============================================================
    
    # Extraction des coordonnées X (coût) et Y (énergie)
    c_exact = [p[0] for p in front_exact]
    e_exact = [p[1] for p in front_exact]
    
    c_meta = [p[0] for p in front_meta]
    e_meta = [p[1] for p in front_meta]

    plt.figure(figsize=(10, 6))

    # Tracé du front Exact (Ligne bleue épaisse)
    plt.plot(c_exact, e_exact, marker='s', markersize=8, linestyle='-', color='#1f77b4', linewidth=4, alpha=0.6, label='Front Exact (DOcplex)')
    
    # Tracé du front Métaheuristique (Ligne rouge pointillée par-dessus)
    plt.plot(c_meta, e_meta, marker='o', markersize=6, linestyle='--', color='#d62728', linewidth=2, label='Front Métaheuristique (PLS)')
    
    # Point de référence Tau (Étoile noire)
    plt.plot(tau[0], tau[1], marker='*', markersize=15, color='black', label=rf'Point de référence $\tau$ {tau}')

    # Annotation des points
    for c, e in front_meta:
        plt.annotate(f"({c:.0f}, {e:.2f})", (c, e), textcoords="offset points", xytext=(10, 10), ha='left', fontsize=9)

    # Mise en forme du graphique
    plt.xlabel("Coût d'exploitation ($z_c$)", fontsize=12)
    plt.ylabel("Consommation énergétique ($z_e$)", fontsize=12)
    plt.title(f"Comparaison Exact vs Métaheuristique - Instance Bowman\n$HV_{{norm}}$ = {hv_norm:.2f} %", fontsize=14, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    # Sauvegarde et affichage
    nom_image = "comparaison_bowman.png"
    plt.savefig(nom_image, dpi=300)
    print(f"Graphique sauvegardé avec succès sous le nom : {nom_image}")
    plt.show()
