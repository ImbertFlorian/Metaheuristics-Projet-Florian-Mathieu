# Metaheuristics-Projet-Florian-Matthieu
Projet de Métaheuristics de Florian IMBERT &amp; Matthieu TEULON

# e_constraint.py
## Description du projet
Le script permet de :

1. **lire une instance texte** décrivant les tâches, stations, modes, temps, énergies, précédences et paramètres de coût ;
2. **construire un modèle MILP** du problème CALBP ;
3. **résoudre deux problèmes extrêmes** :
   - minimisation de l'énergie ;
   - minimisation du coût ;
4. **générer le front de Pareto exact** via une approche epsilon-constraint ;
5. **afficher les solutions** non dominées ;
6. **tracer et sauvegarder** le front de Pareto sous forme de graphique.

## Structure du code
Le script est organisé en 10 grandes parties :

### 1. Lecture d'instance
Fonction principale :
- `read_instance(filepath, energy_scale=100)`

Elle lit le fichier texte et retourne un dictionnaire `data` contenant toutes les données utiles au modèle.

### 2. Construction du modèle MILP
Fonction principale :
- `build_calbp_model(data, epsilon=None, primary="cost")`

Elle construit le modèle DOcplex avec :
- variables de décision,
- fonction objectif,
- contraintes.

### 3. Paramétrage solveur
Fonction :
- `configure_solver(mdl, time_limit=240, mip_gap=0.0, threads=0)`

Permet de régler :
- la limite de temps,
- le gap MIP,
- le nombre de threads.

### 4. Extraction de solution
Fonction :
- `extract_solution(data, mdl, x, o, w, y, l, z_c, z_e)`

Transforme la solution DOcplex en dictionnaire Python lisible.

### 5. Calcul des bornes
Fonction :
- `compute_energy_bounds(data, time_limit=None, mip_gap=0.0, threads=0)`

Calcule :
- l'énergie minimale,
- l'énergie associée à la solution de coût minimal,
- le coût minimal,
- le coût associé à l'énergie minimale.

### 6. Filtrage non dominé
Fonctions :
- `is_dominated(sol, others)`
- `filter_nondominated(solutions)`

Supprime les doublons et les solutions dominées.

### 7. Génération du front de Pareto
Fonction :
- `exact_pareto_front_epsilon_constraint(data, time_limit=None, mip_gap=0.0, threads=0)`

Applique la méthode epsilon-constraint exacte.

### 8. Affichage texte
Fonctions :
- `print_bounds(bounds, scale=1)`
- `print_pareto_front(pareto)`

Affiche les bornes et les solutions du front.

### 9. Tracé graphique
Fonction :
- `plot_pareto_front(pareto, title="Front de Pareto", save_path=None, show_plot=False)`

Trace et sauvegarde le front de Pareto.

### 10. Main
Bloc principal :
- lecture de l'instance ;
- résolution ;
- affichage ;
- sauvegarde du graphique.

