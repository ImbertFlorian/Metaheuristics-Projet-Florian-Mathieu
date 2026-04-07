# Metaheuristics-Projet-Florian-Matthieu
Projet de Métaheuristics de Florian IMBERT &amp; Matthieu TEULON

# genetic.py
## Description du projet (Non-dominated Sorting Genetic Algorithm II) (NSGA-II)

Le code inclut :

- (1)
- une structure `Individual` pour représenter une solution
- la gestion des **contraintes de précédence**
- la **réparation** pour une solution qui respecte les précédences (permutation topologique)
- la **réparation** des modes non faisables
- un **décodeur** qui transforme `(permutation + modes)` en solution complète
- l’évaluation multi-objectif :
  - coût
  - énergie
    
- (2)
- les composants principaux de **NSGA-II** :
  - dominance
  - tri non dominé rapide
  - crowding distance (pour avoir des points espacés sur le front)
  - sélection
- des opérateurs génétiques :
  - crossover d’ordre sur les permutations
  - crossover uniforme sur les modes
  - mutation de permutation
  - mutation des modes
- des heuristiques constructives initiales

- (3)
- le calcul de :
  - **front de Pareto**
  - **hypervolume**
  - **hypervolume normalisé**
- la visualisation graphique du front de Pareto

---

## Structure générale du code

Le fichier est organisé en plusieurs sections :

### 1. Structures
Définition de la classe `Individual` avec :

- `perm` : ordre des tâches
- `modes` : mode sélectionné pour chaque tâche
- `cost`, `energy`, `energy_raw`
- `rank`, `crowding`
- `assignment` : affectation tâche → station
- `stations` : description détaillée des stations
- `feasible` : validité de la solution

### 2. Outils de précédence
Fonctions utilitaires pour :

- construire les prédécesseurs/successeurs
- générer un ordre topologique aléatoire
- réparer une permutation invalide

### 3. Modes faisables / réparation
Fonctions permettant :

- de récupérer les modes compatibles avec le temps de cycle
- de remplacer un mode non faisable par un mode valide

### 4. Décodage et évaluation
Transformation d’un individu en solution exploitable :

- affectation aux stations
- calcul du coût
- calcul de l’énergie
- gestion de l’infaisabilité

### 5. Dominance / NSGA-II
Implémentation de :

- la dominance de Pareto
- `fast_non_dominated_sort`
- `crowding_distance`
- `nsga2_select`
- `tournament_selection`

### 6. Opérateurs génétiques
Implémentation des opérateurs :

- `order_crossover`
- `uniform_mode_crossover`
- `mutate_permutation`
- `mutate_modes`

### 7. Heuristiques constructives initiales
Création de solutions initiales orientées :

- **coût**
- **énergie**
- ou complètement aléatoires

### 8. Algorithme principal
La fonction principale est :

`python
run_nsga2_calbp(...)`

### 9. Algorithme principal
