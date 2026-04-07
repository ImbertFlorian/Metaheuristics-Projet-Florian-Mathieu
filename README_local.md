# Contenu du dépôt

Ce dépôt est organisé autour de trois programmes principaux :

    local.py : Le programme principal contenant l'algorithme complet de recherche locale. Il génère et évalue le front de Pareto, calcule l'Hypervolume et trace le graphique final.

    tuning_local.py : Le programme d'étude de sensibilité (Grid Search) permettant de justifier mathématiquement les paramètres clés de la métaheuristique.

    HV_local.py : Un programme qui évalue les performances via l'Hyper volume normalisé. De plus, le programme va plot le front métaheuristique et le front exact pour les comparer. 

# Algorithme : Multi-Start Pareto Local Search (MS-PLS)

L'algorithme local.py utilise une structure de recherche locale adaptée à l'optimisation multi-objectifs, renforcée par une mécanique Multi-Start pour éviter la stagnation dans des optimums locaux.
## 1. Fonctionnement général

Initialisation : L'archive de départ est remplie avec des solutions générées par des heuristiques constructives extrêmes (optimisation pure Coût / optimisation pure Énergie) ainsi que N solutions générées aléatoirement mais respectant les contraintes de précédence.

Voisinage : À chaque itération, l'algorithme explore le voisinage d'une solution non dominée en :

- Modifiant le mode d'exécution d'une tâche (Humain, Cobot ou Hybride).

- Échangeant la position de deux tâches adjacentes dans la séquence (si les contraintes de précédence le permettent).

Mise à jour de l'archive : Si un voisin n'est dominé par aucune solution actuelle, il intègre l'archive et filtre les anciennes solutions potentiellement dominées.

Multi-Start (Restart) : Si l'algorithme converge et explore entièrement l'archive avant la fin du temps limite, il génère une nouvelle solution aléatoire pour "sauter" dans une zone inexplorée de l'espace de recherche.

## 2. Évaluation de la performance (Hypervolume)

La qualité du front de Pareto obtenu est mesurée via l'indicateur Hypervolume (HV). Le point de référence (τ) est calculé automatiquement en fonction des pires valeurs théoriques possibles de l'instance traitée :
τ=(∣K∣⋅(Cs​+Cw​+Cc​)+1 , ∑max(Ejm​)+Re​∣K∣T+1)

# Étude de Sensibilité (Parameter Tuning)

Le script Parameter_Tuning.py a été développé pour respecter une démarche scientifique stricte. Il effectue une "Recherche sur Grille" (Grid Search) pour calibrer les deux paramètres clés du MS-PLS :

    nb_initial_sols : La taille de la population initiale (Diversification spatiale).

    nb_swaps : Le degré de perturbation appliqué pour générer des solutions aléatoires lors de l'initialisation et des Restarts (Force d'échappement).

L'algorithme teste de multiples configurations (par exemple 5×5=25 scénarios) sur plusieurs runs indépendants. La configuration conservée pour les tests finaux est celle maximisant l'Hypervolume moyen, offrant le meilleur compromis entre Intensification et Diversification.
