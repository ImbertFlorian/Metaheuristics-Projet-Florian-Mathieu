# Local Search

Le local search est organisé autour de trois programmes principaux :

    local.py : Le programme principal contenant l'algorithme complet de recherche locale. Il génère et évalue le front de Pareto, calcule l'Hypervolume et trace le graphique final.

    tuning_local.py : Le programme d'étude de sensibilité (Grid Search) permettant de justifier mathématiquement les paramètres clés de la métaheuristique.

    HV_local.py : Un programme qui évalue les performances via l'Hyper volume normalisé. De plus, le programme va plot le front métaheuristique et le front exact pour la comparaison. 

# Algorithme : Multi-Start Pareto Local Search (MS-PLS)

L'algorithme local.py utilise une structure de recherche locale adaptée à l'optimisation multi-objectifs, renforcée par une mécanique Multi-Start pour éviter la stagnation dans des optimums locaux.
## 1. Fonctionnement général

Initialisation : L'archive de départ est remplie avec des solutions générées par des heuristiques constructives extrêmes (optimisation pure Coût / optimisation pure Énergie) ainsi que N solutions générées aléatoirement mais respectant les contraintes de précédence.

Voisinage : À chaque itération, l'algorithme explore le voisinage d'une solution non dominée en :

- Modifiant le mode d'exécution d'une tâche (Humain, Cobot ou Hybride).

- Échangeant la position de deux tâches adjacentes dans la séquence (si les contraintes de précédence le permettent).

Mise à jour de l'archive : Si un voisin n'est dominé par aucune solution actuelle, il intègre l'archive et filtre les anciennes solutions potentiellement dominées.

Multi-Start (Restart) : Si l'algorithme converge et explore entièrement l'archive courante avant la fin de son temps limite (ex: avant les 60 secondes allouées), il ne s'arrête pas. Il génère une nouvelle solution totalement aléatoire

Afin de pallier la nature stochastique de notre métaheuristique et de garantir la robustesse de nos résultats, l'évaluation n'est pas basée sur une seule exécution :

- Exécutions multiples (Runs) : Pour chaque instance, nous lançons l'algorithme sur 10 runs indépendants dotés chacun d'un budget temps strict (ex: 60 secondes par run). Cela permet de lisser le facteur chance.
- Fusion et Front Global : À l'issue de ces 10 exécutions, l'ensemble des fronts de Pareto locaux trouvés par chaque run est réuni. Un ultime filtrage de dominance est appliqué à cette "super-archive" pour extraire le Front de Pareto Global. C'est la synergie de ces différentes explorations indépendantes qui est finalement conservée pour représenter la performance de notre méthode.

## 2. Évaluation de la performance (Hypervolume)

La qualité du front de Pareto obtenu est mesurée via l'indicateur Hypervolume (HV). Le point de référence (τ) est calculé automatiquement en fonction des pires valeurs théoriques possibles de l'instance traitée :
τ=(∣K∣⋅(Cs​+Cw​+Cc​)+1 , ∑max(Ejm​)+Re​∣K∣T+1)

# Étude de Sensibilité (Parameter Tuning)

Le script Parameter_Tuning.py a été développé pour respecter une démarche scientifique stricte. Il effectue une "Recherche sur Grille" (Grid Search) pour calibrer les deux paramètres clés du MS-PLS :

- nb_initial_sols : La taille de la population initiale (Diversification spatiale). Correspond à la quantité de configurations d'usines générées aléatoirement au tout début de l'algorithme pour peupler l'archive de départ (en complément des deux heuristiques extrêmes). Concrètement, cela revient à placer plusieurs "explorateurs" à des points de départ très éloignés les uns des autres sur la carte des solutions possibles. Cela permet de ratisser large dès les premières secondes et d'éviter que l'algorithme ne s'enferme immédiatement dans une seule stratégie d'assemblage

- nb_swaps : La force de perturbation (Échappement). Correspond au nombre d'échanges aléatoires (permutations de deux tâches adjacentes) réalisés pour créer une nouvelle solution de départ ou de "Restart", tout en respectant les contraintes de précédences.

L'algorithme teste de multiples configurations (par exemple 5×5=25 scénarios) sur plusieurs runs indépendants. La configuration conservée pour les tests finaux est celle maximisant l'Hypervolume moyen, offrant le meilleur compromis entre Intensification et Diversification.
