# 🐜 Optimisation par Colonie de Fourmis (Ant Colony Optimization – ACO)
### Modélisation Mathématique, Simulation Interactive et Résolution du Problème du Voyageur de Commerce (TSP)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly Interactive](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![Notebook Interactif](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/ant_colony_optimization.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Présentation et Objectifs

Ce projet propose une implémentation **pédagogique, épurée et interactive** de l'algorithme d'**Optimisation par Colonie de Fourmis (ACO)** appliquée au **Problème du Voyageur de Commerce (TSP)**.  
L'architecture a été conçue pour être **intégralement présentée et démontrée en 15 minutes**, tout en conservant la rigueur mathématique et expérimentale de la métaheuristique originale de **Marco Dorigo (1991, 1996)**.

### Points clés du projet :
* **1 Moteur mathématique unifié (`src/core_aco.py`) :** Calcul vectorisé NumPy des distances euclidiennes, règle de transition stochastique, évaporation, dépôt de phéromones (standard AS et élitiste EAS) et solveurs de référence (Plus Proche Voisin Glouton et Brute Force Exact).
* **1 Interface Streamlit mono-page à 3 onglets (`app.py`) :** Démonstration en direct, explication des équations avec calculatrice de probabilités, et banc d'essai comparatif.
* **1 Notebook Jupyter complet et autonome (`notebooks/ant_colony_optimization.ipynb`) :** Présentation narrative, calculs détaillés pas-à-pas, étude de sensibilité des paramètres et graphiques haute résolution.

---

## 🌿 1. Inspiration Biologique : La Stigmergie

Dans la nature, les fourmis trouvent collectivement le plus court chemin entre leur nid et une source de nourriture sans vision globale ni coordination centralisée. Ce phénomène repose sur la **stigmergie** (Pierre-Paul Grassé, 1959) et l'expérience du **Double Pont** (Goss et al., 1989) :
1. En se déplaçant, chaque fourmi dépose sur le sol une trace chimique volatile appelée **phéromone**.
2. Sur un chemin plus court, l'aller-retour est plus rapide : la densité de phéromone y augmente plus vite que sur un chemin long (*boucle d'auto-renforcement positif*).
3. Les fourmis suivantes sont attirées avec une plus forte probabilité par les pistes les plus marquées.
4. L'**évaporation naturelle** atténue les pistes anciennes ou longues, évitant que la colonie ne reste figée dans une solution sous-optimale.

---

## 📐 2. Modélisation Mathématique Formelle

### 2.1 Le Problème du Voyageur de Commerce (TSP)
Soit un ensemble de $n$ villes $V = \{0, 1, \dots, n-1\}$ de coordonnées $(x_i, y_i) \in \mathbb{R}^2$ formant un graphe complet symétrique $G = (V, E)$.
* **Distance euclidienne :**
  $$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$
* **Objectif :** Trouver un cycle hamiltonien $\pi = (\pi_0, \pi_1, \dots, \pi_{n-1})$ minimisant la distance totale fermée :
  $$\min_{\pi} L(\pi) = \sum_{i=0}^{n-2} d_{\pi_i, \pi_{i+1}} + d_{\pi_{n-1}, \pi_0}$$
* **Explosion combinatoire :** Pour un graphe symétrique, le nombre de cycles distincts vaut $\Omega = \frac{(n-1)!}{2}$ (soit $12$ cycles pour $5$ villes, $181\,440$ pour $10$ villes, et $6.08 \times 10^{16}$ pour $20$ villes).

### 2.2 Les 3 Équations Maîtresses d'Ant System (Dorigo, 1996)

1. **Visibilité Heuristique Locale (attrait glouton immédiat) :**
   $$\eta_{ij} = \frac{1}{d_{ij}} \quad (\forall i \neq j)$$

2. **Règle Proportionnelle de Transition (Roulette Probabiliste) :**
   Lorsqu'une fourmi $k$ se trouve sur la ville $i$, la probabilité de choisir la ville non visitée $j \in N_i^k$ est :
   $$P_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in N_i^k} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta}$$
   * **$\alpha \ge 0$** : Poids de la trace de phéromone $\tau_{ij}$ (*exploitation de la mémoire collective*).
   * **$\beta \ge 0$** : Poids de la visibilité heuristique $\eta_{ij}$ (*exploration gloutonne locale*).

3. **Évaporation et Dépôt Global de Phéromones :**
   À la fin de chaque itération, après le passage des $m$ fourmis :
   $$\tau_{ij} \leftarrow (1 - \rho)\,\tau_{ij} + \sum_{k=1}^m \Delta \tau_{ij}^k \quad \text{avec} \quad \Delta \tau_{ij}^k = \begin{cases} \dfrac{Q}{L_k} & \text{si } (i, j) \in \text{Tour}_k \\ 0 & \text{sinon} \end{cases}$$
   * **$\rho \in ]0, 1[$** : Taux d'évaporation (oubli progressif des mauvaises pistes).
   * **$Q > 0$** : Constante de dépôt, récompensant les tournées courtes ($L_k$ faible).
   * **Option Élitiste (EAS) :** Ajoute un bonus $e \cdot \frac{Q}{L_{\text{best}}}$ sur les arêtes du meilleur chemin global.

### 2.3 Rôle et Réglage des Hyperparamètres

| Paramètre | Symbole | Valeur Type | Rôle et Impact sur la Colonie |
| :--- | :---: | :---: | :--- |
| **Poids des phéromones** | $\alpha$ | $1.0$ | Contrôle l'**exploitation** de la mémoire collective. Si $\alpha = 0$, l'algorithme ignore les phéromones. |
| **Poids de la visibilité** | $\beta$ | $2.0$ à $5.0$ | Contrôle l'**attrait glouton** des villes proches. Si $\beta = 0$, les fourmis ignorent les distances. |
| **Taux d'évaporation** | $\rho$ | $0.3$ à $0.5$ | Contrôle la **vitesse d'oubli**. Évite la stagnation prématurée dans des minima locaux. |
| **Taille de la colonie** | $m$ | $15$ à $30$ | Nombre de fourmis par génération. Augmente la diversité d'exploration. |
| **Poids élitiste (EAS)** | $e$ | $0$ ou $5.0$ | Renforce la meilleure tournée globale pour accélérer la convergence finale. |

---

## 🧮 3. Exemple Numérique Complet Pas-à-Pas (Instance 5 Villes)

Considérons l'instance didactique `Micro_5_Villes` composée d'un carré de côté $60$ avec une ville au centre :
* **Ville 0 :** $(20, 20)$ — Coin Sud-Ouest
* **Ville 1 :** $(20, 80)$ — Coin Nord-Ouest
* **Ville 2 :** $(80, 80)$ — Coin Nord-Est
* **Ville 3 :** $(80, 20)$ — Coin Sud-Est
* **Ville 4 :** $(50, 50)$ — Centre du carré

### Étape 1 : Calcul des distances depuis la Ville 0
* $d_{01} = \sqrt{(20-20)^2 + (80-20)^2} = 60.00$
* $d_{02} = \sqrt{(80-20)^2 + (80-20)^2} = \sqrt{7200} \approx 84.85$ (diagonale)
* $d_{03} = \sqrt{(80-20)^2 + (20-20)^2} = 60.00$
* $d_{04} = \sqrt{(50-20)^2 + (50-20)^2} = \sqrt{1800} \approx 42.43$ (demi-diagonale vers le centre)

### Étape 2 : Calcul des probabilités de transition depuis la Ville 0
Fixons $\tau_0 = 1.0$ partout, $\alpha = 1.0$ et $\beta = 3.0$. Pour une fourmi partant de la **Ville 0**, les villes candidates sont $N_0 = \{1, 2, 3, 4\}$ :

| Ville Candidate $j$ | Distance $d_{0j}$ | Visibilité $\eta_{0j} = 1/d_{0j}$ | Poids $w_{0j} = \tau_{0j}^1 \cdot \eta_{0j}^3$ | Probabilité $P_{0j} = w_{0j} / \sum w$ |
| :---: | :---: | :---: | :---: | :---: |
| **Ville 1** | $60.00$ | $0.0167$ | $4.63 \times 10^{-6}$ | **19.30 %** |
| **Ville 2** | $84.85$ | $0.0118$ | $1.64 \times 10^{-6}$ | **6.82 %** |
| **Ville 3** | $60.00$ | $0.0167$ | $4.63 \times 10^{-6}$ | **19.30 %** |
| **Ville 4 (Centre)** | $42.43$ | $0.0236$ | $13.09 \times 10^{-6}$ | **54.58 %** |
| **Total** | — | — | $\sum w = 23.99 \times 10^{-6}$ | **100.00 %** |

**Interprétation :** La Ville 4 étant la plus proche ($42.43$), elle possède $54.58\,\%$ de chances d'être choisie au premier pas, mais les Villes 1 et 3 conservent chacune $19.30\,\%$ de chances d'être explorées grâce au tirage par roulette.

### Étape 3 : Mise à jour des phéromones après une tournée optimale
Le cycle optimal consiste à parcourir trois côtés du carré et à visiter le centre entre deux coins adjacents ($0 \to 1 \to 2 \to 3 \to 4 \to 0$) :
$$L^* = 60 + 60 + 60 + 42.4264 + 42.4264 = \mathbf{264.8528}$$
Avec $Q = 100$ et $\rho = 0.5$, une fourmi ayant emprunté ce cycle dépose sur chacune de ses arêtes :
$$\Delta \tau = \frac{100}{264.8528} \approx 0.3776 \implies \tau_{\text{nouveau}} = (1 - 0.5) \times 1.0 + 0.3776 = \mathbf{0.8776}$$
tandis qu'une arête non empruntée (comme la grande diagonale $(0, 2)$) subit uniquement l'évaporation et tombe à $\tau_{02} = 0.5000$.

---

## 📊 4. Résultats Expérimentaux & Benchmarks

Comparaison mesurée sur les 3 instances étalonnées intégrées au projet :

| Instance TSP | Villes ($n$) | Algorithme | Distance Obtenue | Écart / Optimum | Temps | Conclusion |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| **Micro Instance** | 5 | **Brute Force Exact** | **264.85** | **0.00 %** | < 0.1 ms | Optimum absolu certifié (12 cycles) |
| | 5 | **Ant System (ACO)** | **264.85** | **0.00 %** | ~15 ms | Convergence exacte vers l'optimum |
| | 5 | **Plus Proche Voisin (Glouton)** | 264.85 | 0.00 % | < 0.1 ms | Suffisant sur micro-instance |
| **Petite Instance** | 10 | **Brute Force Exact** | **240.90** | **0.00 %** | ~350 ms | 181 440 permutations évaluées |
| | 10 | **Ant System (ACO)** | **240.90** | **0.00 %** | ~30 ms | **10× plus rapide** que Force Brute |
| **Moyenne Instance** | 20 | **Ant System (ACO)** | **374.51** | **Meilleur résultat** | ~70 ms | **Surpasse le Glouton de +4.04 %** |
| | 20 | **Elitist Ant System (EAS)** | **374.51** | **Meilleur résultat** | ~70 ms | Convergence accélérée par l'élite |
| | 20 | **Plus Proche Voisin (Glouton)** | 390.29 | +4.21 % | < 0.2 ms | Piégé par des choix myopes en fin de tour |

---

## 🏗️ 5. Architecture Logicielle & Complexité

### 5.1 Arborescence Simplifiée (3 Fichiers Python)
```text
ant_colony_optimization/
├── app.py                                    # Application Streamlit mono-page à 3 onglets
├── README.md                                 # Documentation complète du projet (Théorie & Architecture)
├── requirements.txt                          # Dépendances Python (numpy, pandas, plotly, streamlit, matplotlib)
├── src/
│   ├── __init__.py
│   ├── core_aco.py                           # Moteur unifié : TSPInstance, AntSystem, Glouton, Brute Force
│   └── visualization.py                      # Graphiques interactifs Plotly (Carte TSP & Convergence)
└── notebooks/
    └── ant_colony_optimization.ipynb         # Présentation interactive complète, autonome et illustrée
```

### 5.2 Contrats de Données (`src/core_aco.py`)
```python
@dataclass(frozen=True)
class City:
    """Sommet 2D représentant une ville."""
    id: int
    name: str
    x: float
    y: float

@dataclass
class TSPInstance:
    """Instance du TSP avec matrices pré-calculées."""
    name: str
    cities: List[City]
    distance_matrix: np.ndarray     # d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
    heuristic_matrix: np.ndarray    # eta_ij = 1 / d_ij (0 sur la diagonale)
    optimal_known: Optional[float] = None

@dataclass
class ACOResult:
    """Résultat d'une simulation Ant System."""
    best_path: List[int]            # Meilleur cycle hamiltonien trouvé
    best_length: float              # Distance minimale associée
    history_best: List[float]       # Évolution de la meilleure distance par itération
    history_mean: List[float]       # Évolution de la distance moyenne de la colonie
    pheromone_matrix: np.ndarray    # Matrice finale des phéromones tau_ij
    execution_time_ms: float        # Temps d'exécution en millisecondes
```

### 5.3 Organisation des 3 Onglets Streamlit (`app.py`)
| Onglet | Objectif Pédagogique | Contenu Visuel |
| :--- | :--- | :--- |
| **1. Démonstration & Simulation TSP** | Montrer la résolution en direct et le gain face au Glouton | 4 KPIs + Carte interactive des phéromones + Courbe de convergence |
| **2. Principe & Modèle Mathématique** | Expliquer la théorie et vérifier un calcul de probabilité | 3 formules LaTeX + Tableau numérique en direct des probabilités depuis la Ville 0 |
| **3. Comparaison & Applications** | Valider la performance face aux méthodes classiques | Tableau comparatif (Glouton, AS, EAS, Brute Force) + Cas d'usage industriels |

### 5.4 Analyse de Complexité Algorithmique
Pour une instance de $n$ villes, une colonie de $m$ fourmis et $T$ itérations :
* **Pré-calcul des matrices $D$ et $\eta$ :** $\mathcal{O}(n^2)$ en temps et en mémoire (vectorisé via broadcasting NumPy).
* **Construction d'une tournée par une fourmi :** $\mathcal{O}(n^2)$ (choix successif parmi $n-1, n-2, \dots, 1$ villes candidates).
* **Complexité temporelle totale d'Ant System :**
  $$\mathcal{O}(T \cdot m \cdot n^2)$$
  Ce coût **polynomial** permet de résoudre en quelques dizaines de millisecondes des instances où la recherche exacte en $\mathcal{O}(n!)$ est totalement impraticable.

---

## 🚀 6. Installation et Exécution

### 6.1 Installation des dépendances
```bash
pip install -r requirements.txt
```

### 6.2 Lancement de l'Application Interactive Streamlit
```bash
streamlit run app.py
```

---

## 🌐 7. Cas d'Usage Industriels

* **🚚 Logistique & Tournées de Livraison (VRP) :** Réduction de 15 % à 25 % des distances parcourues par les flottes de véhicules.
* **📡 Routage Réseau Dynamique (AntNet) :** Adaptation décentralisée du routage de paquets IP face aux congestions réseau.
* **⚡ Perçage de Circuits Imprimés (PCB Drilling) :** Optimisation de la trajectoire laser sur des milliers de points de perçage.
* **⏱️ Ordonnancement d'Ateliers (Job-Shop Scheduling) :** Minimisation du temps total de production (*Makespan*).

---

## 📄 Licence

Ce projet est distribué sous licence **MIT** à des fins pédagogiques, académiques et scientifiques.
