# Traité Théorique et Mathématique de l'Optimisation par Colonie de Fourmis (ACO)
## Application au Problème du Voyageur de Commerce (Traveling Salesperson Problem – TSP)
**Auteur :** El Kachabi Badr-eddine  
**Profil GitHub :** [https://github.com/el-kachabi](https://github.com/el-kachabi)  
**Dépôt :** [ant_colony_optimization](https://github.com/el-kachabi/ant_colony_optimization)  

---

## 1. Introduction et Historique des Algorithmes Bio-Inspirés

L'**Optimisation par Colonie de Fourmis** (*Ant Colony Optimization* – ACO) est une métaheuristique bio-inspirée introduite au début des années 1990 par **Marco Dorigo** dans sa thèse de doctorat (1992), puis formalisée dans des publications majeures avec V. Maniezzo et A. Colorni (1996).

Elle s'inspire du comportement d'auto-organisation observé chez les fourmis réelles, en particulier de leur capacité collective à identifier le chemin le plus court reliant leur nid à des sources de nourriture sans aucune autorité centrale.

### 1.1 Le Concept Fondateur de Stigmergie
Le biologiste français **Pierre-Paul Grassé** a défini en 1959 le concept de **stigmergie** (du grec *stigma* : piqûre/marque, et *ergon* : travail) pour décrire une communication indirecte et asynchrone médiée par des modifications physiques de l'environnement :
$$\text{Action d'un agent} \longrightarrow \text{Modification de l'environnement} \longrightarrow \text{Stimulation d'une action chez d'autres agents}$$

Chez les fourmis, cette modification de l'environnement s'opère par le dépôt sur le sol d'une substance chimique volatile appelée **phéromone de piste**.

### 1.2 L'Expérience du Double Pont (Goss et al., 1989)
Dans une expérience de laboratoire désormais célèbre, des éthologues ont connecté un nid de fourmis d'Argentine (*Linepithema humile*) à une source de nourriture via deux branches de longueurs inégales :
- Une branche courte $A$ de longueur $L_A$.
- Une branche longue $B$ de longueur $L_B = 2 \cdot L_A$.

#### Déroulement de la dynamique collective :
1. **Étape initiale ($t = 0$) :** En l'absence de traces préexistantes, les premières fourmis choisissent aléatoirement chaque branche avec une probabilité égale ($P_A = P_B = 0.5$).
2. **Asymétrie temporelle de retour :** Les fourmis engagées sur la branche courte $A$ effectuent l'aller-retour en un temps $\Delta t_A = 2 L_A / v$, alors que celles sur la branche longue $B$ mettent un temps $\Delta t_B = 2 L_B / v = 2 \cdot \Delta t_A$ (où $v$ est la vitesse de déplacement).
3. **Amplification par rétroaction positive :** Au moment où les premières fourmis de la branche $A$ reviennent au nid, elles ont déjà déposé deux couches de phéromone, alors que celles de la branche $B$ n'ont pas encore atteint la nourriture.
4. **Convergence de la colonie :** Les nouvelles fourmis sortant du nid détectent une concentration en phéromone plus élevée sur la branche $A$ et s'y engagent préférentiellement. Rapidement, la quasi-totalité de la colonie emprunte le chemin le plus court.

---

## 2. Modélisation Mathématique Formelle du TSP

Le **Problème du Voyageur de Commerce** (TSP) est le problème d'optimisation combinatoire NP-difficile de référence pour valider les métaheuristiques.

### 2.1 Définition du Graphe
Soit un ensemble ordonné de $n$ villes $V = \{0, 1, \dots, n-1\}$ localisées dans un plan euclidien bidimensionnel par leurs coordonnées cartésiennes :
$$C_i = (x_i, y_i) \in \mathbb{R}^2 \quad \forall i \in V$$

Le problème est modélisé par un graphe complet non orienté $G = (V, E)$ où l'ensemble des arêtes est $E = \{(i, j) \mid i, j \in V, i \neq j\}$.

### 2.2 Matrice des Distances Euclidiennes
Chaque arête $(i, j)$ est pondérée par la distance euclidienne $d_{ij}$ :
$$d_{ij} = \|C_i - C_j\|_2 = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

La matrice des distances $D = (d_{ij})_{n \times n}$ vérifie les propriétés fondamentales suivantes :
1. **Nullité diagonale :** $d_{ii} = 0, \quad \forall i \in V$
2. **Symétrie :** $d_{ij} = d_{ji}, \quad \forall i, j \in V$
3. **Stricte positivité :** $d_{ij} > 0, \quad \forall i \neq j$
4. **Inégalité triangulaire :** $d_{ik} \le d_{ij} + d_{jk}, \quad \forall i, j, k \in V$

### 2.3 Fonction Objectif
Une solution admissible est un **cycle hamiltonien**, c'est-à-dire une permutation $\pi = (\pi_0, \pi_1, \dots, \pi_{n-1})$ de $V$ visitant chaque ville exactement une fois et revenant au point de départ $\pi_0$.

La fonction objectif à minimiser est la longueur totale fermée :
$$\min_{\pi \in \mathcal{S}_n} L(\pi) = \sum_{i=0}^{n-2} d_{\pi_i, \pi_{i+1}} + d_{\pi_{n-1}, \pi_0}$$
L'espace de recherche comporte $|\mathcal{S}_n| = \frac{(n-1)!}{2}$ tournées distinctes dans le cas symétrique.

---

## 3. Le Moteur Mathématique d'Ant System (AS)

Dans Ant System, une colonie de $m$ fourmis artificielles coopère pour construire itérativement des tournées en exploitant deux sources d'information :
1. Une information **locale et statique** : la visibilité heuristique $\eta_{ij}$.
2. Une information **globale et dynamique** : la trace de phéromone $\tau_{ij}$.

### 3.1 Visibilité Heuristique
L'attractivité immédiate d'une arête $(i, j)$ est inversement proportionnelle à sa distance :
$$\eta_{ij} = \frac{1}{d_{ij}} \quad (\text{avec } \eta_{ii} = 0)$$

### 3.2 Règle Proportionnelle de Transition (Roulette Pondérée)
Soit une fourmi $k$ située à la ville courante $i$. Soit $N_i^k \subset V$ l'ensemble des villes non encore visitées par la fourmi $k$ (conservées dans sa mémoire taboue).

La probabilité $P_{ij}^k(t)$ que la fourmi $k$ choisisse de se déplacer vers la ville $j \in N_i^k$ à l'itération $t$ est :
$$P_{ij}^k(t) = \begin{cases} \dfrac{[\tau_{ij}(t)]^\alpha \cdot [\eta_{ij}]^\beta}{\displaystyle\sum_{l \in N_i^k} [\tau_{il}(t)]^\alpha \cdot [\eta_{il}]^\beta} & \text{si } j \in N_i^k \\ 0 & \text{sinon} \end{cases}$$

#### Interprétation des exposants :
- **$\alpha \ge 0$ (Poids de la Phéromone) :** Détermine l'importance de l'historique collectif. Si $\alpha = 0$, les phéromones sont ignorées et le choix devient purement glouton ou aléatoire.
- **$\beta \ge 0$ (Poids de l'Heuristique) :** Détermine l'attrait de la proximité immédiate. Si $\beta = 0$, la recherche est uniquement guidée par les phéromones, ce qui ralentit la découverte de bonnes solutions initiales.

### 3.3 Évaporation des Phéromones
À la fin de chaque génération (lorsque les $m$ fourmis ont complété leur cycle), une fraction $\rho$ de la phéromone s'évapore sur l'ensemble du réseau :
$$\tau_{ij} \leftarrow (1 - \rho)\,\tau_{ij}, \quad \forall (i, j) \in E$$
où $\rho \in ]0, 1[$ est le coefficient d'évaporation. L'évaporation permet d'éviter l'accumulation illimitée de phéromones sur les chemins sous-optimaux initiaux.

### 3.4 Dépôt de Phéromones
Chaque fourmi $k$ dépose une quantité de phéromone inversement proportionnelle à la longueur $L_k$ de son cycle :
$$\Delta \tau_{ij}^k = \begin{cases} \dfrac{Q}{L_k} & \text{si l'arête } (i, j) \text{ appartient au cycle de la fourmi } k \\ 0 & \text{sinon} \end{cases}$$
où $Q > 0$ est une constante de mise à l'échelle.

La mise à jour globale s'écrit donc :
$$\tau_{ij}(t+1) = (1 - \rho)\,\tau_{ij}(t) + \sum_{k=1}^m \Delta \tau_{ij}^k(t)$$

---

## 4. Exemple Numérique Complet Détaillé (Calcul Manuel Pas-à-Pas)

Pour illustrer rigoureusement le fonctionnement mathématique sans aucune ambiguïté, considérons l'instance didactique à 5 villes :

### 4.1 Coordonnées Cartésiennes
- **Ville 0 :** $C_0 = (20, 20)$ (Coin Sud-Ouest)
- **Ville 1 :** $C_1 = (20, 80)$ (Coin Nord-Ouest)
- **Ville 2 :** $C_2 = (80, 80)$ (Coin Nord-Est)
- **Ville 3 :** $C_3 = (80, 20)$ (Coin Sud-Est)
- **Ville 4 :** $C_4 = (50, 50)$ (Centre du carré)

### 4.2 Calcul Détaillé de la Matrice des Distances $D$
- $d_{01} = \sqrt{(20 - 20)^2 + (80 - 20)^2} = \sqrt{0 + 3600} = 60.00$
- $d_{02} = \sqrt{(80 - 20)^2 + (80 - 20)^2} = \sqrt{3600 + 3600} = \sqrt{7200} = 60\sqrt{2} \approx 84.8528$
- $d_{03} = \sqrt{(80 - 20)^2 + (20 - 20)^2} = \sqrt{3600 + 0} = 60.00$
- $d_{04} = \sqrt{(50 - 20)^2 + (50 - 20)^2} = \sqrt{900 + 900} = \sqrt{1800} = 30\sqrt{2} \approx 42.4264$

Par symétrie, la matrice $D$ s'écrit :
$$D = \begin{pmatrix}
0.00 & 60.00 & 84.85 & 60.00 & 42.43 \\
60.00 & 0.00 & 60.00 & 84.85 & 42.43 \\
84.85 & 60.00 & 0.00 & 60.00 & 42.43 \\
60.00 & 84.85 & 60.00 & 0.00 & 42.43 \\
42.43 & 42.43 & 42.43 & 42.43 & 0.00
\end{pmatrix}$$

### 4.3 Matrice de Visibilité Heuristique $\eta_{ij} = \frac{1}{d_{ij}}$
$$\eta = \begin{pmatrix}
0.0000 & 0.0167 & 0.0118 & 0.0167 & 0.0236 \\
0.0167 & 0.0000 & 0.0167 & 0.0118 & 0.0236 \\
0.0118 & 0.0167 & 0.0000 & 0.0167 & 0.0236 \\
0.0167 & 0.0118 & 0.0167 & 0.0000 & 0.0236 \\
0.0236 & 0.0236 & 0.0236 & 0.0236 & 0.0000
\end{pmatrix}$$

### 4.4 Calcul Manuel d'une Étape de Transition
Supposons que la colonie débute avec une phéromone uniforme $\tau_0 = 1.0$, et fixons $\alpha = 1.0$, $\beta = 3.0$.  
Une fourmi est positionnée à la **Ville 0**. L'ensemble des villes non visitées est $N_0 = \{1, 2, 3, 4\}$.

Calculons pour chaque destination candidate le numérateur $w_{0j} = [\tau_{0j}]^\alpha \cdot [\eta_{0j}]^\beta$ :
- **Vers Ville 1 ($d_{01} = 60$) :**
  $$w_{01} = (1.0)^1 \cdot \left(\frac{1}{60}\right)^3 = \frac{1}{216000} \approx 4.6296 \times 10^{-6}$$
- **Vers Ville 2 ($d_{02} = 84.8528$) :**
  $$w_{02} = (1.0)^1 \cdot \left(\frac{1}{84.8528}\right)^3 = \frac{1}{610940} \approx 1.6368 \times 10^{-6}$$
- **Vers Ville 3 ($d_{03} = 60$) :**
  $$w_{03} = (1.0)^1 \cdot \left(\frac{1}{60}\right)^3 = \frac{1}{216000} \approx 4.6296 \times 10^{-6}$$
- **Vers Ville 4 (Centre, $d_{04} = 42.4264$) :**
  $$w_{04} = (1.0)^1 \cdot \left(\frac{1}{42.4264}\right)^3 = \frac{1}{76367} \approx 13.0946 \times 10^{-6}$$

#### Dénominateur (Somme des poids) :
$$\sum_{l \in N_0} w_{0l} = (4.6296 + 1.6368 + 4.6296 + 13.0946) \times 10^{-6} = 23.9906 \times 10^{-6}$$

#### Probabilités de Transition Normalisées :
$$P_{01} = \frac{4.6296}{23.9906} = 0.1930 \quad (\mathbf{19.30\,\%})$$
$$P_{02} = \frac{1.6368}{23.9906} = 0.0682 \quad (\mathbf{6.82\,\%})$$
$$P_{03} = \frac{4.6296}{23.9906} = 0.1930 \quad (\mathbf{19.30\,\%})$$
$$P_{04} = \frac{13.0946}{23.9906} = 0.5458 \quad (\mathbf{54.58\,\%})$$

$$\sum_{j \in N_0} P_{0j} = 19.30\% + 6.82\% + 19.30\% + 54.58\% = \mathbf{100.00\,\%}$$

> **Interprétation :** La Ville 4 (centre), étant la plus proche géographiquement ($d=42.43$), recueille plus de la moitié des probabilités ($54.58\%$), illustrant parfaitement la force d'attraction du terme heuristique $[\eta]^\beta$.

### 4.5 Calcul du Coût de la Tournée Optimale
Supposons que la fourmi parcoure le cycle :
$$0 \longrightarrow 4 \longrightarrow 3 \longrightarrow 2 \longrightarrow 1 \longrightarrow 0$$

Sa longueur totale fermée est :
$$L = d_{04} + d_{43} + d_{32} + d_{21} + d_{10}$$
$$L = 42.4264 + 42.4264 + 60.00 + 60.00 + 60.00 = 84.8528 + 180.00 = \mathbf{264.8528}$$
Il s'agit de l'**optimum global absolu** pour cette instance à 5 villes (vérifié par solveur exhaustif exact).

### 4.6 Évaporation et Dépôt de Phéromone
Soit $Q = 100.0$ et $\rho = 0.40$ ($1 - \rho = 0.60$) :
1. **Évaporation sur toutes les arêtes :**
   $$\tau_{ij} \leftarrow 0.60 \times 1.0 = 0.60$$
2. **Dépôt sur les arêtes de la tournée :**
   $$\Delta \tau = \frac{Q}{L} = \frac{100.0}{264.8528} \approx 0.3776$$
3. **Mise à jour finale pour les arêtes du cycle (ex: arête (0, 4)) :**
   $$\tau_{04} = 0.60 + 0.3776 = \mathbf{0.9776}$$
   Pour les arêtes non empruntées (ex: diagonale (0, 2)) :
   $$\tau_{02} = 0.60 + 0.00 = \mathbf{0.6000}$$

À l'itération suivante, l'arête $(0, 4)$ possède une concentration en phéromone $63\%$ plus élevée que l'arête $(0, 2)$, amorçant l'apprentissage collectif de la colonie.

---

## 5. Variantes Avancées d'ACO

### 5.1 Elitist Ant System (EAS)
Proposé par Dorigo, Maniezzo et Colorni (1996), l'EAS ajoute un renforcement artificiel sur la meilleure tournée observée depuis le début de l'algorithme ($T_{best}$, de longueur $L_{best}$) :
$$\tau_{ij}(t+1) = (1 - \rho)\,\tau_{ij}(t) + \sum_{k=1}^m \Delta \tau_{ij}^k(t) + e \cdot \Delta \tau_{ij}^{best}$$
où :
$$\Delta \tau_{ij}^{best} = \begin{cases} \dfrac{Q}{L_{best}} & \text{si } (i, j) \in T_{best} \\ 0 & \text{sinon} \end{cases}$$
Le paramètre $e \ge 0$ (poids élitiste, typiquement $e \approx 5$) accélère considérablement la vitesse de convergence vers les solutions d'élite.

### 5.2 Ant Colony System (ACS)
Développé par Dorigo et Gambardella (1997), l'ACS introduit trois ruptures fondamentales :
1. **Règle de décision pseudo-aléatoire proportionnelle :**  
   Pour arbitrer explicitement entre exploitation pure et exploration :
   $$j = \begin{cases} \arg\max_{l \in N_i^k} \left( \tau_{il} \cdot [\eta_{il}]^\beta \right) & \text{si } q \le q_0 \quad (\text{Exploitation pure}) \\ J & \text{sinon } (q > q_0) \quad (\text{Exploration probabiliste standard}) \end{cases}$$
   où $q \sim \mathcal{U}(0, 1)$ et $q_0 \in [0, 1]$ est un paramètre fixé (généralement $q_0 = 0.90$).
2. **Évaporation locale immédiate :**  
   Dès qu'une fourmi franchit l'arête $(i, j)$ lors de la construction de sa tournée :
   $$\tau_{ij} \leftarrow (1 - \xi)\,\tau_{ij} + \xi\,\tau_0$$
   où $\xi \in ]0, 1[$ (typiquement $\xi = 0.10$). Cette évaporation locale abaisse temporairement l'attractivité de l'arête pour les autres fourmis de la même génération, stimulant la diversification des chemins explorés.
3. **Mise à jour globale restreinte :**  
   Seule la meilleure tournée absolue (*best-so-far*) dépose de la phéromone à la fin de la génération :
   $$\tau_{ij} \leftarrow (1 - \rho)\,\tau_{ij} + \rho \cdot \frac{Q}{L_{best}} \quad \forall (i, j) \in T_{best}$$

---

## 6. Analyse de Complexité et Convergence

### 6.1 Complexité Algorithmique
Pour un réseau de $n$ villes, une colonie de $m$ fourmis et un horizon de $T$ générations :
- **Calcul de la matrice $D$ :** $\mathcal{O}(n^2)$ (effectué une seule fois).
- **Sélection d'une ville suivante :** À chaque étape, une fourmi évalue $|N_i| \le n$ villes candidates, soit $\mathcal{O}(n)$.
- **Construction d'une tournée complète :** $n$ étapes de sélection $\implies \mathcal{O}(n^2)$ par fourmi.
- **Génération complète des $m$ tournées :** $\mathcal{O}(m \cdot n^2)$ par itération.
- **Mise à jour des phéromones :** Évaporation et dépôts sur les arêtes $\implies \mathcal{O}(n^2)$ par itération.
- **Complexité Totale de l'Algorithme :**
$$\mathbf{\mathcal{O}(T \cdot m \cdot n^2)}$$

### 6.2 Indicateur d'Entropie de Shannon (Mesure de Convergence)
Pour surveiller scientifiquement l'état d'exploration vs stagnation de la colonie, nous calculons l'entropie de Shannon normalisée de la matrice des phéromones :
$$p_{ij} = \frac{\tau_{ij}}{\sum_{l} \tau_{il}}, \quad H_i = -\sum_{j} p_{ij} \ln(p_{ij})$$
$$H_{\text{norm}} = \frac{\frac{1}{n}\sum_{i=1}^n H_i}{\ln(n - 1)} \in [0, 1]$$
- **$H_{\text{norm}} \approx 1$ :** Distribution uniforme des phéromones (phase exploratoire active, diversité maximale).
- **$H_{\text{norm}} \to 0$ :** Les phéromones sont concentrées sur un unique cycle hamiltonien (convergence et arrêt de la recherche).
