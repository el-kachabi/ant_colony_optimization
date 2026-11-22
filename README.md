# 🐜 Optimisation par Colonie de Fourmis (Ant Colony Optimization – ACO)
### Simulation Interactive et Résolution du Problème du Voyageur de Commerce (TSP)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/el-kachabi/ant_colony_optimization/blob/main/notebooks/ant_colony_optimization.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly Interactive](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 1. Présentation et Objectifs

Ce projet propose une implémentation **pédagogique, épurée et interactive** de l'algorithme d'**Optimisation par Colonie de Fourmis (ACO)** appliquée au **Problème du Voyageur de Commerce (TSP)**.

Inspirée par les travaux de **Marco Dorigo (1991, 1996)** et le mécanisme biologique de **stigmergie** (communication indirecte des fourmis par dépôt et évaporation de phéromones), la plateforme permet de visualiser comment une colonie d'agents simples converge collectivement vers le chemin le plus court.

### Les 3 piliers du projet :
* **Moteur algorithmique unifié (`src/core_aco.py`) :** Implémentation vectorisée d'*Ant System (AS)* avec option élitiste (*EAS*), accompagnée des solveurs de référence (*Plus Proche Voisin Glouton* et *Brute Force Exact*).
* **Application Web interactive (`app.py`) :** Interface Streamlit mono-page structurée en 3 onglets (Simulation en direct, Principe algorithmique et Banc d'essai comparatif).
* **Notebook interactif complet (`notebooks/ant_colony_optimization.ipynb`) :** Documente l'ensemble des **formulations mathématiques**, les démonstrations numériques pas-à-pas, l'étude de sensibilité des paramètres et les visualisations graphiques.  
  👉 **[Ouvrir le Notebook directement dans Google Colab](https://colab.research.google.com/github/el-kachabi/ant_colony_optimization/blob/main/notebooks/ant_colony_optimization.ipynb)**

---

## 📊 2. Résultats Expérimentaux & Benchmarks

Comparaison mesurée sur les 3 instances étalonnées intégrées au projet :

| Instance TSP | Villes | Algorithme | Distance Obtenue | Écart / Optimum | Temps | Conclusion |
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

## 🏗️ 3. Architecture Logicielle

### 3.1 Arborescence du Projet
```text
ant_colony_optimization/
├── app.py                                    # Application Streamlit mono-page à 3 onglets
├── README.md                                 # Présentation générale, benchmarks et guide d'exécution
├── requirements.txt                          # Dépendances Python (numpy, pandas, plotly, streamlit, matplotlib)
├── src/
│   ├── __init__.py
│   ├── core_aco.py                           # Moteur unifié : TSPInstance, AntSystem, Glouton, Brute Force
│   └── visualization.py                      # Graphiques interactifs Plotly (Carte TSP & Convergence)
└── notebooks/
    └── ant_colony_optimization.ipynb         # Théorie mathématique complète, calculs pas-à-pas et expériences
```

### 3.2 Organisation de l'Application Streamlit (`app.py`)
| Onglet | Objectif Pédagogique | Contenu Visuel |
| :--- | :--- | :--- |
| **1. Démonstration & Simulation TSP** | Visualiser la résolution en direct et le gain face au Glouton | 4 KPIs + Carte interactive des phéromones (Niveau 1) + Courbe de convergence (Niveau 2) |
| **2. Principe & Modèle Mathématique** | Comprendre le mécanisme de décision des fourmis | Synthèse du principe + Tableau numérique en direct des probabilités depuis la Ville 0 |
| **3. Comparaison & Applications** | Évaluer la performance face aux méthodes classiques | Tableau comparatif en direct (Glouton, AS, EAS, Brute Force) + Cas d'usage industriels |

---

## 🚀 4. Installation et Exécution

### 4.1 Installation des dépendances
```bash
pip install -r requirements.txt
```

### 4.2 Lancement de l'Application Interactive Streamlit
```bash
streamlit run app.py
```

### 4.3 Utilisation du Notebook Jupyter / Google Colab
* **En ligne (sans installation) :** Cliquez sur le badge **[Open in Colab](https://colab.research.google.com/github/el-kachabi/ant_colony_optimization/blob/main/notebooks/ant_colony_optimization.ipynb)** en haut de ce fichier.
* **En local :** Ouvrez `notebooks/ant_colony_optimization.ipynb` dans VS Code ou Jupyter Lab.

---

## 🌐 5. Cas d'Usage Industriels

* **🚚 Logistique & Tournées de Livraison (VRP) :** Réduction de 15 % à 25 % des distances parcourues par les flottes de véhicules.
* **📡 Routage Réseau Dynamique (AntNet) :** Adaptation décentralisée du routage de paquets IP face aux congestions réseau.
* **⚡ Perçage de Circuits Imprimés (PCB Drilling) :** Optimisation de la trajectoire laser sur des milliers de points de perçage.
* **⏱️ Ordonnancement d'Ateliers (Job-Shop Scheduling) :** Minimisation du temps total de production (*Makespan*).

---

## 📄 Licence

Ce projet est distribué sous licence **MIT** à des fins pédagogiques, académiques et scientifiques.
