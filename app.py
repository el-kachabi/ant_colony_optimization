"""
Application Interactive Unifiée - Optimisation par Colonie de Fourmis (ACO) & TSP.
Interface Streamlit mono-page structurée en 3 onglets pédagogiques.
Chaque section est commentée pour expliquer le rôle des composants d'interface et des appels algorithmiques.
"""

import pandas as pd
import streamlit as st

from src.core_aco import (
    PRESET_INSTANCES,
    AntSystem,
    TSPInstance,
    solve_brute_force,
    solve_nearest_neighbor,
)
from src.visualization import plot_convergence, plot_tsp_solution

# Configuration générale de la fenêtre Streamlit (titre de l'onglet navigateur et mise en page large)
st.set_page_config(
    page_title="ACO & TSP - Plateforme Pédagogique",
    page_icon="🐜",
    layout="wide",
)

# ==============================================================================
# 1. BARRE LATÉRALE : CONFIGURATION DU PROBLÈME TSP ET DES HYPERPARAMÈTRES ACO
# ==============================================================================
with st.sidebar:
    st.title("🐜 Paramètres ACO")
    st.caption("Configuration du problème TSP et de la colonie")

    # Sélection du mode de génération des villes : instance étalonnée ou génération aléatoire
    mode_carte = st.radio("Choix de l'instance :", ["Instances de Référence", "Carte Aléatoire"])
    if mode_carte == "Instances de Référence":
        preset_key = st.selectbox(
            "Instance TSP :",
            options=list(PRESET_INSTANCES.keys()),
            format_func=lambda k: PRESET_INSTANCES[k]["name"],
            index=2,  # Sélectionne l'instance de 20 villes par défaut
        )
        instance = TSPInstance.load_preset(preset_key)
    else:
        n_random = st.slider("Nombre de villes :", min_value=5, max_value=40, value=15, step=5)
        seed_map = st.number_input("Graine de la carte :", min_value=1, max_value=999, value=42)
        instance = TSPInstance.generate_random(num_cities=n_random, seed=int(seed_map))

    st.divider()
    st.subheader("Réglages de la Colonie")
    # Curseurs interactifs permettant de modifier en direct les hyperparamètres de l'algorithme
    alpha = st.slider("α (Poids Phéromones) :", 0.0, 5.0, 1.0, 0.2, help="Exploitation de la mémoire collective")
    beta = st.slider("β (Poids Visibilité 1/d) :", 0.0, 8.0, 3.0, 0.5, help="Exploration gloutonne locale")
    rho = st.slider("ρ (Taux d'Évaporation) :", 0.05, 0.95, 0.50, 0.05, help="Vitesse d'oubli des mauvaises pistes")
    num_ants = st.slider("m (Nombre de Fourmis) :", 5, 50, 20, 5)
    max_iter = st.slider("T (Nombre d'Itérations) :", 10, 100, 40, 5)
    use_elitism = st.checkbox("Activer le renforcement Élitiste (EAS)", value=False)
    seed_aco = st.number_input("Graine aléatoire ACO :", min_value=1, max_value=999, value=42)

# ==============================================================================
# 2. EN-TÊTE PRINCIPAL & EXÉCUTION DU MOTEUR D'OPTIMISATION
# ==============================================================================
st.title("🐜 Optimisation par Colonie de Fourmis (ACO) — Résolution du TSP")
st.markdown(
    "**Métaheuristique bio-inspirée appliquée au Problème du Voyageur de Commerce** • "
    "*Présentation interactive en 3 volets : Simulation, Modèle Mathématique et Comparatif.*"
)

# Instanciation et lancement du solveur ACO avec les paramètres choisis dans la barre latérale
solver = AntSystem(
    alpha=alpha,
    beta=beta,
    rho=rho,
    q=100.0,
    num_ants=num_ants,
    max_iter=max_iter,
    elitist_weight=5.0 if use_elitism else 0.0,
    seed=int(seed_aco),
)
aco_res = solver.solve(instance)

# Exécution de l'heuristique Gloutonne (Plus Proche Voisin) pour disposer d'une base de comparaison
_, greedy_len, greedy_time = solve_nearest_neighbor(instance)
gain_vs_greedy = ((greedy_len - aco_res.best_length) / greedy_len) * 100.0

# ==============================================================================
# 3. NAVIGATION EN 3 ONGLETS PÉDAGOGIQUES
# ==============================================================================
tab_demo, tab_theory, tab_comp = st.tabs([
    "🚀 1. Démonstration & Simulation TSP",
    "📐 2. Principe & Modèle Mathématique",
    "🏆 3. Comparaison & Applications",
])

# ------------------------------------------------------------------------------
# ONGLET 1 : DÉMONSTRATION INTERACTIVE & SIMULATION
# ------------------------------------------------------------------------------
with tab_demo:
    # Bandeau de 4 indicateurs clés de performance (KPIs)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Villes (N)", f"{instance.dimension} sommets")
    k2.metric("Meilleure Distance ACO", f"{aco_res.best_length:.2f}", f"{-gain_vs_greedy:+.2f}% vs Glouton")
    k3.metric("Distance Glouton (Référence)", f"{greedy_len:.2f}")
    k4.metric("Temps de Calcul ACO", f"{aco_res.execution_time_ms:.1f} ms")

    # Niveau 1 : Carte du réseau TSP, pistes de phéromones et meilleure tournée (pleine largeur)
    st.plotly_chart(
        plot_tsp_solution(instance, aco_res.best_path, aco_res.pheromone_matrix),
        use_container_width=True,
    )

    # Affichage textuel de l'ordre exact de passage dans les villes
    circuit_str = " ➔ ".join(f"V{i}" for i in aco_res.best_path + [aco_res.best_path[0]])
    st.info(f"**Meilleur cycle hamiltonien identifié :** `{circuit_str}`")

    st.divider()

    # Niveau 2 : Courbe de convergence de la colonie au fil des générations (pleine largeur)
    st.plotly_chart(
        plot_convergence(aco_res.history_best, aco_res.history_mean, greedy_len, instance.optimal_known),
        use_container_width=True,
    )

# ------------------------------------------------------------------------------
# ONGLET 2 : PRINCIPE BIO-INSPIRÉ & MODÈLE MATHÉMATIQUE
# ------------------------------------------------------------------------------
with tab_theory:
    st.subheader("🌿 1. De la Stigmergie Naturelle à l'Algorithme ACO")
    st.markdown(
        "Inspiré par **Marco Dorigo (1991)** et l'expérience du **Double Pont**, l'algorithme ACO repose sur la "
        "**stigmergie** : les fourmis communiquent indirectement en déposant des **phéromones** sur leur trajet. "
        "Plus un chemin est court, plus les fourmis le parcourent vite et renforcent sa trace chimique, tandis que "
        "l'évaporation naturelle efface les détours sous-optimaux."
    )

    # Présentation des 3 formules mathématiques fondamentales en colonnes
    st.subheader("📐 2. Les 3 Équations Fondamentales du Modèle")
    c_eq1, c_eq2, c_eq3 = st.columns(3)
    with c_eq1:
        st.markdown("**① Visibilité Heuristique Locale**")
        st.latex(r"\eta_{ij} = \frac{1}{d_{ij}}")
        st.caption("Favorise les villes proches (attrait glouton immédiat).")
    with c_eq2:
        st.markdown("**② Probabilité de Transition**")
        st.latex(r"P_{ij}^k = \frac{\tau_{ij}^\alpha \cdot \eta_{ij}^\beta}{\sum_{l \in N_i^k} \tau_{il}^\alpha \cdot \eta_{il}^\beta}")
        st.caption("Arbitrage entre mémoire collective (α) et proximité (β).")
    with c_eq3:
        st.markdown("**③ Évaporation & Dépôt Global**")
        st.latex(r"\tau_{ij} \leftarrow (1 - \rho)\tau_{ij} + \sum_{k=1}^{m} \frac{Q}{L_k}")
        st.caption("Oubli progressif (ρ) et récompense inversement proportionnelle à la distance.")

    st.divider()
    st.subheader("🧮 3. Exemple Numérique en Direct : Décision depuis la Ville 0")
    st.caption("Décomposition du calcul des probabilités de transition au départ de la Ville 0 vers ses voisins :")

    # Calcul en temps réel des probabilités de transition depuis la Ville 0 vers les premières villes voisines
    unvisited_demo = list(range(1, min(6, instance.dimension)))
    probs_demo = solver.compute_probabilities(0, unvisited_demo, aco_res.pheromone_matrix, instance.heuristic_matrix)
    rows_calc = []
    for city_j, prob_val in zip(unvisited_demo, probs_demo):
        d_val = instance.distance_matrix[0, city_j]
        eta_val = instance.heuristic_matrix[0, city_j]
        tau_val = aco_res.pheromone_matrix[0, city_j]
        rows_calc.append({
            "Ville Candidate (j)": f"Ville {city_j}",
            "Distance d(0,j)": round(float(d_val), 2),
            "Visibilité η = 1/d": round(float(eta_val), 4),
            "Phéromone τ(0,j)": round(float(tau_val), 3),
            "Probabilité P(0➔j)": f"{prob_val * 100:.1f} %",
        })
    st.dataframe(pd.DataFrame(rows_calc), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# ONGLET 3 : COMPARAISON ALGORITHMIQUE & CAS D'USAGE
# ------------------------------------------------------------------------------
with tab_comp:
    st.subheader(f"⚔️ Banc d'Essai Comparatif sur : {instance.name}")
    # Comparaison directe entre Ant System standard (e=0), Elitist Ant System (e=5) et le Glouton
    res_as = AntSystem(alpha, beta, rho, 100.0, num_ants, max_iter, elitist_weight=0.0, seed=int(seed_aco)).solve(instance)
    res_eas = AntSystem(alpha, beta, rho, 100.0, num_ants, max_iter, elitist_weight=5.0, seed=int(seed_aco)).solve(instance)

    comp_rows = [
        {"Algorithme": "Plus Proche Voisin (Glouton)", "Catégorie": "Heuristique constructive", "Distance": round(greedy_len, 2), "Temps (ms)": round(greedy_time, 2)},
        {"Algorithme": "Ant System (AS Standard)", "Catégorie": "Métaheuristique ACO", "Distance": round(res_as.best_length, 2), "Temps (ms)": round(res_as.execution_time_ms, 1)},
        {"Algorithme": "Elitist Ant System (EAS)", "Catégorie": "Métaheuristique ACO Élitiste", "Distance": round(res_eas.best_length, 2), "Temps (ms)": round(res_eas.execution_time_ms, 1)},
    ]
    # Si le nombre de villes est <= 10, on ajoute la référence exacte par Force Brute
    if instance.dimension <= 10:
        _, bf_len, bf_time = solve_brute_force(instance)
        comp_rows.append({"Algorithme": "Brute Force Exact ((N-1)!/2)", "Catégorie": "Optimum Absolu Garanti", "Distance": round(bf_len, 2), "Temps (ms)": round(bf_time, 2)})

    st.dataframe(pd.DataFrame(comp_rows).sort_values("Distance"), use_container_width=True, hide_index=True)

    st.markdown(
        """
        ### 🌐 Synthèse & Applications Industrielles
        * **Pourquoi l'ACO surpasse le Glouton ?** L'algorithme glouton fait des choix myopes au début et subit de lourds détours en fin de parcours. Grâce à l'exploration collective et aux phéromones, l'ACO découvre une structure globale optimale.
        * **Applications concrètes :** Logistique et tournées de livraison (*Vehicle Routing Problem*), routage de paquets télécoms (*AntNet*), perçage de circuits imprimés (*PCB Drilling*) et ordonnancement d'usines.
        """
    )
