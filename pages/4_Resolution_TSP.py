"""
Interactive TSP Solver Page with multi-generator support and customizable ACO parameters.
Custom map generator, parameter tuning, and execution monitoring.
"""

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Résolution TSP | ACO Lab", page_icon="🧭", layout="wide")

from config.settings import PRESET_INSTANCES
from src.aco import AntSystem
from src.variants import ElitistAntSystem, AntColonySystem
from src.baselines import NearestNeighborSolver, TwoOptSolver, BruteForceSolver
from src.tsp import TSPInstance
from src.ui_components import inject_custom_css, initialize_session_state, render_page_header, render_sidebar
from src.visualization import plot_tour, plot_convergence_curves

inject_custom_css()
initialize_session_state()
render_sidebar()

render_page_header(
    title="Résolution Interactive du TSP",
    subtitle="Configuration complète du problème, sélection des métaheuristiques et résolution en direct",
    badge="MODULE 4 • SOLVEUR TSP",
)

# Configuration Card: Instance Generation & Selection
st.markdown("### 1. Sélection & Génération de l'Instance TSP")
c_type, c_param1, c_param2 = st.columns([1.2, 1, 1])

with c_type:
    instance_mode = st.radio(
        "Mode de définition du problème :",
        ["Presets Didactiques & Benchmarks", "Génération Aléatoire Personnalisée", "Instance Circulaire Régulière"],
        horizontal=True,
    )

if instance_mode == "Presets Didactiques & Benchmarks":
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        preset_key = st.selectbox(
            "Choisir parmi les instances étalonnées :",
            options=list(PRESET_INSTANCES.keys()),
            format_func=lambda k: PRESET_INSTANCES[k]["name"],
            key="tsp_solver_preset",
        )
    with col_p2:
        if st.button("Charger l'instance sélectionnée", use_container_width=True):
            st.session_state.tsp_instance = TSPInstance.load_preset(preset_key)
            st.success(f"Instance '{PRESET_INSTANCES[preset_key]['name']}' chargée !")
            st.rerun()

elif instance_mode == "Génération Aléatoire Personnalisée":
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        n_cities = st.slider("Nombre de villes :", min_value=5, max_value=60, value=25, step=5)
    with col_r2:
        r_seed = st.number_input("Graine aléatoire :", min_value=1, max_value=9999, value=42)
    with col_r3:
        if st.button("🎲 Générer Nouvelle Carte", use_container_width=True):
            st.session_state.tsp_instance = TSPInstance.generate_random(num_cities=n_cities, seed=r_seed)
            st.success(f"Génération aléatoire de {n_cities} villes réussie !")
            st.rerun()

else:  # Circulaire
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        n_circ = st.slider("Nombre de sommets :", min_value=6, max_value=30, value=12, step=2)
    with col_c2:
        noise = st.slider("Bruit radial :", min_value=0.0, max_value=10.0, value=0.0, step=1.0)
    with col_c3:
        if st.button("⭕ Générer Carte Circulaire", use_container_width=True):
            st.session_state.tsp_instance = TSPInstance.generate_circular(num_cities=n_circ, noise_std=noise)
            st.success("Carte circulaire générée !")
            st.rerun()

inst = st.session_state.tsp_instance

st.markdown("---")
# Algorithm Selection and Hyperparameters Card
st.markdown("### 2. Choix de l'Algorithme et Hyperparamètres")

col_a1, col_a2 = st.columns([1, 2])

with col_a1:
    available_algos = [
        "Ant System (AS)",
        "Elitist Ant System (EAS)",
        "Ant Colony System (ACS)",
        "Nearest Neighbor (Glouton)",
        "2-Opt Local Search",
    ]
    if inst.dimension <= 10:
        available_algos.append("Exact Brute Force")

    selected_algo = st.selectbox("Algorithme de Résolution :", available_algos, index=0)

with col_a2:
    if "Ant" in selected_algo:
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            h_alpha = st.slider("Alpha (α) :", 0.0, 4.0, 1.0, 0.2)
        with h2:
            h_beta = st.slider("Beta (β) :", 1.0, 8.0, 3.0, 0.5)
        with h3:
            h_rho = st.slider("Rho (ρ) :", 0.05, 0.90, 0.40, 0.05)
        with h4:
            h_ants = st.slider("Fourmis (m) :", 5, 80, 20, 5)

        h5, h6, h7 = st.columns(3)
        with h5:
            h_iter = st.slider("Itérations (T) :", 10, 200, 50, 10)
        with h6:
            h_q = st.number_input("Constante Q :", 10.0, 1000.0, 100.0, 10.0)
        with h7:
            h_seed = st.number_input("Graine de Reproductibilité :", 1, 9999, 42)

st.markdown("<br>", unsafe_allow_html=True)
solve_btn = st.button("🚀 Résoudre le TSP Maintenant", type="primary", use_container_width=True)

if solve_btn:
    progress_bar = st.progress(0, text="Initialisation de la colonie...")

    def update_progress(curr, total, best_len):
        pct = int(curr / total * 100)
        progress_bar.progress(pct, text=f"Itération {curr}/{total} - Meilleure longueur : {best_len:.2f}")

    if selected_algo == "Ant System (AS)":
        solver = AntSystem(alpha=h_alpha, beta=h_beta, rho=h_rho, q=h_q, num_ants=h_ants, max_iter=h_iter, seed=h_seed)
        res = solver.solve(inst, progress_callback=update_progress)
    elif selected_algo == "Elitist Ant System (EAS)":
        solver = ElitistAntSystem(alpha=h_alpha, beta=h_beta, rho=h_rho, q=h_q, num_ants=h_ants, max_iter=h_iter, elitist_weight=5.0, seed=h_seed)
        res = solver.solve(inst, progress_callback=update_progress)
    elif selected_algo == "Ant Colony System (ACS)":
        solver = AntColonySystem(alpha=h_alpha, beta=h_beta, rho=0.10, q=h_q, num_ants=h_ants, max_iter=h_iter, q0=0.90, xi=0.10, seed=h_seed)
        res = solver.solve(inst, progress_callback=update_progress)
    elif "Nearest Neighbor" in selected_algo:
        solver = NearestNeighborSolver(multi_start=True)
        res = solver.solve(inst)
    elif "2-Opt" in selected_algo:
        solver = TwoOptSolver()
        res = solver.solve(inst)
    else:  # Brute Force
        solver = BruteForceSolver(max_cities=10)
        res = solver.solve(inst)

    progress_bar.progress(100, text="Résolution terminée avec succès !")

    # Display Metrics
    st.markdown("### 3. Résultats de Résolution")
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-val">{res.best_tour.length:.2f}</div>
                <div class="metric-lbl">Longueur Minimale</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k2:
        if inst.optimal_known:
            gap = (res.best_tour.length - inst.optimal_known) / inst.optimal_known * 100
            gap_str = f"{gap:+.2f} %"
            gap_col = "#10B981" if gap <= 0.05 else ("#FF9900" if gap <= 3.0 else "#EF4444")
        else:
            gap_str = "N/A"
            gap_col = "#94A3B8"
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-val" style="color: {gap_col};">{gap_str}</div>
                <div class="metric-lbl">Écart à l'Optimum Référence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k3:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-val" style="color: #38BDF8;">{res.execution_time_sec * 1000:.1f} ms</div>
                <div class="metric-lbl">Temps de Calcul</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with k4:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-val" style="color: #8B5CF6;">{inst.dimension}</div>
                <div class="metric-lbl">Villes Visitées</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    res_l, res_r = st.columns(2)

    with res_l:
        st.markdown("#### 🏆 Graphe de la Tournée Optimale")
        fig_t = plot_tour(inst, res.best_tour)
        st.plotly_chart(fig_t, use_container_width=True)

    with res_r:
        if hasattr(res, "convergence_history") and res.convergence_history:
            st.markdown("#### 📈 Courbe de Convergence")
            fig_c = plot_convergence_curves(res.convergence_history, optimal_known=inst.optimal_known)
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.markdown("#### ℹ️ Méthode Non Itérative")
            st.info("Cet algorithme baseline résout le problème en un seul passage sans boucle itérative.")

    # Tour details in expander
    with st.expander("🔍 Voir la séquence détaillée des étapes de la tournée"):
        path = res.best_tour.path
        n = len(path)
        step_data = []
        for i in range(n):
            u = path[i]
            v = path[(i + 1) % n]
            step_data.append({
                "Étape": i + 1,
                "Départ": inst.cities[u].name,
                "Arrivée": inst.cities[v].name,
                "Distance": round(float(inst.distance_matrix[u, v]), 2),
            })
        st.dataframe(pd.DataFrame(step_data), use_container_width=True)
