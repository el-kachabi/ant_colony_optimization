"""
Dashboard Page - Executive Overview and Real-Time ACO Optimization.
Interactive monitoring of optimization iterations and pheromone dynamics.
"""

import streamlit as st

st.set_page_config(page_title="Tableau de Bord | ACO Lab", page_icon="📊", layout="wide")

from src.aco import AntSystem
from src.variants import ElitistAntSystem, AntColonySystem
from src.ui_components import inject_custom_css, initialize_session_state, render_page_header, render_sidebar
from src.visualization import (
    plot_tour,
    plot_convergence_curves,
    plot_pheromone_network,
    plot_colony_entropy_and_pheromones,
)

inject_custom_css()
initialize_session_state()
render_sidebar()

render_page_header(
    title="Tableau de Bord Exécutif",
    subtitle="Monitoring en temps réel des performances de la colonie et cartographie des tournées",
    badge="MODULE 1 • MONITORING",
)

inst = st.session_state.tsp_instance

# Control Sidebar in Main Page
st.markdown("### ⚙️ Paramètres d'Exécution Rapide")
col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4, col_ctrl5 = st.columns(5)

with col_ctrl1:
    algo_choice = st.selectbox(
        "Métaheuristique :",
        ["Ant System (AS)", "Elitist Ant System (EAS)", "Ant Colony System (ACS)"],
        index=0,
    )

with col_ctrl2:
    num_ants = st.slider("Fourmis (m) :", min_value=5, max_value=60, value=20, step=5)

with col_ctrl3:
    max_iter = st.slider("Itérations (T) :", min_value=10, max_value=150, value=40, step=10)

with col_ctrl4:
    alpha = st.slider("Poids Phéromone (α) :", min_value=0.0, max_value=4.0, value=1.0, step=0.2)

with col_ctrl5:
    beta = st.slider("Poids Heuristique (β) :", min_value=1.0, max_value=8.0, value=3.0, step=0.5)

run_clicked = st.button("🚀 Lancer l'Optimisation en Temps Réel", type="primary", use_container_width=True)

if run_clicked or st.session_state.last_result is None:
    with st.spinner("Exécution de la colonie de fourmis en cours..."):
        if "Elitist" in algo_choice:
            solver = ElitistAntSystem(
                alpha=alpha,
                beta=beta,
                rho=0.40,
                q=100.0,
                num_ants=num_ants,
                max_iter=max_iter,
                elitist_weight=5.0,
                seed=st.session_state.aco_params["seed"],
            )
        elif "Colony System" in algo_choice:
            solver = AntColonySystem(
                alpha=alpha,
                beta=beta,
                rho=0.10,
                q=100.0,
                num_ants=num_ants,
                max_iter=max_iter,
                q0=0.90,
                xi=0.10,
                seed=st.session_state.aco_params["seed"],
            )
        else:
            solver = AntSystem(
                alpha=alpha,
                beta=beta,
                rho=0.40,
                q=100.0,
                num_ants=num_ants,
                max_iter=max_iter,
                seed=st.session_state.aco_params["seed"],
            )

        res = solver.solve(inst)
        st.session_state.last_result = res

res = st.session_state.last_result

# KPI Metrics Bar
st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-val">{res.best_tour.length:.2f}</div>
            <div class="metric-lbl">Meilleure Distance Trouvée</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    if inst.optimal_known:
        gap = (res.best_tour.length - inst.optimal_known) / inst.optimal_known * 100
        gap_color = "#10B981" if gap <= 0.05 else ("#FF9900" if gap <= 3.0 else "#EF4444")
        gap_txt = f"{gap:+.2f} %"
    else:
        gap_txt = "N/A"
        gap_color = "#94A3B8"
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-val" style="color: {gap_color};">{gap_txt}</div>
            <div class="metric-lbl">Écart à l'Optimum Référence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-val" style="color: #38BDF8;">{res.execution_time_sec * 1000:.1f} ms</div>
            <div class="metric-lbl">Temps de Calcul Total</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    last_entropy = res.convergence_history[-1].entropy
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-val" style="color: #8B5CF6;">{last_entropy:.3f}</div>
            <div class="metric-lbl">Entropie Finale (Convergence)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main Visual Plots
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("### 🏆 Meilleure Tournée Obtenue")
    fig_tour = plot_tour(inst, res.best_tour, title=f"Tournée Optimale : L = {res.best_tour.length:.2f}")
    st.plotly_chart(fig_tour, use_container_width=True)

with c_right:
    st.markdown("### 📈 Dynamique de Convergence Multi-Générations")
    fig_conv = plot_convergence_curves(res.convergence_history, optimal_known=inst.optimal_known)
    st.plotly_chart(fig_conv, use_container_width=True)

# Secondary Row: Pheromones and Entropy
p_left, p_right = st.columns(2)

with p_left:
    st.markdown("### 🧪 Réseau des Traces de Phéromones")
    fig_phero = plot_pheromone_network(inst, res.final_pheromone_matrix, best_tour=res.best_tour)
    st.plotly_chart(fig_phero, use_container_width=True)

with p_right:
    st.markdown("### 🧬 Transition de Phase (Entropie vs Phéromones)")
    fig_ent = plot_colony_entropy_and_pheromones(res.convergence_history)
    st.plotly_chart(fig_ent, use_container_width=True)
