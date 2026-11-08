"""
Comparative Algorithmic Benchmark Tournament Page.
Author: El Kachabi Badr-eddine
GitHub: https://github.com/el-kachabi
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Comparaison des Algorithmes | ACO Lab", page_icon="⚔️", layout="wide")

from src.aco import AntSystem
from src.variants import ElitistAntSystem, AntColonySystem
from src.baselines import NearestNeighborSolver, TwoOptSolver, BruteForceSolver
from src.ui_components import inject_custom_css, initialize_session_state, render_page_header, render_sidebar
from src.visualization import apply_dark_theme, plot_algorithm_comparison_barchart
from config.settings import THEME_COLORS

inject_custom_css()
initialize_session_state()
render_sidebar()

render_page_header(
    title="Banc d'Essai Comparatif des Algorithmes",
    subtitle="Tournoi d'optimisation en direct opposant les variantes ACO aux méthodes heuristiques et exactes",
    badge="MODULE 6 • BENCHMARK",
)

inst = st.session_state.tsp_instance

st.markdown(
    f"""
    <div class="custom-card" style="border-left: 4px solid #00F2FE;">
        <b>Instance active du tournoi :</b> <span style="color: #00F2FE;">{inst.name}</span> ({inst.dimension} villes)<br>
        <b>Optimum de référence :</b> {f"{inst.optimal_known:.2f}" if inst.optimal_known else "À déterminer par compétition"}
    </div>
    """,
    unsafe_allow_html=True,
)

col_ctrl1, col_ctrl2 = st.columns([1, 2])
with col_ctrl1:
    tournament_ants = st.slider("Fourmis par génération :", min_value=10, max_value=50, value=20, step=5)
with col_ctrl2:
    tournament_iter = st.slider("Nombre d'itérations :", min_value=15, max_value=100, value=35, step=5)

launch_tournament = st.button("⚔️ Lancer le Tournoi Comparatif en Direct", type="primary", use_container_width=True)

if launch_tournament or "benchmark_df" not in st.session_state:
    with st.spinner("Exécution du tournoi multi-algorithmes en cours..."):
        solvers = {
            "Ant System (AS)": AntSystem(alpha=1.0, beta=3.0, rho=0.4, num_ants=tournament_ants, max_iter=tournament_iter, seed=42),
            "Elitist Ant System (EAS)": ElitistAntSystem(alpha=1.0, beta=3.0, rho=0.4, num_ants=tournament_ants, max_iter=tournament_iter, elitist_weight=5.0, seed=42),
            "Ant Colony System (ACS)": AntColonySystem(alpha=1.0, beta=3.0, rho=0.1, num_ants=max(10, tournament_ants // 2), max_iter=tournament_iter, q0=0.90, xi=0.10, seed=42),
            "Nearest Neighbor": NearestNeighborSolver(multi_start=True),
            "2-Opt Local Search": TwoOptSolver(),
        }

        if inst.dimension <= 10:
            solvers["Exact Brute Force"] = BruteForceSolver(max_cities=10)

        results = []
        histories = {}

        for name, s in solvers.items():
            res = s.solve(inst)
            tour_len = res.best_tour.length
            time_ms = res.execution_time_sec * 1000

            gap = ""
            if inst.optimal_known:
                diff = (tour_len - inst.optimal_known) / inst.optimal_known * 100
                gap = f"{diff:+.2f}%"

            results.append({
                "Algorithme": name,
                "Type": "Métaheuristique" if "Ant" in name else ("Exact" if "Exact" in name else "Heuristique"),
                "Longueur": round(tour_len, 2),
                "Écart Optimum": gap if gap else "N/A",
                "Temps (ms)": round(time_ms, 2),
            })

            if hasattr(res, "convergence_history") and res.convergence_history:
                histories[name] = res.convergence_history

        df_bench = pd.DataFrame(results)
        df_bench = df_bench.sort_values(by="Longueur", ascending=True).reset_index(drop=True)
        df_bench.index = df_bench.index + 1
        df_bench.index.name = "Rang"
        st.session_state.benchmark_df = df_bench
        st.session_state.benchmark_histories = histories

df_res = st.session_state.benchmark_df

st.markdown("### 🏆 Classement Général du Tournoi")
st.dataframe(df_res, use_container_width=True)

# Visual Comparison Row
st.markdown("<br>", unsafe_allow_html=True)
g_left, g_right = st.columns(2)

with g_left:
    st.markdown("#### 📊 Longueur de Tournée Minimale par Algorithme")
    fig_bar = go.Figure()
    colors = [THEME_COLORS["primary_cyan"] if "Ant" in a else (THEME_COLORS["success_green"] if "Exact" in a else THEME_COLORS["accent_orange"]) for a in df_res["Algorithme"]]
    fig_bar.add_trace(go.Bar(x=df_res["Algorithme"], y=df_res["Longueur"], marker_color=colors, text=df_res["Longueur"], textposition="auto"))
    apply_dark_theme(fig_bar, title=f"Performances Comparées sur {inst.name}")
    st.plotly_chart(fig_bar, use_container_width=True)

with g_right:
    st.markdown("#### ⏱️ Vitesse d'Exécution (Temps en millisecondes)")
    fig_time = go.Figure()
    fig_time.add_trace(go.Bar(x=df_res["Algorithme"], y=df_res["Temps (ms)"], marker_color=THEME_COLORS["secondary_blue"], text=df_res["Temps (ms)"], textposition="auto"))
    apply_dark_theme(fig_time, title=f"Temps de Calcul (ms) sur {inst.name}")
    st.plotly_chart(fig_time, use_container_width=True)

# Convergence Comparison for iterative ACO variants
if "benchmark_histories" in st.session_state and st.session_state.benchmark_histories:
    st.markdown("### 📉 Comparaison des Dynamiques de Convergence (AS vs EAS vs ACS)")
    fig_comp_conv = go.Figure()
    palette_map = {
        "Ant System (AS)": THEME_COLORS["primary_cyan"],
        "Elitist Ant System (EAS)": THEME_COLORS["accent_orange"],
        "Ant Colony System (ACS)": THEME_COLORS["accent_purple"],
    }

    for algo_name, hist in st.session_state.benchmark_histories.items():
        iters = [r.iteration for r in hist]
        best_l = [r.best_length for r in hist]
        color = palette_map.get(algo_name, "#FFFFFF")
        fig_comp_conv.add_trace(go.Scatter(x=iters, y=best_l, mode="lines+markers", line=dict(color=color, width=3), name=algo_name))

    if inst.optimal_known:
        fig_comp_conv.add_trace(go.Scatter(x=[1, tournament_iter], y=[inst.optimal_known, inst.optimal_known], mode="lines", line=dict(color=THEME_COLORS["success_green"], width=2, dash="dashdot"), name="Optimum Référence"))

    apply_dark_theme(fig_comp_conv, title=f"Trajectoires de Convergence Comparées sur {inst.name}")
    fig_comp_conv.update_xaxes(title="Itérations")
    fig_comp_conv.update_yaxes(title="Longueur de Tournée")
    st.plotly_chart(fig_comp_conv, use_container_width=True)
