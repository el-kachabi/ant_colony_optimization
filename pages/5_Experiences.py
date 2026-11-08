"""
Experimental Laboratory & Hyperparameter Sensitivity Analysis Page.
Author: El Kachabi Badr-eddine
GitHub: https://github.com/el-kachabi
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Expériences & Sensibilité | ACO Lab", page_icon="🧪", layout="wide")

from src.aco import AntSystem
from src.ui_components import inject_custom_css, initialize_session_state, render_page_header, render_sidebar
from src.visualization import apply_dark_theme, plot_sensitivity_heatmap
from config.settings import THEME_COLORS

inject_custom_css()
initialize_session_state()
render_sidebar()

render_page_header(
    title="Laboratoire d'Expérimentation & Sensibilité",
    subtitle="Étude systématique de l'impact des hyperparamètres sur la qualité et la vitesse de convergence",
    badge="MODULE 5 • EXPÉRIMENTATIONS",
)

inst = st.session_state.tsp_instance

exp_tabs = st.tabs([
    "1️⃣ Poids Phéromone (α)",
    "2️⃣ Poids Heuristique (β)",
    "3️⃣ Taux d'Évaporation (ρ)",
    "4️⃣ Taille Colonie (m)",
    "🔥 5️⃣ Heatmap 2D (α × β)",
])

# 1. ALPHA EXPERIMENT
with exp_tabs[0]:
    st.markdown("### 🧬 Rôle de la Mémoire Collective : Impact du Paramètre α")
    st.markdown(
        "Lorsque $\\alpha = 0$, les traces de phéromones sont totalement ignorées (recherche heuristique purement gloutonne ou aléatoire). "
        "Lorsque $\\alpha$ est très élevé, la colonie s'enferme prématurément dans les premières pistes tracées."
    )

    if st.button("Lancer l'Analyse de Sensibilité sur α", key="btn_alpha"):
        alpha_values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        results = []
        progress = st.progress(0)

        for idx, a_val in enumerate(alpha_values):
            aco = AntSystem(alpha=a_val, beta=3.0, rho=0.4, num_ants=15, max_iter=30, seed=42)
            res = aco.solve(inst)
            results.append({
                "Alpha": a_val,
                "Longueur": res.best_tour.length,
                "Temps (ms)": res.execution_time_sec * 1000,
                "Entropie": res.convergence_history[-1].entropy,
            })
            progress.progress((idx + 1) / len(alpha_values))

        df_alpha = pd.DataFrame(results)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_alpha["Alpha"], y=df_alpha["Longueur"], mode="lines+markers", line=dict(color=THEME_COLORS["primary_cyan"], width=3), marker=dict(size=8)))
        apply_dark_theme(fig, title=f"Sensibilité à α sur {inst.name}")
        fig.update_xaxes(title="Poids de la Phéromone α")
        fig.update_yaxes(title="Longueur de la Meilleure Tournée")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_alpha, use_container_width=True)

# 2. BETA EXPERIMENT
with exp_tabs[1]:
    st.markdown("### 🎯 Rôle de la Visibilité Heuristique : Impact du Paramètre β")
    st.markdown(
        "Lorsque $\\beta = 0$, les distances physiques ne guident plus les fourmis (seules les phéromones comptent, ce qui ralentit le démarrage). "
        "Pour des valeurs modérées de $\\beta$ ($2 \\le \\beta \\le 5$), la visibilité oriente efficacement la recherche vers les villes géographiquement proches."
    )

    if st.button("Lancer l'Analyse de Sensibilité sur β", key="btn_beta"):
        beta_values = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
        results = []
        progress = st.progress(0)

        for idx, b_val in enumerate(beta_values):
            aco = AntSystem(alpha=1.0, beta=b_val, rho=0.4, num_ants=15, max_iter=30, seed=42)
            res = aco.solve(inst)
            results.append({
                "Beta": b_val,
                "Longueur": res.best_tour.length,
                "Temps (ms)": res.execution_time_sec * 1000,
            })
            progress.progress((idx + 1) / len(beta_values))

        df_beta = pd.DataFrame(results)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_beta["Beta"], y=df_beta["Longueur"], mode="lines+markers", line=dict(color=THEME_COLORS["accent_orange"], width=3), marker=dict(size=8)))
        apply_dark_theme(fig, title=f"Sensibilité à β sur {inst.name}")
        fig.update_xaxes(title="Poids de la Visibilité Heuristique β")
        fig.update_yaxes(title="Longueur de la Meilleure Tournée")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_beta, use_container_width=True)

# 3. RHO EXPERIMENT
with exp_tabs[2]:
    st.markdown("### 💨 Vitesse d'Oubli : Impact du Taux d'Évaporation ρ")
    st.markdown(
        "Si $\\rho \\to 0$, les phéromones ne s'évaporent quasiment jamais (saturation rapide et stagnation). "
        "Si $\\rho \\to 1$, la mémoire est immédiatement effacée (perte totale d'apprentissage collectif)."
    )

    if st.button("Lancer l'Analyse de Sensibilité sur ρ", key="btn_rho"):
        rho_values = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
        results = []
        progress = st.progress(0)

        for idx, r_val in enumerate(rho_values):
            aco = AntSystem(alpha=1.0, beta=3.0, rho=r_val, num_ants=15, max_iter=30, seed=42)
            res = aco.solve(inst)
            results.append({
                "Rho (Evaporation)": r_val,
                "Longueur": res.best_tour.length,
                "Entropie": res.convergence_history[-1].entropy,
            })
            progress.progress((idx + 1) / len(rho_values))

        df_rho = pd.DataFrame(results)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_rho["Rho (Evaporation)"], y=df_rho["Longueur"], mode="lines+markers", line=dict(color=THEME_COLORS["accent_purple"], width=3), marker=dict(size=8)))
        apply_dark_theme(fig, title=f"Sensibilité à l'Évaporation ρ sur {inst.name}")
        fig.update_xaxes(title="Taux d'évaporation ρ")
        fig.update_yaxes(title="Longueur de la Meilleure Tournée")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_rho, use_container_width=True)

# 4. COLONY SIZE EXPERIMENT
with exp_tabs[3]:
    st.markdown("### 🐜 Taille de Colonie : Compromis Échantillonnage vs Temps")

    if st.button("Lancer l'Expérience sur la Taille de la Colonie (m)", key="btn_ants"):
        ants_values = [2, 5, 10, 20, 40, 60]
        results = []
        progress = st.progress(0)

        for idx, m_val in enumerate(ants_values):
            aco = AntSystem(alpha=1.0, beta=3.0, rho=0.4, num_ants=m_val, max_iter=25, seed=42)
            res = aco.solve(inst)
            results.append({
                "Fourmis (m)": m_val,
                "Longueur": res.best_tour.length,
                "Temps (ms)": res.execution_time_sec * 1000,
            })
            progress.progress((idx + 1) / len(ants_values))

        df_ants = pd.DataFrame(results)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=df_ants["Fourmis (m)"], y=df_ants["Longueur"], mode="lines+markers", line=dict(color=THEME_COLORS["primary_cyan"], width=3)))
            apply_dark_theme(fig1, title="Qualité de la Solution vs Taille Colonie")
            fig1.update_xaxes(title="Nombre de fourmis (m)")
            fig1.update_yaxes(title="Longueur Tournée")
            st.plotly_chart(fig1, use_container_width=True)

        with col_g2:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df_ants["Fourmis (m)"], y=df_ants["Temps (ms)"], marker_color=THEME_COLORS["accent_orange"]))
            apply_dark_theme(fig2, title="Temps de Calcul vs Taille Colonie")
            fig2.update_xaxes(title="Nombre de fourmis (m)")
            fig2.update_yaxes(title="Temps de Calcul (ms)")
            st.plotly_chart(fig2, use_container_width=True)

# 5. 2D HEATMAP (Alpha x Beta)
with exp_tabs[4]:
    st.markdown("### 🔥 Cartographie 2D d'Interaction : Alpha × Beta")
    st.markdown("Explore la surface de réponse multidimensionnelle pour révéler la zone optimale de compromis exploration / exploitation.")

    if st.button("Générer la Surface Thermique 2D (α × β)", key="btn_heatmap"):
        alphas = [0.5, 1.0, 1.5, 2.0, 3.0]
        betas = [1.0, 2.0, 3.0, 4.0, 6.0]
        z_grid = np.zeros((len(betas), len(alphas)))

        progress = st.progress(0)
        total_runs = len(alphas) * len(betas)
        count = 0

        for i, b in enumerate(betas):
            for j, a in enumerate(alphas):
                aco = AntSystem(alpha=a, beta=b, rho=0.4, num_ants=15, max_iter=20, seed=42)
                res = aco.solve(inst)
                z_grid[i, j] = res.best_tour.length
                count += 1
                progress.progress(count / total_runs)

        fig_heat = plot_sensitivity_heatmap(
            param_x_name="Paramètre Alpha (α)",
            x_vals=alphas,
            param_y_name="Paramètre Beta (β)",
            y_vals=betas,
            z_matrix=z_grid,
            title=f"Surface Thermique de Résolution sur {inst.name}",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
