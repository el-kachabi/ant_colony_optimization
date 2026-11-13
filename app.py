"""
Main Application Entry Point - Ant Colony Optimization (ACO) & TSP Platform.
Interactive scientific laboratory for bio-inspired metaheuristics.
"""

import streamlit as st

st.set_page_config(
    page_title="Ant Colony Optimization & TSP Platform",
    page_icon="🐜",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config.settings import PRESET_INSTANCES
from src.tsp import TSPInstance
from src.ui_components import (
    inject_custom_css,
    initialize_session_state,
    render_page_header,
    render_sidebar,
)
from src.visualization import plot_tsp_instance

# Initialize CSS and session state
inject_custom_css()
initialize_session_state()
render_sidebar()

# Hero Header
render_page_header(
    title="Plateforme Interactive d'Optimisation par Colonie de Fourmis",
    subtitle="Exploration scientifique, modélisation mathématique et résolution du Problème du Voyageur de Commerce (TSP)",
    badge="ACO & TSP LAB • ÉDITION SCIENTIFIQUE",
)

# Top KPIs Row
inst = st.session_state.tsp_instance
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-val">{inst.dimension}</div>
            <div class="metric-lbl">Villes Actives ({inst.name.split('(')[0].strip()})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="metric-tile">
            <div class="metric-val" style="color: #FF9900;">3</div>
            <div class="metric-lbl">Variantes ACO (AS, EAS, ACS)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="metric-tile">
            <div class="metric-val" style="color: #10B981;">3</div>
            <div class="metric-lbl">Baselines (NN, 2-Opt, Brute Force)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    opt_text = f"{inst.optimal_known:.2f}" if inst.optimal_known else "Calculé via ACO"
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-val" style="color: #8B5CF6;">{opt_text}</div>
            <div class="metric-lbl">Optimum de Référence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Main 2-column layout: Network Preview & Feature Guide
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.markdown("### 🗺️ Carte du Réseau Actif")
    fig = plot_tsp_instance(inst)
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("### 🧭 Guide des Modules de la Plateforme")

    modules = [
        ("1️⃣ Tableau de Bord", "Vue d'ensemble exécutive, métriques instantanées et état de la colonie.", "pages/1_Dashboard.py"),
        ("2️⃣ Théorie & Fondements", "Cours complet, formules mathématiques en LaTeX et calculatrice didactique pas-à-pas.", "pages/2_Theorie_ACO.py"),
        ("3️⃣ Simulation Microscopique", "Visualisation animée du déplacement individuel des fourmis et de la roulette pondérée.", "pages/3_Simulation.py"),
        ("4️⃣ Résolution Interactive TSP", "Solveur complet avec choix des algorithmes, jauges de progression et tracé optimal.", "pages/4_Resolution_TSP.py"),
        ("5️⃣ Expériences & Sensibilité", "Analyse de l'impact des hyperparamètres (alpha, beta, rho, colonie) et heatmaps.", "pages/5_Experiences.py"),
        ("6️⃣ Banc d'Essai Comparatif", "Tournoi comparatif multi-algorithmes avec métriques réelles et benchmarks de référence.", "pages/6_Comparaison.py"),
        ("7️⃣ Cas d'Usage Industriels", "Applications concrètes : logistique, routage télécoms, PCB et ordonnancement.", "pages/7_Applications.py"),
        ("8️⃣ Architecture & À Propos", "Spécifications techniques, analyse de complexité et bibliographie scientifique.", "pages/8_A_propos.py"),
    ]

    for title, desc, path in modules:
        st.markdown(
            f"""
            <div class="custom-card" style="padding: 12px 16px; margin-bottom: 10px;">
                <div style="font-weight: 700; color: #00F2FE; font-size: 0.95rem;">{title}</div>
                <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 3px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Scientific Manifesto Box
st.markdown("---")
st.markdown(
    """
    <div style="background: rgba(19, 27, 46, 0.5); border: 1px solid #1E293B; border-radius: 10px; padding: 18px 24px; text-align: justify; font-size: 0.9rem; color: #94A3B8;">
        <span style="color: #00F2FE; font-weight: 700;">Engagement d'intégrité scientifique :</span> 
        Cette application est conçue pour l'ingénierie avancée et la recherche opérationnelle. 
        Chaque courbe de convergence, chaque matrice de probabilité et chaque dépôt de phéromone résulte d'une exécution algorithmique réelle en temps réel.
        Plateforme open-source dédiée aux métaheuristiques bio-inspirées et à l'optimisation combinatoire.
    </div>
    """,
    unsafe_allow_html=True,
)
