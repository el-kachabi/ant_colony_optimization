"""
Reusable UI Components and Style Injectors for Streamlit ACO Platform.
Clean, modern, professional interface.
"""

import streamlit as st
from config.settings import (
    APP_TITLE,
    APP_SUBTITLE,
    PROJECT_NAME,
    THEME_COLORS,
    PRESET_INSTANCES,
)
from src.tsp import TSPInstance


def inject_custom_css():
    """Injects high-end modern dark glassmorphic styling into Streamlit."""
    st.markdown(
        """
        <style>
        /* Modern Typography & Dark Canvas */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* App Background */
        .stApp {
            background: linear-gradient(135deg, #0B0F19 0%, #0F172A 50%, #0B0F19 100%) !important;
            color: #F8FAFC !important;
        }

        /* Top Header Container */
        .header-box {
            background: linear-gradient(90deg, rgba(19, 27, 46, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(30, 41, 59, 0.9);
            border-left: 4px solid #00F2FE;
            border-radius: 12px;
            padding: 24px 30px;
            margin-bottom: 25px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }

        .header-title {
            font-size: 2.1rem;
            font-weight: 800;
            background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            padding-bottom: 6px;
        }

        .header-subtitle {
            color: #94A3B8;
            font-size: 1.05rem;
            margin: 0;
            font-weight: 400;
        }

        .header-meta {
            color: #38BDF8;
            font-size: 0.9rem;
            margin-top: 8px;
            font-weight: 500;
        }

        /* Custom Cards */
        .custom-card {
            background: rgba(19, 27, 46, 0.75);
            border: 1px solid #1E293B;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: 0 4px 15px -2px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .custom-card:hover {
            border-color: #00F2FE;
            transform: translateY(-2px);
        }

        /* Metric Tile */
        .metric-tile {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid #1E293B;
            border-radius: 10px;
            padding: 16px 20px;
            text-align: center;
        }

        .metric-val {
            font-size: 1.8rem;
            font-weight: 700;
            color: #00F2FE;
            margin: 0;
        }

        .metric-lbl {
            font-size: 0.85rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 4px;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0B0F19 !important;
            border-right: 1px solid #1E293B !important;
        }

        /* Sidebar Navigation Menu Links - Always Bright & High Contrast */
        [data-testid="stSidebarNav"] {
            padding-top: 15px !important;
        }

        [data-testid="stSidebarNav"] span,
        [data-testid="stSidebarNavLink"] span,
        [data-testid="stSidebarNavItems"] a span {
            color: #E2E8F0 !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            transition: color 0.15s ease !important;
        }

        [data-testid="stSidebarNavLink"] {
            padding: 8px 14px !important;
            border-radius: 8px !important;
            margin: 2px 8px !important;
            transition: all 0.2s ease !important;
        }

        [data-testid="stSidebarNavLink"]:hover {
            background-color: rgba(255, 255, 255, 0.06) !important;
        }

        [data-testid="stSidebarNavLink"]:hover span {
            color: #00F2FE !important;
        }

        [data-testid="stSidebarNavLink"][aria-current="page"] {
            background: rgba(0, 242, 254, 0.14) !important;
            border-left: 3px solid #00F2FE !important;
        }

        [data-testid="stSidebarNavLink"][aria-current="page"] span {
            color: #00F2FE !important;
            font-weight: 700 !important;
        }

        /* Sidebar text, headers & labels */
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: #F1F5F9 !important;
        }

        /* Top Header Styling (Visible Icons & Deploy / Settings Menu) */
        header[data-testid="stHeader"] {
            background-color: rgba(11, 15, 25, 0.85) !important;
            backdrop-filter: blur(10px) !important;
            border-bottom: 1px solid rgba(30, 41, 59, 0.5) !important;
        }

        header[data-testid="stHeader"] button,
        header[data-testid="stHeader"] svg,
        header[data-testid="stHeader"] [data-testid="stIconMaterial"] {
            color: #F8FAFC !important;
            fill: #F8FAFC !important;
        }

        /* Sidebar Collapse / Expand chevron button */
        [data-testid="stSidebarCollapseButton"] button svg,
        button[kind="header"] svg {
            color: #00F2FE !important;
            fill: #00F2FE !important;
        }

        /* Global Labels and Form Controls */
        label, .stSlider label, .stSelectbox label, .stRadio label, .stNumberInput label {
            color: #E2E8F0 !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        /* Primary Action Buttons */
        .stButton button[kind="primary"] {
            background: linear-gradient(90deg, #00F2FE 0%, #4FACFE 100%) !important;
            color: #0B0F19 !important;
            border: none !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            padding: 10px 24px !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 15px rgba(0, 242, 254, 0.25) !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }

        .stButton button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4) !important;
        }

        /* Secondary Buttons */
        .stButton button[kind="secondary"],
        .stButton button:not([kind="primary"]) {
            background-color: rgba(19, 27, 46, 0.8) !important;
            color: #F8FAFC !important;
            border: 1px solid #1E293B !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            transition: all 0.15s ease !important;
        }

        .stButton button[kind="secondary"]:hover,
        .stButton button:not([kind="primary"]):hover {
            border-color: #00F2FE !important;
            color: #00F2FE !important;
            background-color: rgba(0, 242, 254, 0.08) !important;
        }

        /* Dataframe & Tables styling */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            border: 1px solid #1E293B !important;
            border-radius: 10px !important;
            overflow: hidden !important;
            background-color: #131B2E !important;
        }

        /* Streamlit Slider Cyan Accent */
        div[data-baseweb="slider"] div[role="slider"] {
            background-color: #00F2FE !important;
            border: 2px solid #FFFFFF !important;
        }

        div[data-baseweb="slider"] div[data-testid="stSliderTickBar"] {
            background-color: #1E293B !important;
        }

        div[data-baseweb="slider"] div:nth-child(2) {
            background-color: #00F2FE !important;
        }

        /* Badges */
        .badge-cyan {
            background: rgba(0, 242, 254, 0.15);
            color: #00F2FE;
            border: 1px solid rgba(0, 242, 254, 0.3);
            border-radius: 6px;
            padding: 3px 8px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, badge: str = "ACO LAB"):
    """Renders consistent top banner across all pages."""
    st.markdown(
        f"""
        <div class="header-box">
            <span class="badge-cyan">{badge}</span>
            <h1 class="header-title">{title}</h1>
            <p class="header-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state():
    """Initializes global session state variables if not already set."""
    if "tsp_instance" not in st.session_state:
        st.session_state.tsp_instance = TSPInstance.load_preset("Micro_5_Villes")
    if "aco_params" not in st.session_state:
        st.session_state.aco_params = {
            "alpha": 1.0,
            "beta": 3.0,
            "rho": 0.40,
            "q": 100.0,
            "num_ants": 20,
            "max_iter": 40,
            "seed": 42,
            "elitist_weight": 5.0,
            "q0": 0.90,
            "xi": 0.10,
        }
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "simulation_step_idx" not in st.session_state:
        st.session_state.simulation_step_idx = 0


def render_sidebar():
    """Renders standard sidebar branding and navigation shortcuts."""
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 10px 0 15px 0; border-bottom: 1px solid #1E293B;">
                <h2 style="color: #00F2FE; margin: 0; font-size: 1.4rem;">🐜 ACO & TSP Lab</h2>
                <p style="color: #94A3B8; font-size: 0.85rem; margin: 2px 0 0 0;">Optimisation par Colonie de Fourmis</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🗺️ Instance Active")
        preset_options = list(PRESET_INSTANCES.keys())
        current_name = "Micro_5_Villes"
        if hasattr(st.session_state, "tsp_instance") and st.session_state.tsp_instance:
            for k, p in PRESET_INSTANCES.items():
                if p["name"] == st.session_state.tsp_instance.name or k == st.session_state.tsp_instance.name:
                    current_name = k
                    break

        selected_preset = st.selectbox(
            "Choisir l'instance :",
            options=preset_options,
            index=preset_options.index(current_name) if current_name in preset_options else 0,
            format_func=lambda k: PRESET_INSTANCES[k]["name"],
            key="sidebar_preset_selector",
        )

        if st.sidebar.button("🔄 Charger l'instance", use_container_width=True):
            st.session_state.tsp_instance = TSPInstance.load_preset(selected_preset)
            st.session_state.last_result = None
            st.rerun()

        inst = st.session_state.tsp_instance
        st.markdown(
            f"""
            <div style="background: rgba(19, 27, 46, 0.6); padding: 12px; border-radius: 8px; border: 1px solid #1E293B; margin-top: 10px; font-size: 0.85rem;">
                <div><b>Villes :</b> {inst.dimension}</div>
                <div><b>Optimum connu :</b> {f"{inst.optimal_known:.2f}" if inst.optimal_known else "Inconnu"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; color: #64748B; text-align: center;">
                <b>ACO & TSP Lab</b> • Optimisation Bio-Inspirée<br>
                Résolution du Problème du Voyageur de Commerce
            </div>
            """,
            unsafe_allow_html=True,
        )
