"""
About Page - Software Architecture, Scientific References, and Author Attribution.
Author: El Kachabi Badr-eddine
GitHub: https://github.com/el-kachabi
"""

import streamlit as st

st.set_page_config(page_title="À Propos & Architecture | ACO Lab", page_icon="ℹ️", layout="wide")

from config.settings import AUTHOR, GITHUB_PROFILE, GITHUB_REPO, VERSION
from src.ui_components import inject_custom_css, initialize_session_state, render_page_header, render_sidebar

inject_custom_css()
initialize_session_state()
render_sidebar()

render_page_header(
    title="Architecture Logicielle, Références & Auteur",
    subtitle="Spécifications d'ingénierie, analyse de complexité et bibliographie scientifique",
    badge="MODULE 8 • À PROPOS",
)

col_author, col_specs = st.columns([1, 1.2])

with col_author:
    st.markdown("### 👤 Auteur & Dépôt Officiel")
    st.markdown(
        f"""
        <div class="custom-card" style="border-left: 4px solid #00F2FE;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC;">{AUTHOR}</div>
            <div style="color: #38BDF8; font-size: 0.95rem; margin-top: 4px;">Ingénieur en Intelligence Artificielle & Systèmes Décisionnels</div>
            <div style="margin-top: 15px; color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                Projet conçu dans le cadre d'une suite avancée de métaheuristiques bio-inspirées et d'optimisation combinatoire sous Python.
            </div>
            <div style="margin-top: 18px;">
                <a href="{GITHUB_PROFILE}" target="_blank" style="background: rgba(0, 242, 254, 0.15); color: #00F2FE; border: 1px solid rgba(0, 242, 254, 0.4); padding: 8px 14px; border-radius: 6px; text-decoration: none; font-size: 0.85rem; font-weight: 600; display: inline-block;">
                    🔗 Visiter le Profil GitHub
                </a>
                <a href="{GITHUB_REPO}" target="_blank" style="background: rgba(148, 163, 184, 0.1); color: #E2E8F0; border: 1px solid #334155; padding: 8px 14px; border-radius: 6px; text-decoration: none; font-size: 0.85rem; font-weight: 600; display: inline-block; margin-left: 8px;">
                    📂 Dépôt GitHub du Projet
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_specs:
    st.markdown("### 🛠️ Piliers Techniques de la Plateforme")
    st.markdown(
        """
        <div class="custom-card">
            <ul style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.7; padding-left: 20px; margin: 0;">
                <li><b>Vectorisation NumPy :</b> Calcul accéléré des matrices de distance euclidienne et des probabilités de transition sans boucles Python imbriquées.</li>
                <li><b>Visualisation Réactive Plotly :</b> Graphiques interactifs accélérés par WebGL permettant le zoom, le survol et l'inspection individuelle des arêtes.</li>
                <li><b>Architecture Découplée :</b> Séparation stricte entre le moteur algorithmique mathématique (<code>src/</code>) et la couche de présentation (<code>pages/</code>).</li>
                <li><b>Reproductibilité Scientifique :</b> Gestion centralisée des générateurs pseudo-aléatoires (<code>np.random.default_rng(seed)</code>).</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown("### ⏱️ Analyse de Complexité Algorithmique")

st.markdown(
    """
    Pour une instance de $n$ villes avec une colonie de $m$ fourmis et un maximum de $T$ itérations :
    
    | Phase Algorithmique | Complexité Temporelle | Complexité Spatiale | Commentaire |
    | :--- | :---: | :---: | :--- |
    | **Matrice des Distances $D$** | $\\mathcal{O}(n^2)$ | $\\mathcal{O}(n^2)$ | Calculée une seule fois à l'initialisation |
    | **Construction de Tournées** | $\\mathcal{O}(T \\cdot m \\cdot n^2)$ | $\\mathcal{O}(m \\cdot n)$ | Chaque fourmi évalue $n - k$ candidats à chaque étape |
    | **Évaporation & Dépôt** | $\\mathcal{O}(T \\cdot n^2)$ | $\\mathcal{O}(n^2)$ | Mise à jour matricielle globale vectorisée |
    | **Complexité Globale ACO** | $\\mathbf{\\mathcal{O}(T \\cdot m \\cdot n^2)}$ | $\\mathbf{\\mathcal{O}(n^2)}$ | Linéaire par rapport à $T$ et $m$, quadratique en $n$ |
    | **Solveur Exact Brute Force** | $\\mathcal{O}((n-1)!)$ | $\\mathcal{O}(n)$ | Explosion factorielle prohibitive dès $n > 11$ |
    """
)

st.markdown("---")
st.markdown("### 📚 Bibliographie Scientifique de Référence")

st.markdown(
    """
    1. **Dorigo, M., Maniezzo, V., & Colorni, A. (1996).** *Ant system: optimization by a colony of cooperating agents.* IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 26(1), 29-41.
    2. **Dorigo, M., & Gambardella, L. M. (1997).** *Ant colony system: a cooperative learning approach to the traveling salesman problem.* IEEE Transactions on Evolutionary Computation, 1(1), 53-66.
    3. **Stützle, T., & Hoos, H. H. (2000).** *MAX-MIN Ant System.* Future Generation Computer Systems, 16(8), 889-914.
    4. **Goss, S., Aron, S., Deneubourg, J. L., & Pasteels, J. M. (1989).** *Self-organized shortcuts in the Argentine ant.* Naturwissenschaften, 76(12), 579-581.
    5. **Di Caro, G., & Dorigo, M. (1998).** *AntNet: distributed stigmergetic control for communications networks.* Journal of Artificial Intelligence Research, 9, 317-365.
    """
)
