"""
About Page - Software Architecture, Scientific Specifications, and References.
"""

import streamlit as st

st.set_page_config(page_title="À Propos & Architecture | ACO Lab", page_icon="ℹ️", layout="wide")

from config.settings import PROJECT_NAME, VERSION
from src.ui_components import inject_custom_css, initialize_session_state, render_page_header, render_sidebar

inject_custom_css()
initialize_session_state()
render_sidebar()

render_page_header(
    title="Architecture Logicielle & Spécifications Scientifiques",
    subtitle="Ingénierie des algorithmes, analyse de complexité et bibliographie de référence",
    badge="MODULE 8 • À PROPOS",
)

col_mission, col_specs = st.columns([1, 1.2])

with col_mission:
    st.markdown("### 🏛️ Mission & Cadre du Projet")
    st.markdown(
        f"""
        <div class="custom-card" style="border-left: 4px solid #00F2FE;">
            <div style="font-size: 1.3rem; font-weight: 700; color: #F8FAFC;">{PROJECT_NAME} (v{VERSION})</div>
            <div style="color: #38BDF8; font-size: 0.95rem; margin-top: 4px;">Plateforme d'Optimisation Combinatoire & Métaheuristiques Bio-Inspirées</div>
            <div style="margin-top: 15px; color: #94A3B8; font-size: 0.9rem; line-height: 1.6;">
                Environnement scientifique interactif dédié à la simulation, à l'analyse mathématique et à la résolution haute performance du Problème du Voyageur de Commerce (TSP).
            </div>
            <div style="margin-top: 15px; color: #CBD5E1; font-size: 0.85rem; line-height: 1.6;">
                • Implémentation rigoureuse de 3 variantes ACO (AS, EAS, ACS).<br>
                • Benchmarks comparatifs face aux heuristiques classiques (2-Opt, Plus Proche Voisin, Brute Force).<br>
                • Calcul vectorisé haute vitesse sous NumPy sans dépendances lourdes.
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
