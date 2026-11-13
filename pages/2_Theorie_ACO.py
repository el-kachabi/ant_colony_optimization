"""
Theoretical Foundations & Interactive Step-by-Step Mathematical Calculator.
Detailed equations, transition rules, and step-by-step probability decomposition.
"""

import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="Théorie & Fondements | ACO Lab", page_icon="📚", layout="wide")

from src.tsp import TSPInstance
from src.ui_components import inject_custom_css, initialize_session_state, render_page_header, render_sidebar

inject_custom_css()
initialize_session_state()
render_sidebar()

render_page_header(
    title="Fondements Théoriques & Modélisation Mathématique",
    subtitle="Inspiration biologique, formulations rigoureuses et calculatrice numérique pas-à-pas",
    badge="MODULE 2 • THÉORIE & CALCULS",
)

# Tabs: Biological inspiration / Mathematical Formulations / Numerical Calculator
tab_bio, tab_math, tab_calc = st.tabs([
    "🐜 1. Inspiration Biologique & Stigmergie",
    "📐 2. Formulations Mathématiques Exactes",
    "🧮 3. Calculatrice Didactique Pas-à-Pas",
])

with tab_bio:
    st.markdown("### 🌿 Du Comportement Naturel des Fourmis à l'Algorithme")
    col_b1, col_b2 = st.columns([1.1, 0.9])

    with col_b1:
        st.markdown(
            r"""
            Dans la nature, une fourmi isolée dispose de capacités cognitives et visuelles extrêmement restreintes. 
            Pourtant, à l'échelle de la colonie, les fourmis accomplissent une prouesse collective remarquable : 
            **identifier systématiquement le chemin le plus court reliant leur nid à une source de nourriture**.

            #### 🧪 L'Expérience du Double Pont (Goss et al., 1989)
            Des chercheurs en éthologie ont placé une colonie devant deux branches reliant le nid à de la nourriture : 
            une branche courte ($A$) et une branche longue ($B$).
            1. **Phase initiale aléatoire :** Au départ, aucune phéromone n'est déposée. Les fourmis choisissent chaque branche avec une probabilité égale ($50\% / 50\%$).
            2. **Avantage temporel de la branche courte :** Les fourmis empruntant la branche courte atteignent la nourriture plus rapidement et font l'aller-retour en premier.
            3. **Boucle de rétroaction positive :** En parcourant la branche courte deux fois plus vite, la concentration en phéromone y augmente plus rapidement.
            4. **Stigmergie et convergence collective :** Les fourmis suivantes détectent cette trace chimique plus intense et s'y engagent préférentiellement. Rapidement, toute la colonie converge vers la branche courte.
            """
        )

    with col_b2:
        st.markdown(
            """
            <div class="custom-card">
                <h4 style="color: #00F2FE; margin-top: 0;">Concepts Clés de la Métaheuristique</h4>
                <ul style="color: #CBD5E1; font-size: 0.9rem; line-height: 1.7;">
                    <li><b>Stigmergie :</b> Forme de communication indirecte via modification de l'environnement physique (dépôt de phéromones).</li>
                    <li><b>Mémoire Collective :</b> La matrice des phéromones mémorise l'expérience accumulée par l'ensemble des agents artificiels.</li>
                    <li><b>Visibilité Heuristique :</b> Connaissance locale a priori (attractivité inversement proportionnelle à la distance).</li>
                    <li><b>Évaporation :</b> Oubli progressif évitant le piégeage prématuré dans des optima locaux sous-optimaux.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_math:
    st.markdown("### 📐 Formulation Mathématique du Problème du Voyageur de Commerce (TSP)")

    st.markdown(
        r"""
        Soit un graphe complet non orienté $G = (V, E)$ où $V = \{0, 1, \dots, n-1\}$ représente les $n$ villes, 
        et chaque arête $(i, j) \in E$ est pondérée par la distance euclidienne $d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$.
        
        L'objectif est de trouver un cycle hamiltonien $\pi$ qui minimise la longueur totale :
        $$\min_{\pi} L(\pi) = \sum_{i=0}^{n-2} d_{\pi_i, \pi_{i+1}} + d_{\pi_{n-1}, \pi_0}$$
        """
    )

    st.markdown("---")
    st.markdown("### ⚙️ Les Équations Fondamentales d'Ant System (Dorigo, 1996)")

    c_m1, c_m2 = st.columns(2)

    with c_m1:
        st.markdown(
            r"""
            #### 1. Règle Proportionnelle de Transition
            Pour une fourmi $k$ située à la ville $i$, la probabilité de se rendre à la ville non encore visitée $j \in N_i^k$ est :
            $$P_{ij}^k(t) = \frac{[\tau_{ij}(t)]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in N_i^k} [\tau_{il}(t)]^\alpha \cdot [\eta_{il}]^\beta}$$

            * **$\eta_{ij} = \frac{1}{d_{ij}}$** : Visibilité heuristique locale (gloutonnerie).
            * **$\tau_{ij}(t)$** : Intensité de la trace de phéromone accumulée.
            * **$\alpha \ge 0$** : Poids accordé à l'historique collectif (exploitation).
            * **$\beta \ge 0$** : Poids accordé à la proximité géographique (exploration gloutonne).
            """
        )

    with c_m2:
        st.markdown(
            r"""
            #### 2. Évaporation et Mise à Jour Globale
            À la fin de chaque génération, les pistes s'évaporent au taux $\rho \in ]0, 1[$ et les $m$ fourmis déposent leur phéromone :
            $$\tau_{ij}(t+1) = (1 - \rho)\,\tau_{ij}(t) + \sum_{k=1}^m \Delta \tau_{ij}^k(t)$$
            
            $$\Delta \tau_{ij}^k(t) = \begin{cases} \dfrac{Q}{L_k} & \text{si } (i, j) \in \text{tournée } k \\ 0 & \text{sinon} \end{cases}$$

            * **$Q > 0$** : Quantité totale de phéromone déposée par une fourmi.
            * **$L_k$** : Longueur totale de la tournée achevée par la fourmi $k$.
            * **$\rho \in ]0, 1[$** : Taux d'évaporation contrôlant la persistance de la mémoire.
            """
        )

with tab_calc:
    st.markdown("### 🧮 Calculatrice Didactique Pas-à-Pas")
    st.markdown(
        "Ce module décompose en direct **chaque opération mathématique élémentaire** sur la micro-instance didactique (5 villes), "
        "permettant de vérifier manuellement le calcul exact des probabilités de transition."
    )

    inst_micro = TSPInstance.load_preset("Micro_5_Villes")

    col_calc_ctrl1, col_calc_ctrl2, col_calc_ctrl3 = st.columns(3)
    with col_calc_ctrl1:
        current_node = st.selectbox(
            "Ville actuelle de la fourmi :",
            options=range(inst_micro.dimension),
            format_func=lambda i: f"{inst_micro.cities[i].name} (x={inst_micro.cities[i].x:.0f}, y={inst_micro.cities[i].y:.0f})",
            index=0,
        )
    with col_calc_ctrl2:
        calc_alpha = st.slider("Paramètre α :", min_value=0.0, max_value=3.0, value=1.0, step=0.5, key="calc_alpha")
    with col_calc_ctrl3:
        calc_beta = st.slider("Paramètre β :", min_value=0.0, max_value=5.0, value=3.0, step=0.5, key="calc_beta")

    unvisited_nodes = [j for j in range(inst_micro.dimension) if j != current_node]

    # Assume baseline pheromone tau0 = 1.0 for manual demonstration
    tau_demo = 1.0
    records = []
    denominators = 0.0

    for cand in unvisited_nodes:
        dist = inst_micro.distance_matrix[current_node, cand]
        eta = 1.0 / dist
        tau_alpha = tau_demo ** calc_alpha
        eta_beta = eta ** calc_beta
        numerator = tau_alpha * eta_beta
        denominators += numerator
        records.append({
            "Ville Voisine": inst_micro.cities[cand].name,
            "Distance d_ij": round(dist, 2),
            "Visibilité η_ij = 1/d": round(eta, 4),
            "τ^α": round(tau_alpha, 4),
            "η^β": round(eta_beta, 6),
            "Numérateur w_ij": round(numerator, 6),
        })

    for r in records:
        prob = r["Numérateur w_ij"] / denominators if denominators > 0 else 0.0
        r["Probabilité P_ij"] = f"{prob * 100:.2f} %"

    df_calc = pd.DataFrame(records)

    st.markdown(f"#### 📊 Tableau de Décomposition Pas-à-Pas (depuis {inst_micro.cities[current_node].name}) :")
    st.dataframe(df_calc, use_container_width=True)

    st.markdown(
        f"""
        <div class="custom-card" style="border-left: 4px solid #10B981; font-size: 0.95rem;">
            <b>Formule appliquée :</b><br>
            Somme des numérateurs $\\sum w_{{il}} =$ <code>{denominators:.6f}</code><br>
            Chaque probabilité est calculée par : $P_{{ij}} = \\frac{{w_{{ij}}}}{{{denominators:.6f}}}$. 
            La somme totale des probabilités vaut rigoureusement <b>100.00 %</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
