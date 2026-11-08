"""
Industrial Applications & Real-World Use Cases of Ant Colony Optimization.
Author: El Kachabi Badr-eddine
GitHub: https://github.com/el-kachabi
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Applications Industrielles | ACO Lab", page_icon="🏭", layout="wide")

from src.ui_components import inject_custom_css, initialize_session_state, render_page_header, render_sidebar
from src.visualization import apply_dark_theme
from config.settings import THEME_COLORS

inject_custom_css()
initialize_session_state()
render_sidebar()

render_page_header(
    title="Applications Pratiques & Cas d'Usage Industriels",
    subtitle="De la théorie aux systèmes cyber-physiques : logistique, télécoms, micro-électronique et ordonnancement",
    badge="MODULE 7 • APPLICATIONS",
)

st.markdown(
    """
    Bien que souvent introduit à travers le Problème du Voyageur de Commerce (TSP), le paradigme d'optimisation par colonie de fourmis 
    s'applique avec succès à un large éventail de défis industriels complexes caractérisés par des **espaces de recherche combinatoires massifs**.
    """
)

app_tabs = st.tabs([
    "🚚 1. Logistique & Dernier Kilomètre",
    "📡 2. Routage Télécoms (AntNet)",
    "⚡ 3. Circuits Imprimés (PCB Drilling)",
    "⏱️ 4. Ordonnancement Industriel (JSSP)",
])

# 1. LOGISTIQUE
with app_tabs[0]:
    st.markdown("### 🚚 Optimisation des Tournées de Véhicules (VRP / Last-Mile Delivery)")
    c_l1, c_l2 = st.columns([1.1, 0.9])

    with c_l1:
        st.markdown(
            """
            Dans la logistique de distribution urbaine, le problème s'étend au **Vehicle Routing Problem (VRP)** : 
            desservir une flotte de clients avec plusieurs camions au départ d'un dépôt central sous contraintes de capacité ($CVRP$) 
            et de fenêtres horaires ($VRPTW$).

            #### Impact Concret de l'Optimisation par Fourmis :
            * **Réduction de 15% à 25% des distances kilométriques parcourues.**
            * **Diminution directe des émissions de CO₂ et de la consommation de carburant.**
            * **Adaptation dynamique aux aléas de circulation :** des fourmis numériques patrouillent en continu pour réacheminer les livreurs lors d'embouteillages.
            """
        )

    with c_l2:
        # Mini visual demonstration of multi-depot routing
        rng = np.random.default_rng(101)
        depot = (50.0, 50.0)
        clients_x = rng.uniform(15, 85, size=15)
        clients_y = rng.uniform(15, 85, size=15)

        fig_vrp = go.Figure()
        # Draw Depot
        fig_vrp.add_trace(go.Scatter(x=[depot[0]], y=[depot[1]], mode="markers+text", text=["Dépôt Central"], marker=dict(size=18, color=THEME_COLORS["accent_orange"], symbol="square"), name="Dépôt"))
        # Draw Clients
        fig_vrp.add_trace(go.Scatter(x=clients_x, y=clients_y, mode="markers", marker=dict(size=10, color=THEME_COLORS["primary_cyan"]), name="Points de Livraison"))

        apply_dark_theme(fig_vrp, title="Flotte de Livraison Multi-Véhicules", height=380)
        st.plotly_chart(fig_vrp, use_container_width=True)

# 2. ANTNET
with app_tabs[1]:
    st.markdown("### 📡 Routage Adaptatif dans les Réseaux Télécoms (AntNet)")
    st.markdown(
        """
        Développé par **Gianni Di Caro et Marco Dorigo (1998)**, le protocole **AntNet** transpose l'ACO aux réseaux de paquets (IP, 5G, réseaux ad-hoc).
        
        * **Fourmis Aller (Forward Ants) :** De petits paquets explorateurs circulent de routeur en routeur vers une destination, mesurant les temps de latence réels et les files d'attente.
        * **Fourmis Retour (Backward Ants) :** Une fois la destination atteinte, la fourmi revient vers la source en renforçant les tables de routage des nœuds intermédiaires avec de la phéromone virtuelle.
        * **Avantage majeur :** Absence totale d'organe de contrôle centralisé ; robustesse naturelle face aux pannes matérielles et à la congestion.
        """
    )

# 3. PCB DRILLING
with app_tabs[2]:
    st.markdown("### ⚡ Perçage Haute Précision de Circuits Imprimés (PCB Drilling)")
    st.markdown(
        """
        La fabrication industrielle de cartes électroniques modernes nécessite le perçage de plusieurs milliers de trous de traversée (*vias*). 
        Le temps de déplacement à vide de la tête laser entre deux perçages successifs constitue le principal goulot d'étranglement de la chaîne d'assemblage.

        En modélisant la position des trous sous forme d'un TSP géant ($N > 10\\,000$), l'ACO permet d'ordonner les micro-perçages pour **minimiser l'usure mécanique des servomoteurs et accroître les cadences de production de 30%**.
        """
    )

# 4. ORDONNANCEMENT (JSSP)
with app_tabs[3]:
    st.markdown("### ⏱️ Ordonnancement d'Ateliers (Job Shop Scheduling Problem - JSSP)")
    st.markdown(
        """
        Dans une usine manufacturière, $n$ pièces doivent subir une séquence d'opérations sur $m$ machines distinctes. 
        L'objectif est d'ordonner les tâches pour minimiser la durée totale de fabrication (*Makespan* $C_{\\max}$).

        L'ACO modélise ce problème sous la forme d'un graphe disjonctif où les phéromones mémorisent l'ordre de passage le plus fluide, 
        évitant les temps morts et les congestions de machines.
        """
    )
