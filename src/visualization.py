"""
Visualisations Interactives Plotly - Ant Colony Optimization (ACO) & TSP.
Contient les 2 graphiques essentiels (Carte du réseau & Courbe de convergence),
entièrement commentés pour expliquer chaque couche graphique.
"""

from typing import List, Optional
import numpy as np
import plotly.graph_objects as go
from src.core_aco import TSPInstance


def plot_tsp_solution(
    instance: TSPInstance,
    best_path: List[int],
    pheromone_matrix: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    Graphique 1 : Carte 2D affichant :
    - Couche 1 (Arrière-plan orange) : L'intensité des pistes de phéromones tau_ij sur les arêtes.
    - Couche 2 (Ligne continue cyan) : Le meilleur cycle hamiltonien découvert par la colonie.
    - Couche 3 (Sommets)             : Les villes (Ville 0 de départ mise en évidence en vert).
    """
    fig = go.Figure()
    n = instance.dimension
    cities = instance.cities

    # --------------------------------------------------------------------------
    # COUCHE 1 : Affichage des traces de phéromones en arrière-plan
    # --------------------------------------------------------------------------
    if pheromone_matrix is not None:
        max_tau = float(np.max(pheromone_matrix)) if np.max(pheromone_matrix) > 0 else 1.0
        first_phero = True
        for i in range(n):
            for j in range(i + 1, n):
                # Normalisation de la phéromone entre 0 et 1 par rapport à l'arête la plus marquée
                ratio = float(pheromone_matrix[i, j]) / max_tau
                # On filtre les traces inférieures à 15% pour garder une carte propre et lisible
                if ratio > 0.15:
                    fig.add_trace(
                        go.Scatter(
                            x=[cities[i].x, cities[j].x],
                            y=[cities[i].y, cities[j].y],
                            mode="lines",
                            name="Pistes de Phéromones (τ)",
                            # Épaisseur et opacité proportionnelles à la concentration en phéromone
                            line=dict(color=f"rgba(255, 153, 0, {min(0.85, ratio * 0.7):.2f})", width=1 + 4 * ratio),
                            hoverinfo="skip",
                            showlegend=first_phero,  # N'affiche l'entrée dans la légende qu'une seule fois
                        )
                    )
                    first_phero = False

    # --------------------------------------------------------------------------
    # COUCHE 2 : Tracé du meilleur circuit hamiltonien découvert par les fourmis
    # --------------------------------------------------------------------------
    if best_path:
        # On ajoute best_path[0] à la fin de la liste pour fermer visuellement la boucle
        closed_path = best_path + [best_path[0]]
        xs = [cities[idx].x for idx in closed_path]
        ys = [cities[idx].y for idx in closed_path]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name="Meilleure Tournée ACO",
                line=dict(color="#00F2FE", width=3),
                marker=dict(size=6, color="#00F2FE"),
                hoverinfo="skip",
            )
        )

    # --------------------------------------------------------------------------
    # COUCHE 3 : Affichage des sommets des villes et de leurs étiquettes (V0, V1...)
    # --------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=[c.x for c in cities],
            y=[c.y for c in cities],
            mode="markers+text",
            name="Villes",
            text=[f"V{c.id}" for c in cities],
            textposition="top center",
            marker=dict(
                size=[16 if c.id == 0 else 12 for c in cities],
                color=["#10B981" if c.id == 0 else "#1E293B" for c in cities],
                line=dict(width=2, color="#00F2FE"),
            ),
            hovertemplate="<b>%{text}</b><br>x = %{x:.1f}, y = %{y:.1f}<extra></extra>",
        )
    )

    # Configuration de la mise en page (titre à gauche, légende encadrée à droite sans chevauchement)
    fig.update_layout(
        title=dict(
            text="🗺️ Carte du Réseau : Pistes de Phéromones & Meilleur Circuit ACO",
            x=0.0,
            xanchor="left",
            y=0.96,
            yanchor="top",
        ),
        template="plotly_dark",
        height=520,
        margin=dict(l=40, r=40, t=90, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        xaxis=dict(title="Coordonnée X", range=[0, 105], showgrid=True, gridcolor="#1E293B"),
        yaxis=dict(title="Coordonnée Y", range=[0, 105], showgrid=True, gridcolor="#1E293B", scaleanchor="x"),
    )
    return fig


def plot_convergence(
    history_best: List[float],
    history_mean: List[float],
    greedy_length: Optional[float] = None,
    optimal_length: Optional[float] = None,
) -> go.Figure:
    """
    Graphique 2 : Courbe de convergence illustrant l'apprentissage de la colonie :
    - Courbe pointillée grise : Distance moyenne des m fourmis à chaque itération.
    - Courbe continue cyan    : Meilleure distance trouvée depuis le début (Best-so-far).
    - Lignes horizontales     : Seuils de référence (Algorithme Glouton et Optimum connu).
    """
    iters = list(range(1, len(history_best) + 1))
    fig = go.Figure()

    # 1. Évolution de la distance moyenne de la population de fourmis
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=history_mean,
            mode="lines",
            name="Distance Moyenne de la Colonie",
            line=dict(color="#94A3B8", width=1.5, dash="dot"),
        )
    )

    # 2. Évolution de la meilleure distance globale découverte (monotone décroissante)
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=history_best,
            mode="lines+markers",
            name="Meilleure Distance ACO",
            line=dict(color="#00F2FE", width=3),
            marker=dict(size=5),
        )
    )

    # 3. Ligne horizontale de comparaison avec l'heuristique Gloutonne (Plus Proche Voisin)
    if greedy_length is not None:
        fig.add_hline(
            y=greedy_length,
            line_dash="dash",
            line_color="#FF9900",
            annotation_text=f"Référence Glouton ({greedy_length:.2f})",
            annotation_position="top right",
        )

    # 4. Ligne horizontale indiquant l'optimum mathématique connu
    if optimal_length is not None:
        fig.add_hline(
            y=optimal_length,
            line_dash="solid",
            line_color="#10B981",
            annotation_text=f"Optimum Connu ({optimal_length:.2f})",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title=dict(
            text="📉 Dynamique de Convergence de la Colonie par Itération",
            x=0.0,
            xanchor="left",
            y=0.96,
            yanchor="top",
        ),
        template="plotly_dark",
        height=460,
        margin=dict(l=40, r=40, t=90, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(15, 23, 42, 0.75)",
            bordercolor="#334155",
            borderwidth=1,
        ),
        xaxis=dict(title="Itération (Génération)", showgrid=True, gridcolor="#1E293B"),
        yaxis=dict(title="Longueur de la Tournée", showgrid=True, gridcolor="#1E293B"),
    )
    return fig
