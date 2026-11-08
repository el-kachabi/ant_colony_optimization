"""
Interactive Plotly Visualization Engine for Ant Colony Optimization & TSP.
Author: El Kachabi Badr-eddine
GitHub: https://github.com/el-kachabi
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config.settings import THEME_COLORS, PLOTLY_LAYOUT_DEFAULTS
from src.aco import ACOResult, IterationRecord, StepDecision
from src.tsp import City, Tour, TSPInstance


def apply_dark_theme(fig: go.Figure, title: str = "", height: int = 550) -> go.Figure:
    """Applies clean modern dark-mode aesthetic to a Plotly figure."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 17, "color": THEME_COLORS["text_main"], "family": "Inter, Roboto, sans-serif"},
            "x": 0.02,
            "y": 0.96,
        },
        paper_bgcolor=PLOTLY_LAYOUT_DEFAULTS["paper_bgcolor"],
        plot_bgcolor=PLOTLY_LAYOUT_DEFAULTS["plot_bgcolor"],
        font=PLOTLY_LAYOUT_DEFAULTS["font"],
        height=height,
        margin={"l": 45, "r": 35, "t": 60, "b": 45},
        legend={
            "bgcolor": "rgba(19, 27, 46, 0.8)",
            "bordercolor": THEME_COLORS["card_border"],
            "borderwidth": 1,
            "font": {"size": 12, "color": THEME_COLORS["text_main"]},
        },
        xaxis=PLOTLY_LAYOUT_DEFAULTS["xaxis"],
        yaxis=PLOTLY_LAYOUT_DEFAULTS["yaxis"],
    )
    return fig


def plot_tsp_instance(instance: TSPInstance, title: Optional[str] = None) -> go.Figure:
    """
    Renders 2D coordinates of cities as glowing interactive scatter nodes.
    """
    xs = [c.x for c in instance.cities]
    ys = [c.y for c in instance.cities]
    labels = [f"<b>{c.name}</b><br>ID: {c.id}<br>X: {c.x:.2f}<br>Y: {c.y:.2f}" for c in instance.cities]

    fig = go.Figure()

    # Cities scatter
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=[c.name for c in instance.cities],
            textposition="top center",
            textfont={"size": 11, "color": THEME_COLORS["text_main"]},
            hovertext=labels,
            hoverinfo="text",
            marker=dict(
                size=14,
                color=THEME_COLORS["primary_cyan"],
                line=dict(color="#FFFFFF", width=2),
                symbol="circle",
            ),
            name="Villes (Sommets)",
        )
    )

    chart_title = title or f"Réseau de Villes : {instance.name} ({instance.dimension} villes)"
    apply_dark_theme(fig, title=chart_title)
    fig.update_xaxes(title="Coordonnée X", range=[min(xs) - 8, max(xs) + 8])
    fig.update_yaxes(title="Coordonnée Y", range=[min(ys) - 8, max(ys) + 8], scaleanchor="x", scaleratio=1)
    return fig


def plot_tour(
    instance: TSPInstance,
    tour: Tour,
    title: Optional[str] = None,
    color: str = THEME_COLORS["primary_cyan"],
    show_best_optimum: bool = True,
) -> go.Figure:
    """
    Renders a closed Hamiltonian tour overlay on the TSP cities.
    """
    fig = plot_tsp_instance(instance, title=title)
    xs, ys = tour.to_coordinates(instance.cities)

    # Tour path lines
    step_labels = []
    for step, idx in enumerate(tour.path):
        next_idx = tour.path[(step + 1) % len(tour.path)]
        d = instance.distance_matrix[idx, next_idx]
        step_labels.append(f"Étape {step+1}: {instance.cities[idx].name} -> {instance.cities[next_idx].name} (dist: {d:.2f})")
    step_labels.append(step_labels[0])

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=3.5),
            hovertext=step_labels,
            hoverinfo="text",
            name=f"Tournée (L = {tour.length:.2f})",
        )
    )

    # Highlight starting city
    start_city = instance.cities[tour.path[0]]
    fig.add_trace(
        go.Scatter(
            x=[start_city.x],
            y=[start_city.y],
            mode="markers",
            marker=dict(
                size=18,
                color=THEME_COLORS["accent_orange"],
                symbol="star",
                line=dict(color="#FFFFFF", width=2),
            ),
            name=f"Départ ({start_city.name})",
        )
    )

    custom_title = title or f"Tournée Optimale : L = {tour.length:.2f}"
    if show_best_optimum and instance.optimal_known:
        gap = (tour.length - instance.optimal_known) / instance.optimal_known * 100
        custom_title += f" (Optimum: {instance.optimal_known:.2f} | Écart: {gap:+.2f}%)"

    apply_dark_theme(fig, title=custom_title)
    return fig


def plot_pheromone_network(
    instance: TSPInstance,
    pheromone_matrix: np.ndarray,
    best_tour: Optional[Tour] = None,
    threshold_percentile: float = 30.0,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Renders the complete pheromone field where edge width and color represent tau_ij.
    """
    n = instance.dimension
    fig = go.Figure()

    # Extract upper triangular edges
    edges = []
    weights = []
    for i in range(n):
        for j in range(i + 1, n):
            w = pheromone_matrix[i, j]
            edges.append((i, j, w))
            weights.append(w)

    weights = np.array(weights)
    min_w = np.min(weights)
    max_w = np.max(weights)
    cutoff = np.percentile(weights, threshold_percentile) if len(weights) > 0 else min_w

    # Group edges into buckets to avoid drawing thousands of individual traces
    num_buckets = 6
    buckets = np.linspace(cutoff, max_w + 1e-9, num_buckets + 1)
    # Colormap from transparent purple to burning orange/yellow
    palette = [
        "rgba(139, 92, 246, 0.20)",
        "rgba(79, 172, 254, 0.35)",
        "rgba(0, 242, 254, 0.50)",
        "rgba(255, 153, 0, 0.70)",
        "rgba(255, 102, 0, 0.85)",
        "rgba(255, 230, 0, 0.95)",
    ]

    for b in range(num_buckets):
        low = buckets[b]
        high = buckets[b + 1]
        bx, by = [], []
        hover_texts = []
        for i, j, w in edges:
            if low <= w < high:
                c1, c2 = instance.cities[i], instance.cities[j]
                bx.extend([c1.x, c2.x, None])
                by.extend([c1.y, c2.y, None])
                hover_texts.extend([f"Piste {c1.name}-{c2.name}<br>Tau: {w:.4f}", f"Piste {c1.name}-{c2.name}<br>Tau: {w:.4f}", None])

        if bx:
            width = 1.0 + (b * 0.9)
            fig.add_trace(
                go.Scatter(
                    x=bx,
                    y=by,
                    mode="lines",
                    line=dict(color=palette[b], width=width),
                    hoverinfo="text",
                    hovertext=hover_texts,
                    showlegend=False,
                )
            )

    # If best tour is provided, overlay it
    if best_tour:
        tx, ty = best_tour.to_coordinates(instance.cities)
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=ty,
                mode="lines",
                line=dict(color=THEME_COLORS["primary_cyan"], width=3.5),
                name=f"Meilleure Tournée (L={best_tour.length:.2f})",
            )
        )

    # Overlay city nodes
    cx = [c.x for c in instance.cities]
    cy = [c.y for c in instance.cities]
    fig.add_trace(
        go.Scatter(
            x=cx,
            y=cy,
            mode="markers+text",
            text=[c.name for c in instance.cities],
            textposition="top center",
            textfont={"size": 10, "color": THEME_COLORS["text_main"]},
            marker=dict(size=12, color=THEME_COLORS["card_bg"], line=dict(color=THEME_COLORS["primary_cyan"], width=2)),
            name="Villes",
        )
    )

    t = title or f"Champ de Phéromones τ_ij (Intensité & Concentration)"
    apply_dark_theme(fig, title=t)
    fig.update_xaxes(title="Coordonnée X")
    fig.update_yaxes(title="Coordonnée Y", scaleanchor="x", scaleratio=1)
    return fig


def plot_convergence_curves(
    convergence_history: List[IterationRecord],
    optimal_known: Optional[float] = None,
    title: str = "Dynamique de Convergence de la Colonie",
) -> go.Figure:
    """
    Renders multi-metric convergence: Best length, Mean length, and Worst length.
    """
    iters = [r.iteration for r in convergence_history]
    best_lens = [r.best_length for r in convergence_history]
    mean_lens = [r.mean_length for r in convergence_history]
    worst_lens = [r.worst_length for r in convergence_history]

    fig = go.Figure()

    # Dispersion ribbon (between worst and best)
    fig.add_trace(
        go.Scatter(
            x=iters + iters[::-1],
            y=worst_lens + best_lens[::-1],
            fill="toself",
            fillcolor="rgba(79, 172, 254, 0.08)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="Dispersion (Pire - Meilleur)",
        )
    )

    # Worst length
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=worst_lens,
            mode="lines",
            line=dict(color="#64748B", width=1.5, dash="dot"),
            name="Pire Tournée",
        )
    )

    # Mean length
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=mean_lens,
            mode="lines",
            line=dict(color=THEME_COLORS["accent_orange"], width=2.2, dash="dash"),
            name="Longueur Moyenne",
        )
    )

    # Best length (glowing cyan)
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=best_lens,
            mode="lines+markers",
            line=dict(color=THEME_COLORS["primary_cyan"], width=3.5),
            marker=dict(size=5, color=THEME_COLORS["primary_cyan"]),
            name="Meilleure Tournée (Best-So-Far)",
        )
    )

    # Theoretical optimal horizontal line if known
    if optimal_known:
        fig.add_trace(
            go.Scatter(
                x=[min(iters), max(iters)],
                y=[optimal_known, optimal_known],
                mode="lines",
                line=dict(color=THEME_COLORS["success_green"], width=2, dash="dashdot"),
                name=f"Optimum Référence ({optimal_known:.2f})",
            )
        )

    apply_dark_theme(fig, title=title)
    fig.update_xaxes(title="Itérations (Générations de fourmis)")
    fig.update_yaxes(title="Longueur de la Tournée")
    return fig


def plot_colony_entropy_and_pheromones(convergence_history: List[IterationRecord]) -> go.Figure:
    """
    Renders dual-axis plot: Normalized Shannon Entropy vs Mean Pheromone Concentration.
    """
    iters = [r.iteration for r in convergence_history]
    entropies = [r.entropy for r in convergence_history]
    tau_means = [r.pheromone_mean for r in convergence_history]
    tau_maxs = [r.pheromone_max for r in convergence_history]

    fig = go.Figure()

    # Shannon Entropy on left Y axis
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=entropies,
            mode="lines+markers",
            line=dict(color=THEME_COLORS["accent_purple"], width=2.8),
            marker=dict(size=4),
            name="Entropie de Shannon (Exploration)",
            yaxis="y1",
        )
    )

    # Pheromone Max on right Y axis
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=tau_maxs,
            mode="lines",
            line=dict(color=THEME_COLORS["accent_orange"], width=2.2, dash="dash"),
            name="Phéromone Max τ_max",
            yaxis="y2",
        )
    )

    # Pheromone Mean on right Y axis
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=tau_means,
            mode="lines",
            line=dict(color=THEME_COLORS["primary_cyan"], width=2.2),
            name="Phéromone Moyenne",
            yaxis="y2",
        )
    )

    apply_dark_theme(fig, title="Évolution de l'Entropie & Concentration des Phéromones")
    fig.update_layout(
        xaxis=dict(title="Itérations"),
        yaxis=dict(title="Entropie Normalisée H ∈ [0, 1]", range=[0, 1.05], side="left"),
        yaxis2=dict(
            title="Concentration de Phéromone τ",
            side="right",
            overlaying="y",
            showgrid=False,
            title_font=dict(color=THEME_COLORS["accent_orange"]),
            tickfont=dict(color=THEME_COLORS["accent_orange"]),
        ),
    )
    return fig


def plot_algorithm_comparison_barchart(df_comparison: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart comparing solvers by tour length and execution time.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_comparison["Algorithme"],
            y=df_comparison["Longueur Tournée"],
            name="Longueur Tournée (Plus bas = Meilleur)",
            marker_color=THEME_COLORS["primary_cyan"],
            text=df_comparison["Longueur Tournée"].apply(lambda v: f"{v:.1f}"),
            textposition="auto",
        )
    )

    apply_dark_theme(fig, title="Comparaison des Longueurs de Tournée Obtenues")
    fig.update_yaxes(title="Longueur de Tournée")
    return fig


def plot_step_simulation_view(
    instance: TSPInstance,
    step: StepDecision,
    current_path: List[int],
    pheromone_matrix: np.ndarray,
) -> go.Figure:
    """
    Renders fine-grained step-by-step decision view for Page 3 (Simulation):
    - Current ant position
    - Transition arrows to unvisited candidates with opacity proportional to probability
    - Chosen next city highlighted in neon emerald
    """
    fig = go.Figure()
    curr_city = instance.cities[step.current_city]

    # Draw completed segment so far
    if len(current_path) > 1:
        pxs = [instance.cities[i].x for i in current_path]
        pys = [instance.cities[i].y for i in current_path]
        fig.add_trace(
            go.Scatter(
                x=pxs,
                y=pys,
                mode="lines+markers",
                line=dict(color=THEME_COLORS["secondary_blue"], width=3),
                marker=dict(size=8, color=THEME_COLORS["secondary_blue"]),
                name="Chemin déjà parcouru",
            )
        )

    # Draw candidate transition vectors with line width & opacity proportional to probability
    for cand_idx, prob in zip(step.candidate_cities, step.probabilities):
        target = instance.cities[cand_idx]
        is_chosen = (cand_idx == step.chosen_city)
        line_color = THEME_COLORS["success_green"] if is_chosen else "rgba(148, 163, 184, 0.45)"
        width = 4.5 if is_chosen else max(1.0, prob * 6.0)

        fig.add_trace(
            go.Scatter(
                x=[curr_city.x, target.x],
                y=[curr_city.y, target.y],
                mode="lines",
                line=dict(color=line_color, width=width, dash="solid" if is_chosen else "dot"),
                hovertext=f"Candidat: {target.name}<br>P = {prob*100:.2f}%<br>Tau: {pheromone_matrix[curr_city.id, target.id]:.4f}",
                hoverinfo="text",
                showlegend=False,
            )
        )

    # Background cities
    cx = [c.x for c in instance.cities]
    cy = [c.y for c in instance.cities]
    fig.add_trace(
        go.Scatter(
            x=cx,
            y=cy,
            mode="markers+text",
            text=[c.name for c in instance.cities],
            textposition="top center",
            marker=dict(size=12, color=THEME_COLORS["card_bg"], line=dict(color=THEME_COLORS["text_muted"], width=1.5)),
            showlegend=False,
        )
    )

    # Highlight chosen city
    chosen_c = instance.cities[step.chosen_city]
    fig.add_trace(
        go.Scatter(
            x=[chosen_c.x],
            y=[chosen_c.y],
            mode="markers",
            marker=dict(size=18, color=THEME_COLORS["success_green"], symbol="circle-dot", line=dict(color="#FFFFFF", width=2)),
            name=f"Prochaine Ville Choisie ({chosen_c.name})",
        )
    )

    # Current ant position
    fig.add_trace(
        go.Scatter(
            x=[curr_city.x],
            y=[curr_city.y],
            mode="markers",
            marker=dict(size=20, color=THEME_COLORS["accent_orange"], symbol="diamond", line=dict(color="#FFFFFF", width=2.5)),
            name=f"Position Fourmi ({curr_city.name})",
        )
    )

    apply_dark_theme(fig, title=f"Décision de Déplacement : Fourmi à {curr_city.name} → Choix de {chosen_c.name}")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_step_probabilities_barchart(step: StepDecision, instance: TSPInstance) -> go.Figure:
    """
    Horizontal bar chart of candidate probabilities decomposed into tau and eta.
    """
    names = [instance.cities[cid].name for cid in step.candidate_cities]
    probs_pct = [p * 100 for p in step.probabilities]
    colors = [THEME_COLORS["success_green"] if cid == step.chosen_city else THEME_COLORS["secondary_blue"] for cid in step.candidate_cities]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=names,
            x=probs_pct,
            orientation="h",
            marker_color=colors,
            text=[f"{p:.1f}% {'(Sélectionnée)' if cid == step.chosen_city else ''}" for p, cid in zip(probs_pct, step.candidate_cities)],
            textposition="auto",
        )
    )

    apply_dark_theme(fig, title=f"Distribution de Probabilités P_{{ij}} (depuis {instance.cities[step.current_city].name})", height=350)
    fig.update_xaxes(title="Probabilité de Transition (%)", range=[0, 105])
    fig.update_yaxes(title="Villes Candidates")
    return fig


def plot_sensitivity_heatmap(
    param_x_name: str,
    x_vals: List[float],
    param_y_name: str,
    y_vals: List[float],
    z_matrix: np.ndarray,
    title: str = "Cartographie de Sensibilité Paramétrique",
) -> go.Figure:
    """
    2D Heatmap showing sensitivity surface for hyperparameter tuning.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=x_vals,
            y=y_vals,
            colorscale="Viridis_r",
            colorbar=dict(title="Longueur Tournée"),
            hoverongaps=False,
        )
    )

    apply_dark_theme(fig, title=title)
    fig.update_xaxes(title=param_x_name)
    fig.update_yaxes(title=param_y_name)
    return fig
