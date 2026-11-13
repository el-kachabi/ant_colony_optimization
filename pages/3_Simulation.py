"""
Microscopic Step-by-Step Ant Colony Simulation Page.
Animated visualization of ant movements and pheromone deposition.
"""

import streamlit as st

st.set_page_config(page_title="Simulation Microscopique | ACO Lab", page_icon="🔬", layout="wide")

from src.aco import AntSystem
from src.ui_components import inject_custom_css, initialize_session_state, render_page_header, render_sidebar
from src.visualization import plot_step_simulation_view, plot_step_probabilities_barchart

inject_custom_css()
initialize_session_state()
render_sidebar()

render_page_header(
    title="Simulation Microscopique Pas-à-Pas",
    subtitle="Observation visuelle du déplacement individuel d'une fourmi et de la règle de décision probabiliste",
    badge="MODULE 3 • SIMULATEUR",
)

inst = st.session_state.tsp_instance

# Run single iteration with step recording if not already recorded or on instance change
if (
    st.session_state.last_result is None
    or st.session_state.last_result.step_records is None
    or st.session_state.last_result.instance_name != inst.name
):
    with st.spinner("Génération des étapes microscopiques de simulation..."):
        sim_solver = AntSystem(
            alpha=st.session_state.aco_params["alpha"],
            beta=st.session_state.aco_params["beta"],
            rho=st.session_state.aco_params["rho"],
            num_ants=10,
            max_iter=5,
            seed=st.session_state.aco_params["seed"],
        )
        st.session_state.last_result = sim_solver.solve(inst, record_steps=True)
        st.session_state.simulation_step_idx = 0

step_records = st.session_state.last_result.step_records

if not step_records:
    st.warning("Aucune étape de simulation enregistrée pour cette instance.")
    st.stop()

total_steps = len(step_records)

# Stepper Navigation Bar
st.markdown("### 🕹️ Contrôleur Temporel de la Fourmi")
c_prev, c_slider, c_next, c_reset = st.columns([1, 4, 1, 1])

with c_prev:
    if st.button("◀ Précédent", use_container_width=True):
        st.session_state.simulation_step_idx = max(0, st.session_state.simulation_step_idx - 1)
        st.rerun()

with c_slider:
    step_idx = st.slider(
        "Étape de décision :",
        min_value=0,
        max_value=total_steps - 1,
        value=st.session_state.simulation_step_idx,
        format="Étape %d",
        key="sim_slider",
    )
    st.session_state.simulation_step_idx = step_idx

with c_next:
    if st.button("Suivant ▶", use_container_width=True):
        st.session_state.simulation_step_idx = min(total_steps - 1, st.session_state.simulation_step_idx + 1)
        st.rerun()

with c_reset:
    if st.button("⏮ Début", use_container_width=True):
        st.session_state.simulation_step_idx = 0
        st.rerun()

current_step = step_records[st.session_state.simulation_step_idx]

# Reconstruct path walked up to this step
path_so_far = [step_records[0].current_city]
for s in step_records[: st.session_state.simulation_step_idx + 1]:
    path_so_far.append(s.chosen_city)

# Status telemetry banner
curr_c = inst.cities[current_step.current_city]
next_c = inst.cities[current_step.chosen_city]
st.markdown(
    f"""
    <div class="custom-card" style="display: flex; justify-content: space-between; align-items: center; border-left: 4px solid #FF9900;">
        <div>
            <b>Position Actuelle :</b> <span style="color: #FF9900;">{curr_c.name}</span> (x={curr_c.x:.1f}, y={curr_c.y:.1f})<br>
            <b>Candidats Restants :</b> {len(current_step.candidate_cities)} villes non visitées
        </div>
        <div style="text-align: right;">
            <b>Choix Effectué :</b> <span style="color: #10B981; font-weight: 700;">{next_c.name}</span><br>
            <b>Probabilité Associée :</b> {dict(zip(current_step.candidate_cities, current_step.probabilities)).get(current_step.chosen_city, 0.0) * 100:.2f} %
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Visual Simulation Layout
col_sim_map, col_sim_chart = st.columns([1.2, 0.8])

with col_sim_map:
    st.markdown("#### 🗺️ Vecteurs de Transition Microscopiques")
    fig_map = plot_step_simulation_view(
        instance=inst,
        step=current_step,
        current_path=path_so_far,
        pheromone_matrix=st.session_state.last_result.final_pheromone_matrix,
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_sim_chart:
    st.markdown("#### 📊 Roue de Loterie (Probabilités de Transition)")
    fig_bar = plot_step_probabilities_barchart(current_step, inst)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(
        r"""
        <div style="background: rgba(15, 23, 42, 0.6); padding: 12px; border-radius: 8px; border: 1px solid #1E293B; font-size: 0.85rem; color: #94A3B8;">
            <b style="color: #00F2FE;">Interprétation du choix :</b><br>
            La fourmi pondère l'attractivité de chaque ville candidate selon $[\tau]^\alpha \cdot [\eta]^\beta$. 
            La barre verte représente la ville effectivement tirée au sort selon cette distribution.
        </div>
        """,
        unsafe_allow_html=True,
    )
