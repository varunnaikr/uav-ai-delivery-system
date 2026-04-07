# app.py

import streamlit as st
import matplotlib.pyplot as plt

from optimizer import generate_routes, mpdd_best, milp_best, pareto, evaluate, simulate_route
from agent import decide_best_algorithm
from rag import load_knowledge, build_index, retrieve
from data import weights, points


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="UAV AI System", layout="wide")

st.title("🚁 AI-Powered UAV Medical Delivery System")
st.markdown("### Agentic AI + RAG + Multi-Objective Optimization")

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("⚙️ Input Parameters")

battery_level = st.sidebar.slider("Battery Level (%)", 0, 100, 50)
is_emergency = st.sidebar.checkbox("🚨 Emergency Delivery")
num_routes = st.sidebar.slider("Number of Routes", 500, 5000, 2000)

load_level = sum(weights.values())

# Keep latest simulation results across Streamlit reruns (e.g., dropdown changes)
if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None

# ----------------------------
# BUTTON
# ----------------------------
if st.sidebar.button("🚀 Run Simulation"):

    # ----------------------------
    # RAG SYSTEM
    # ----------------------------
    docs = load_knowledge()
    index, embeddings = build_index(docs)

    query = f"battery {battery_level} emergency {is_emergency} load {load_level}"
    retrieved = retrieve(query, docs, index)

    # ----------------------------
    # GENERATE ROUTES
    # ----------------------------
    routes = generate_routes(num_routes)

    # ----------------------------
    # RUN ALL ALGORITHMS
    # ----------------------------
    mpdd_route = mpdd_best(routes)
    milp_route = milp_best(routes)
    pareto_routes = pareto(routes)
    nsga_route = pareto_routes[0][0] if pareto_routes else None

    # ----------------------------
    # METRICS + PERFORMANCE-BASED DECISION
    # ----------------------------
    mpdd_vals = evaluate(mpdd_route)
    milp_vals = evaluate(milp_route)
    nsga_vals = evaluate(nsga_route)

    E1, F1, T1 = mpdd_vals
    E2, F2, T2 = milp_vals
    E3, F3, T3 = nsga_vals

    if is_emergency:
        decision = "MILP"
    else:
        decision = decide_best_algorithm(mpdd_vals, milp_vals, nsga_vals)

    # Save results so UI interactions do not reset the app to the initial state
    st.session_state.simulation_results = {
        "retrieved": retrieved,
        "decision": decision,
        "mpdd_route": mpdd_route,
        "milp_route": milp_route,
        "nsga_route": nsga_route,
        "metrics": {
            "mpdd": mpdd_vals,
            "milp": milp_vals,
            "nsga": nsga_vals,
        },
    }

# Render latest results (if present) even after Streamlit reruns
if st.session_state.simulation_results:
    results = st.session_state.simulation_results
    retrieved = results["retrieved"]
    decision = results["decision"]
    mpdd_route = results["mpdd_route"]
    milp_route = results["milp_route"]
    nsga_route = results["nsga_route"]
    mpdd_vals = results["metrics"]["mpdd"]
    milp_vals = results["metrics"]["milp"]
    nsga_vals = results["metrics"]["nsga"]

    E1, F1, T1 = mpdd_vals
    E2, F2, T2 = milp_vals
    E3, F3, T3 = nsga_vals

    # ----------------------------
    # DISPLAY SECTIONS
    # ----------------------------
    col1, col2 = st.columns(2)

    # 📚 RAG OUTPUT
    with col1:
        st.subheader("📚 Retrieved Knowledge")
        for r in retrieved:
            st.success(r)

    # 🧠 AGENT
    with col2:
        st.subheader("🧠 Agent Decision")
        st.info(f"Selected Algorithm: **{decision}**")

    # ----------------------------
    # METRICS TABLE
    # ----------------------------
    st.subheader("📊 Performance Comparison")

    st.table({
        "Algorithm": ["MPDD", "MILP", "NSGA"],
        "Energy": [round(E1,2), round(E2,2), round(E3,2)],
        "Fatigue": [round(F1,2), round(F2,2), round(F3,2)],
        "Time": [round(T1,2), round(T2,2), round(T3,2)]
    })

    # ----------------------------
    # METRIC GRAPHS
    # ----------------------------
    st.subheader("📈 Energy, Fatigue, and Time Graphs")

    algorithms = ["MPDD", "MILP", "NSGA"]
    energy_values = [E1, E2, E3]
    fatigue_values = [F1, F2, F3]
    time_values = [T1, T2, T3]

    def plot_metric(metric_values, metric_title, color):
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(algorithms, metric_values, color=color)
        ax.set_title(metric_title)
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.3)

        # Add values above bars
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.02 * max(metric_values) if max(metric_values) > 0 else 0.01),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        return fig

    g1, g2, g3 = st.columns(3)

    with g1:
        st.pyplot(plot_metric(energy_values, "Energy", "#1f77b4"))
    with g2:
        st.pyplot(plot_metric(fatigue_values, "Fatigue", "#ff7f0e"))
    with g3:
        st.pyplot(plot_metric(time_values, "Time", "#2ca02c"))

    # ----------------------------
    # PLOTTING FUNCTION
    # ----------------------------
    def plot_route(route, title):
        if route is None:
            st.write("No valid route")
            return

        fig, ax = plt.subplots()

        x = [points[p][0] for p in route]
        y = [points[p][1] for p in route]

        # Plot all points and labels
        for name, (px, py) in points.items():
            ax.scatter(px, py, color="black", s=20)
            ax.text(px, py, name, fontsize=8)

        # Draw full route once (no animation)
        ax.plot(x, y, marker='o')
        ax.set_title(title)
        ax.grid()

        st.pyplot(fig)

    # ----------------------------
    # ROUTE VISUALIZATION
    # ----------------------------
    st.subheader("🗺️ Route Comparison")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("MPDD")
        plot_route(mpdd_route, "MPDD Route")

    with c2:
        st.write("MILP")
        plot_route(milp_route, "MILP Route")

    with c3:
        st.write("NSGA-II")
        plot_route(nsga_route, "NSGA Route")

    # ----------------------------
    # STEP-BY-STEP ROUTE SIMULATION
    # ----------------------------
    st.subheader("🔍 Route Simulation (Step-by-Step)")

    algo_choice = st.selectbox(
        "Select Algorithm to Simulate",
        ["MPDD", "MILP", "NSGA"],
        key="simulation_algo_choice",
    )

    if algo_choice == "MPDD":
        sim_route = mpdd_route
    elif algo_choice == "MILP":
        sim_route = milp_route
    else:
        sim_route = nsga_route

    steps, total_E, total_F, total_T = simulate_route(sim_route)

    st.write("### 📦 Step-by-Step Movement")

    for i, step in enumerate(steps):
        st.write(
            f"""
            **Step {i + 1}: {step['from']} → {step['to']}**
            - Distance: {step['distance']}
            - Load: {step['load']}
            - Energy Used: {step['energy']}
            - Fatigue: {step['fatigue']}
            - Time: {step['time']}
            """
        )

    st.subheader("📊 Final Totals")
    st.write(f"Energy: {round(total_E, 2)}")
    st.write(f"Fatigue: {round(total_F, 2)}")
    st.write(f"Time: {round(total_T, 2)}")

    # ----------------------------
    # FINAL INSIGHT
    # ----------------------------
    st.subheader("🔍 Insight")

    st.write(
        f"""
        - Agent selected **{decision}** using route performance (E + F + T)  
        - MILP minimizes distance (fastest route)  
        - NSGA balances trade-offs  
        - MPDD provides weighted optimization  
        """
    )
