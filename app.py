# app.py

import streamlit as st
import matplotlib.pyplot as plt

from optimizer import generate_routes, mpdd_best, milp_best, pareto, evaluate
from agent import decide_algorithm_with_rag
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
    # AGENT DECISION
    # ----------------------------
    decision = decide_algorithm_with_rag(retrieved)

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
    # METRICS
    # ----------------------------
    E1, F1, T1 = evaluate(mpdd_route)
    E2, F2, T2 = evaluate(milp_route)
    E3, F3, T3 = evaluate(nsga_route)

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
    # PLOTTING FUNCTION
    # ----------------------------
    def plot_route(route, title):
        fig, ax = plt.subplots()

        x = [points[p][0] for p in route]
        y = [points[p][1] for p in route]

        ax.plot(x, y, marker='o')

        for name, (px, py) in points.items():
            ax.text(px, py, name, fontsize=8)

        ax.set_title(title)
        ax.grid()

        return fig

    # ----------------------------
    # ROUTE VISUALIZATION
    # ----------------------------
    st.subheader("🗺️ Route Comparison")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write("MPDD")
        st.pyplot(plot_route(mpdd_route, "MPDD"))

    with c2:
        st.write("MILP")
        st.pyplot(plot_route(milp_route, "MILP"))

    with c3:
        st.write("NSGA-II")
        st.pyplot(plot_route(nsga_route, "NSGA"))

    # ----------------------------
    # FINAL INSIGHT
    # ----------------------------
    st.subheader("🔍 Insight")

    st.write(
        f"""
        - Agent selected **{decision}** based on retrieved knowledge  
        - MILP minimizes distance (fastest route)  
        - NSGA balances trade-offs  
        - MPDD provides weighted optimization  
        """
    )