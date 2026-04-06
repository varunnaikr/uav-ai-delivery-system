# main.py

from optimizer import generate_routes, mpdd_best, milp_best, pareto
from utils import plot_route
from agent import decide_algorithm_with_rag   # ✅ FIXED IMPORT
from data import weights
from rag import load_knowledge, build_index, retrieve   # ✅ IMPORT RAG


def main():
    print("Loading knowledge base...")
    docs = load_knowledge()
    index, embeddings = build_index(docs)

    print("Generating routes...")
    routes = generate_routes(2000)

    # 🧠 Context input
    battery_level = 25
    is_emergency = True
    load_level = sum(weights.values())

    # 🔎 Create query
    query = f"battery {battery_level} emergency {is_emergency} load {load_level}"

    print("\nRetrieving knowledge...")
    retrieved = retrieve(query, docs, index)   # ✅ NOW DEFINED

    print("\nRelevant Knowledge:")
    for r in retrieved:
        print("-", r)

    print("\nAgent deciding...")
    decision = decide_algorithm_with_rag(retrieved)   # ✅ WORKS NOW

    print(f"Selected Algorithm: {decision}")

    # 🔀 Run selected algorithm
    if decision == "MPDD":
        route = mpdd_best(routes)

    elif decision == "MILP":
        route = milp_best(routes)

    elif decision == "NSGA":
        pareto_routes = pareto(routes)
        route = pareto_routes[0][0] if pareto_routes else None

    else:
        route = None

    print("\nPlotting route...")
    plot_route(route, f"{decision} Selected Route")


if __name__ == "__main__":
    main()