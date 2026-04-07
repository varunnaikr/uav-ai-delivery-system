# optimizer.py

import numpy as np
import random
from data import points, weights, nodes
from config import BATTERY_CAPACITY


# ----------------------------
# DISTANCE FUNCTION
# ----------------------------
def dist(a, b):
    return np.linalg.norm(np.array(points[a]) - np.array(points[b]))


# ----------------------------
# OBJECTIVE FUNCTION
# ----------------------------
def evaluate(route):
    load = sum(weights.values())
    E = F = T = 0
    current_energy = 0

    for i in range(len(route) - 1):
        d = dist(route[i], route[i + 1])

        # Energy
        segment_energy = d * (10 + 2 * load)
        E += segment_energy
        current_energy += segment_energy

        # Battery constraint
        if current_energy > BATTERY_CAPACITY:
            return float('inf'), float('inf'), float('inf')

        # Fatigue
        F += (load ** 2) * d

        # Time
        speed = 10 / (1 + 0.1 * load)
        T += d / speed

        # Reduce load after delivery
        if route[i + 1] in weights:
            load -= weights[route[i + 1]]

    return E, F, T


# ----------------------------
# ROUTE GENERATION (FIXED DEPOTS)
# ----------------------------
def generate_routes(n=1000, start='Hospital_Hub', end='Regional_Depot'):
    routes = []

    for _ in range(n):
        perm = random.sample(nodes, len(nodes))
        route = [start] + perm + [end]
        routes.append(route)

    return routes


# ----------------------------
# MPDD (Weighted Objective)
# ----------------------------
def mpdd_best(routes):
    best = None
    best_cost = float('inf')

    for r in routes:
        E, F, T = evaluate(r)

        if any(x == float('inf') for x in (E, F, T)):
            continue

        J = E + 0.3 * F + 50 * T

        if J < best_cost:
            best_cost = J
            best = r

    return best


# ----------------------------
# MILP (Shortest Feasible Route)
# ----------------------------
def milp_best(routes):
    best = None
    best_dist = float('inf')

    for r in routes:
        E, F, T = evaluate(r)

        # Skip infeasible routes
        if any(x == float('inf') for x in (E, F, T)):
            continue

        d = sum(dist(r[i], r[i + 1]) for i in range(len(r) - 1))

        if d < best_dist:
            best_dist = d
            best = r

    return best


# ----------------------------
# PARETO DOMINANCE (FIXED BUG)
# ----------------------------
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


# ----------------------------
# NSGA-II (Pareto Front)
# ----------------------------
def pareto(routes):
    objs = [evaluate(r) for r in routes]

    # Filter feasible solutions
    feasible = [
        (routes[i], objs[i])
        for i in range(len(routes))
        if not any(x == float('inf') for x in objs[i])
    ]

    pareto_front = []

    for i in range(len(feasible)):
        dominated_flag = False

        for j in range(len(feasible)):
            if i != j and dominates(feasible[j][1], feasible[i][1]):
                dominated_flag = True
                break

        if not dominated_flag:
            pareto_front.append(feasible[i])

    return pareto_front


def simulate_route(route):
    steps = []

    load = sum(weights.values())
    total_E = total_F = total_T = 0
    current_energy = 0

    for i in range(len(route) - 1):
        start = route[i]
        end = route[i + 1]

        d = dist(start, end)

        segment_energy = d * (10 + 2 * load)
        total_E += segment_energy
        current_energy += segment_energy

        fatigue = (load ** 2) * d
        total_F += fatigue

        speed = 10 / (1 + 0.1 * load)
        time = d / speed
        total_T += time

        steps.append({
            "from": start,
            "to": end,
            "distance": round(d, 2),
            "load": round(load, 2),
            "energy": round(segment_energy, 2),
            "fatigue": round(fatigue, 2),
            "time": round(time, 2),
        })

        if end in weights:
            load -= weights[end]

    return steps, total_E, total_F, total_T
