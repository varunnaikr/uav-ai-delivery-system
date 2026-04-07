# agent.py


def _normalize(values):
    """Min-max normalize metric values where lower is better."""
    vmin = min(values)
    vmax = max(values)

    if vmax == vmin:
        return [0.0 for _ in values]

    return [(v - vmin) / (vmax - vmin) for v in values]


def calculate_algorithm_scores(
    mpdd_vals,
    milp_vals,
    nsga_vals,
    battery_level,
    num_routes,
    is_emergency=False,
):
    """
    Compute weighted performance score (lower is better) for each algorithm.
    Weights adapt to mission context instead of using a hardcoded decision table.
    """
    energy_vals = [mpdd_vals[0], milp_vals[0], nsga_vals[0]]
    fatigue_vals = [mpdd_vals[1], milp_vals[1], nsga_vals[1]]
    time_vals = [mpdd_vals[2], milp_vals[2], nsga_vals[2]]

    n_energy = _normalize(energy_vals)
    n_fatigue = _normalize(fatigue_vals)
    n_time = _normalize(time_vals)

    is_low_battery = battery_level < 70
    is_high_routes = num_routes >= 3000

    # Context-sensitive objective importance
    w_energy = 0.50 if is_low_battery else 0.25
    w_time = 0.50 if is_emergency else 0.25

    # Route-space complexity: with more routes, multi-objective robustness matters
    w_fatigue = 0.30 if is_high_routes else 0.20

    # Rebalance to keep sum(weights)=1.0
    total = w_energy + w_fatigue + w_time
    w_energy, w_fatigue, w_time = (
        w_energy / total,
        w_fatigue / total,
        w_time / total,
    )

    labels = ["MPDD", "MILP", "NSGA"]
    scores = {}

    for i, name in enumerate(labels):
        scores[name] = (
            w_energy * n_energy[i]
            + w_fatigue * n_fatigue[i]
            + w_time * n_time[i]
        )

    return scores, {
        "energy": round(w_energy, 3),
        "fatigue": round(w_fatigue, 3),
        "time": round(w_time, 3),
    }


def decide_best_algorithm(
    mpdd_vals,
    milp_vals,
    nsga_vals,
    battery_level,
    num_routes,
    is_emergency=False,
):
    scores, _ = calculate_algorithm_scores(
        mpdd_vals,
        milp_vals,
        nsga_vals,
        battery_level=battery_level,
        num_routes=num_routes,
        is_emergency=is_emergency,
    )

    return min(scores, key=scores.get)
