# agent.py


def decide_best_algorithm(
    mpdd_vals,
    milp_vals,
    nsga_vals,
    battery_level,
    num_routes,
    is_emergency=False,
):
    """
    Decide the best algorithm from route metrics and mission context.

    Rules:
    - Emergency: prioritize minimum delivery time.
    - High battery + non-emergency + large route space: prefer NSGA-II.
    - Otherwise: minimize combined objective (energy + fatigue + time).
    """
    metrics = {
        "MPDD": mpdd_vals,
        "MILP": milp_vals,
        "NSGA": nsga_vals,
    }

    if is_emergency:
        return min(metrics, key=lambda name: metrics[name][2])

    if battery_level >= 70 and num_routes >= 3000:
        return "NSGA"

    return min(metrics, key=lambda name: sum(metrics[name]))
