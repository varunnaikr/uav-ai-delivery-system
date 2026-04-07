# agent.py


def decide_best_algorithm(mpdd_vals, milp_vals, nsga_vals, is_emergency=False):
    """
    Decide the best algorithm from route metrics.

    - Emergency: prioritize minimum delivery time.
    - Non-emergency: minimize combined objective (energy + fatigue + time).
    """
    metrics = {
        "MPDD": mpdd_vals,
        "MILP": milp_vals,
        "NSGA": nsga_vals,
    }

    if is_emergency:
        return min(metrics, key=lambda name: metrics[name][2])

    return min(metrics, key=lambda name: sum(metrics[name]))
