# agent.py


def decide_best_algorithm(mpdd_vals, milp_vals, nsga_vals):
    """
    Decide the best algorithm based on actual route performance.
    Lower combined objective (energy + fatigue + time) is better.
    """
    e1, f1, t1 = mpdd_vals
    e2, f2, t2 = milp_vals
    e3, f3, t3 = nsga_vals

    scores = {
        "MPDD": e1 + f1 + t1,
        "MILP": e2 + f2 + t2,
        "NSGA": e3 + f3 + t3,
    }

    return min(scores, key=scores.get)
