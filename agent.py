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
    Decide algorithm strategy from mission context.

    Decision table:
    - High battery + no emergency + low routes  -> MPDD
    - High battery + no emergency + high routes -> NSGA-II
    - High battery + emergency + low routes     -> MILP
    - High battery + emergency + high routes    -> MILP + NSGA-II
    - Low battery + no emergency + low routes   -> MILP
    - Low battery + no emergency + high routes  -> Constrained NSGA-II
    - Low battery + emergency + low routes      -> MILP
    - Low battery + emergency + high routes     -> MILP
    """
    is_high_battery = battery_level >= 70
    is_high_routes = num_routes >= 3000

    if is_high_battery and not is_emergency and not is_high_routes:
        return "MPDD"

    if is_high_battery and not is_emergency and is_high_routes:
        return "NSGA-II"

    if is_high_battery and is_emergency and not is_high_routes:
        return "MILP"

    if is_high_battery and is_emergency and is_high_routes:
        return "MILP + NSGA-II"

    if not is_high_battery and not is_emergency and not is_high_routes:
        return "MILP"

    if not is_high_battery and not is_emergency and is_high_routes:
        return "Constrained NSGA-II"

    # Low battery + emergency (both route-space levels): reliability first.
    return "MILP"
