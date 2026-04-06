# agent.py

def decide_algorithm_with_rag(retrieved_knowledge):
    text = " ".join(retrieved_knowledge).lower()

    score = {
        "MPDD": 0,
        "MILP": 0,
        "NSGA": 0
    }

    if "emergency" in text or "minimize time" in text:
        score["MILP"] += 2
        score["NSGA"] += 1

    if "low battery" in text or "energy" in text:
        score["NSGA"] += 2

    if "trade" in text or "multi" in text:
        score["NSGA"] += 2
        score["MPDD"] += 1

    if "heavy" in text:
        score["NSGA"] += 1
        score["MPDD"] += 1

    score["MPDD"] += 1

    return max(score, key=score.get)