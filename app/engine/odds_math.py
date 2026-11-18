"""
odds_math.py: Precision odds conversion and vig-removal algorithms.
"""

def american_to_implied(odds: int) -> float:
    """Converts American odds to raw implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)

def implied_to_american(prob: float) -> int:
    """Converts a true probability back to American odds format."""
    if prob <= 0.0 or prob >= 1.0:
        raise ValueError("Probability must be strictly between 0 and 1.")
    if prob >= 0.5:
        odds = int(round(-100.0 * prob / (1.0 - prob)))
    else:
        odds = int(round(100.0 * (1.0 - prob) / prob))
    return odds

def remove_vig_multiplicative(odds_list: list[int]) -> list[float]:
    """
    Removes juice using the standard multiplicative (proportional) method.
    Best for balanced markets where juice is distributed evenly.
    """
    raw_probs = [american_to_implied(o) for o in odds_list]
    total_market_prob = sum(raw_probs)
    
    # Normalize probabilities to sum to 1.0
    true_probs = [p / total_market_prob for p in raw_probs]
    return true_probs

def remove_vig_power(odds_list: list[int], tol: float = 1e-6, max_iter: int = 100) -> list[float]:
    """
    Removes juice using Shin's / Power Method algorithm.
    More accurate for asymmetric markets (heavy favorites vs longshots) 
    by accounting for favorite-longshot bias.
    """
    raw_probs = [american_to_implied(o) for o in odds_list]
    sum_probs = sum(raw_probs)
    
    # If market is already fair
    if abs(sum_probs - 1.0) < tol:
        return raw_probs

    # Solve for power parameter 'k' such that sum(p_i ^ k) = 1.0
    low, high = 0.0, 10.0
    k = 1.0
    for _ in range(max_iter):
        k = (low + high) / 2.0
        val = sum(p ** k for p in raw_probs)
        if abs(val - 1.0) < tol:
            break
        if val > 1.0:
            low = k
        else:
            high = k

    true_probs = [p ** k for p in raw_probs]
    return true_probs