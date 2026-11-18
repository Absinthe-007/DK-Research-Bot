"""
kelly.py: Bankroll allocation and optimal sizing using fractional Kelly criterion.
"""

def calculate_kelly_stake(
    true_prob: float, 
    odds: int, 
    bankroll: float, 
    fraction: float = 0.25, 
    max_bankroll_pct: float = 0.05
) -> dict:
    """
    Calculates the recommended wager size using a fractional Kelly criterion.
    
    :param true_prob: Model's assessed true probability of winning (0.0 to 1.0)
    :param odds: Offered American odds
    :param bankroll: Current total bankroll units/dollars
    :param fraction: Kelly multiplier (default 0.25 for Quarter-Kelly to mitigate variance)
    :param max_bankroll_pct: Hard cap on single-bet exposure (default 5%)
    """
    # Calculate decimal odds net payout multiplier (b)
    if odds > 0:
        b = odds / 100.0
    else:
        b = 100.0 / abs(odds)

    q = 1.0 - true_prob
    
    # Kelly Formula: f* = (bp - q) / b
    kelly_fraction = ((b * true_prob) - q) / b

    if kelly_fraction <= 0:
        return {
            "recommended_stake": 0.0,
            "bankroll_percentage": 0.0,
            "raw_kelly_pct": round(kelly_fraction * 100, 2),
            "reason": "Negative or zero edge; no bet recommended."
        }

    # Apply fractional multiplier for safety
    adjusted_kelly = kelly_fraction * fraction

    # Apply hard risk cap
    final_pct = min(adjusted_kelly, max_bankroll_pct)
    recommended_stake = round(bankroll * final_pct, 2)

    return {
        "recommended_stake": recommended_stake,
        "bankroll_percentage": round(final_pct * 100.0, 2),
        "raw_kelly_pct": round(kelly_fraction * 100.0, 2),
        "fraction_used": fraction
    }