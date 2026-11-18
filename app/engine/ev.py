"""
ev.py: Expected Value (EV) and mathematical edge calculations.
"""
from app.engine.odds_math import american_to_implied

def calculate_ev(true_prob: float, odds: int) -> dict:
    """
    Calculates Expected Value (EV), decimal payout, and raw edge.
    
    :param true_prob: Model's assessed true win probability (0.0 to 1.0)
    :param odds: Offered American odds (e.g., +150 or -110)
    """
    # Calculate decimal odds
    if odds > 0:
        decimal_odds = (odds / 100.0) + 1.0
    else:
        decimal_odds = (100.0 / abs(odds)) + 1.0

    # EV formula: (Probability of Winning * Net Profit) - (Probability of Losing * Stake)
    net_profit_multiplier = decimal_odds - 1.0
    ev_percentage = (true_prob * net_profit_multiplier) - ((1.0 - true_prob) * 1.0)
    
    # Market implied probability for edge comparison
    market_implied = american_to_implied(odds)
    edge = true_prob - market_implied

    return {
        "true_prob": round(true_prob, 4),
        "market_implied_prob": round(market_implied, 4),
        "decimal_odds": round(decimal_odds, 4),
        "ev_percentage": round(ev_percentage * 100.0, 2), # Expressed as a percentage
        "edge_percentage": round(edge * 100.0, 2),
        "is_positive_ev": ev_percentage > 0.0
    }