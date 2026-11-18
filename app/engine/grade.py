"""
grade.py: Bet settlement, grading, and Closing Line Value (CLV) calculation.
"""
from app.engine.odds_math import american_to_implied

def grade_bet(
    bet_result: str, 
    stake: float, 
    odds: int, 
    closing_odds: int
) -> dict:
    """
    Grades a settled bet and evaluates Closing Line Value (CLV).
    
    :param bet_result: Outcome ('win', 'loss', 'push')
    :param stake: Amount wagered
    :param odds: Odds at which the bet was placed
    :param closing_odds: Final market closing odds right before game start
    """
    # 1. Calculate Payout / Profit
    profit = 0.0
    if bet_result == 'win':
        if odds > 0:
            profit = stake * (odds / 100.0)
        else:
            profit = stake * (100.0 / abs(odds))
    elif bet_result == 'loss':
        profit = -stake
    elif bet_result == 'push':
        profit = 0.0
    else:
        raise ValueError("Invalid bet_result. Use 'win', 'loss', or 'push'.")

    # 2. Calculate Closing Line Value (CLV)
    placed_implied = american_to_implied(odds)
    closing_implied = american_to_implied(closing_odds)
    
    # Positive CLV means you secured a better price than the market closed at
    clv_edge_pct = (closing_implied - placed_implied) * 100.0

    return {
        "result": bet_result,
        "stake": round(stake, 2),
        "profit_loss": round(profit, 2),
        "roi_percentage": round((profit / stake) * 100.0, 2) if stake > 0 else 0.0,
        "placed_odds": odds,
        "closing_odds": closing_odds,
        "clv_percentage": round(clv_edge_pct, 2),
        "beat_closing_line": closing_implied > placed_implied
    }