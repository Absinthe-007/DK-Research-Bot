"""
Canonical market dictionary + mapping from ESPN/DraftKings prop "type" names.

ESPN's core API republishes DraftKings (provider id 100) prices, including player props:
  * "Total X" types are two-way Over/Under markets. ESPN lists the pair consecutively with
    the OVER FIRST (verified against DK's milestone ladder: the first price always matches the
    `N+` milestone at the same threshold).
  * "X Milestones" are one-sided alt lines "N+" -> stored as Over (N - 0.5), kind="alt".
  * Anytime TD / 2+ TDs are one-sided yes bets -> Over 0.5 / Over 1.5, kind="alt".
"""

# key, sport, label, stat_expr (over the sport's gamelog columns), distribution family
MARKETS = [
    ("player_points", "NBA", "Points", "pts", "normal"),
    ("player_rebounds", "NBA", "Rebounds", "reb", "negbin"),
    ("player_assists", "NBA", "Assists", "ast", "negbin"),
    ("player_threes", "NBA", "3PT Made", "fg3m", "negbin"),
    ("player_blocks", "NBA", "Blocks", "blk", "poisson"),
    ("player_steals", "NBA", "Steals", "stl", "poisson"),
    ("player_steals_blocks", "NBA", "Stl+Blk", "stl+blk", "poisson"),
    ("player_points_rebounds_assists", "NBA", "Pts+Reb+Ast", "pts+reb+ast", "normal"),
    ("player_points_rebounds", "NBA", "Pts+Reb", "pts+reb", "normal"),
    ("player_points_assists", "NBA", "Pts+Ast", "pts+ast", "normal"),
    ("player_rebounds_assists", "NBA", "Reb+Ast", "reb+ast", "negbin"),
    ("player_pass_yds", "NFL", "Passing Yards", "pass_yds", "normal"),
    ("player_pass_tds", "NFL", "Passing TDs", "pass_tds", "poisson"),
    ("player_pass_completions", "NFL", "Completions", "completions", "normal"),
    ("player_pass_attempts", "NFL", "Pass Attempts", "pass_attempts", "normal"),
    ("player_pass_interceptions", "NFL", "Interceptions", "interceptions", "poisson"),
    ("player_rush_yds", "NFL", "Rushing Yards", "rush_yds", "gamma"),
    ("player_rush_attempts", "NFL", "Rush Attempts", "carries", "negbin"),
    ("player_receptions", "NFL", "Receptions", "receptions", "negbin"),
    ("player_reception_yds", "NFL", "Receiving Yards", "rec_yds", "gamma"),
    ("player_rush_reception_yds", "NFL", "Rush+Rec Yards", "rush_yds+rec_yds", "gamma"),
    ("player_pass_rush_yds", "NFL", "Pass+Rush Yards", "pass_yds+rush_yds", "normal"),
    ("player_anytime_td", "NFL", "Touchdowns", "rush_tds+rec_tds", "poisson"),
]
MARKET_INDEX = {m[0]: m for m in MARKETS}

# ESPN prop type name -> (market_key, kind)   kind: "ou" two-way | "alt" one-sided N+ | "yes" one-sided fixed line
PROP_TYPES = {
    # NBA
    "Total Points": ("player_points", "ou"),
    "Total Rebounds": ("player_rebounds", "ou"),
    "Total Assists": ("player_assists", "ou"),
    "Total 3-Point Field Goals": ("player_threes", "ou"),
    "Total Steals": ("player_steals", "ou"),
    "Total Blocks": ("player_blocks", "ou"),
    "Total Steals and Blocks": ("player_steals_blocks", "ou"),
    "Total Points and Rebounds": ("player_points_rebounds", "ou"),
    "Total Points and Assists": ("player_points_assists", "ou"),
    "Total Assists and Rebounds": ("player_rebounds_assists", "ou"),
    "Total Points, Rebounds, and Assists": ("player_points_rebounds_assists", "ou"),
    "Points Milestones": ("player_points", "alt"),
    "Rebounds Milestones": ("player_rebounds", "alt"),
    "Assists Milestones": ("player_assists", "alt"),
    "3-Point Field Goals Milestones": ("player_threes", "alt"),
    "Steals Milestones": ("player_steals", "alt"),
    "Blocks Milestones": ("player_blocks", "alt"),
    "Points + Rebounds Milestones": ("player_points_rebounds", "alt"),
    "Points + Assists Milestones": ("player_points_assists", "alt"),
    "Points + Assists + Rebounds Milestones": ("player_points_rebounds_assists", "alt"),
    # NFL
    "Total Passing Yards (incl. overtime)": ("player_pass_yds", "ou"),
    "Total Pass Completions (incl. overtime)": ("player_pass_completions", "ou"),
    "Total Passing Attempts (incl. overtime)": ("player_pass_attempts", "ou"),
    "Total Passing Touchdowns (incl. overtime)": ("player_pass_tds", "ou"),
    "Total Passing Interceptions (incl. overtime)": ("player_pass_interceptions", "ou"),
    "Total Rushing Yards (incl. overtime)": ("player_rush_yds", "ou"),
    "Total Carries (incl. overtime)": ("player_rush_attempts", "ou"),
    "Total Receiving Yards (incl. overtime)": ("player_reception_yds", "ou"),
    "Total Receptions (incl. overtime)": ("player_receptions", "ou"),
    "Total Rushing Plus Receiving Yards (incl. overtime)": ("player_rush_reception_yds", "ou"),
    "Total Passing Plus Rushing Yards (incl. overtime)": ("player_pass_rush_yds", "ou"),
    "Passing Yards Milestones": ("player_pass_yds", "alt"),
    "Passing Completions Milestones": ("player_pass_completions", "alt"),
    "Passing Attempts Milestones": ("player_pass_attempts", "alt"),
    "Passing Touchdown Milestones": ("player_pass_tds", "alt"),
    "Interceptions Thrown Milestones": ("player_pass_interceptions", "alt"),
    "Rushing Yards Milestones": ("player_rush_yds", "alt"),
    "Rushing Attempts Milestones": ("player_rush_attempts", "alt"),
    "Receiving Yards Milestones": ("player_reception_yds", "alt"),
    "Receptions Milestones": ("player_receptions", "alt"),
    "Rushing + Receiving Yards Milestones": ("player_rush_reception_yds", "alt"),
    "Passing + Rushing Yards Milestones": ("player_pass_rush_yds", "alt"),
    "Anytime Touchdown Scorer": ("player_anytime_td", "yes0.5"),
    "Player to score 2 or more touchdowns": ("player_anytime_td", "yes1.5"),
}

ESPN_PATH = {"NBA": "basketball/nba", "NFL": "football/nfl"}

# Neutral-site / international venues ESPN uses (lat, lon, elevation ft). Home stadiums are
# geocoded automatically from ESPN's venue address (Open-Meteo geocoder) and cached.
KNOWN_VENUES = {
    "Stade de France": (48.9245, 2.3602, 115),
    "Tottenham Hotspur Stadium": (51.6043, -0.0664, 100),
    "Wembley Stadium": (51.5560, -0.2796, 150),
    "Allianz Arena": (48.2188, 11.6247, 1650),
    "Deutsche Bank Park": (50.0686, 8.6455, 330),
    "Olympiastadion": (52.5147, 13.2395, 160),
    "Estadio Banorte": (19.3029, -99.1505, 7350),
    "Estadio Azteca": (19.3029, -99.1505, 7350),
    "Santiago Bernabéu": (40.4531, -3.6883, 2130),
    "Estadio Santiago Bernabéu": (40.4531, -3.6883, 2130),
    "Croke Park": (53.3607, -6.2512, 50),
    "Melbourne Cricket Ground": (-37.8199, 144.9834, 100),
    "Neo Química Arena": (-23.5453, -46.4742, 2500),
    "Arena Corinthians": (-23.5453, -46.4742, 2500),
}
