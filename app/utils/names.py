"""Cross-source player-name resolution (Odds API vs nba_api vs nflverse vs ESPN)."""
import re
import unicodedata

from rapidfuzz import fuzz, process

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.I)
_NONALPHA = re.compile(r"[^a-z ]")
# Known nickname/legal-name mismatches between books and stat feeds. Extend as you hit them.
ALIASES = {
    "cam johnson": "cameron johnson",
    "nic claxton": "nicolas claxton",
    "herb jones": "herbert jones",
    "gabe davis": "gabriel davis",
    "hollywood brown": "marquise brown",
    "kenneth walker": "kenneth walker",
}


def name_key(name: str) -> str:
    """'Luka Dončić' -> 'luka doncic'; 'Kenneth Walker III' -> 'kenneth walker'."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = s.replace(".", "").replace("'", "").replace("-", " ")
    s = _SUFFIX.sub("", s)
    s = _NONALPHA.sub("", s)
    s = " ".join(s.split())
    return ALIASES.get(s, s)


def best_match(name: str, candidates: dict[str, int], threshold: int = 90) -> int | None:
    """candidates: {name_key: player_id}. Exact key first, then token-sort fuzzy match."""
    k = name_key(name)
    if k in candidates:
        return candidates[k]
    hit = process.extractOne(k, candidates.keys(), scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    return candidates[hit[0]] if hit else None
