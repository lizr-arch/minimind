"""
P2.2 Domain Taxonomy — competition classification labels.

Supports domestic leagues, cups, continental, international tournaments,
even when current data only has top-5 domestic leagues.
"""

from typing import Dict, Optional

# Known league → domain taxonomy
LEAGUE_TAXONOMY: Dict[str, dict] = {
    "EPL": {
        "competition_name": "English Premier League",
        "country_id": "ENG",
        "competition_type": "domestic_league",
        "domain_family": "top_domestic_league",
        "market_tier": "elite_global",
        "is_knockout": False,
        "is_group_stage": False,
        "is_final_stage": False,
        "is_international": False,
        "is_known_domain": True,
    },
    "LaLiga": {
        "competition_name": "LaLiga",
        "country_id": "ESP",
        "competition_type": "domestic_league",
        "domain_family": "top_domestic_league",
        "market_tier": "elite_global",
        "is_knockout": False, "is_group_stage": False, "is_final_stage": False,
        "is_international": False, "is_known_domain": True,
    },
    "SerieA": {
        "competition_name": "Serie A",
        "country_id": "ITA",
        "competition_type": "domestic_league",
        "domain_family": "top_domestic_league",
        "market_tier": "elite_global",
        "is_knockout": False, "is_group_stage": False, "is_final_stage": False,
        "is_international": False, "is_known_domain": True,
    },
    "Bundesliga": {
        "competition_name": "Bundesliga",
        "country_id": "GER",
        "competition_type": "domestic_league",
        "domain_family": "top_domestic_league",
        "market_tier": "elite_global",
        "is_knockout": False, "is_group_stage": False, "is_final_stage": False,
        "is_international": False, "is_known_domain": True,
    },
    "Ligue1": {
        "competition_name": "Ligue 1",
        "country_id": "FRA",
        "competition_type": "domestic_league",
        "domain_family": "top_domestic_league",
        "market_tier": "high",
        "is_knockout": False, "is_group_stage": False, "is_final_stage": False,
        "is_international": False, "is_known_domain": True,
    },
    # Future domains — taxonomy supports these even without data
    "UCL": {
        "competition_name": "UEFA Champions League",
        "country_id": "INT",
        "competition_type": "continental_club",
        "domain_family": "continental_elite",
        "market_tier": "elite_global",
        "is_knockout": True, "is_group_stage": True, "is_final_stage": True,
        "is_international": True, "is_known_domain": False,
    },
    "UEL": {
        "competition_name": "UEFA Europa League",
        "country_id": "INT",
        "competition_type": "continental_club",
        "domain_family": "continental_elite",
        "market_tier": "high",
        "is_knockout": True, "is_group_stage": True, "is_final_stage": True,
        "is_international": True, "is_known_domain": False,
    },
    "WorldCup": {
        "competition_name": "FIFA World Cup",
        "country_id": "INT",
        "competition_type": "international_tournament",
        "domain_family": "international_major",
        "market_tier": "elite_global",
        "is_knockout": True, "is_group_stage": True, "is_final_stage": True,
        "is_international": True, "is_known_domain": False,
    },
    "Euro": {
        "competition_name": "UEFA European Championship",
        "country_id": "INT",
        "competition_type": "international_tournament",
        "domain_family": "international_major",
        "market_tier": "elite_global",
        "is_knockout": True, "is_group_stage": True, "is_final_stage": True,
        "is_international": True, "is_known_domain": False,
    },
}

_UNKNOWN_TAXONOMY = {
    "competition_name": "unknown",
    "country_id": "UNK",
    "competition_type": "unknown",
    "domain_family": "unknown",
    "market_tier": "unknown",
    "is_knockout": False, "is_group_stage": False, "is_final_stage": False,
    "is_international": False, "is_known_domain": False,
}


def get_domain(league_id: str) -> dict:
    """Return domain taxonomy dict for a league_id. Unknown leagues get _UNKNOWN_TAXONOMY."""
    return LEAGUE_TAXONOMY.get(league_id, {**_UNKNOWN_TAXONOMY, "competition_id": league_id})


def get_domain_family(league_id: str) -> str:
    return get_domain(league_id).get("domain_family", "unknown")


def get_competition_type(league_id: str) -> str:
    return get_domain(league_id).get("competition_type", "unknown")


def get_market_tier(league_id: str) -> str:
    return get_domain(league_id).get("market_tier", "unknown")


def is_known_domain(league_id: str) -> bool:
    return get_domain(league_id).get("is_known_domain", False)


def is_same_domain_family(lg1: str, lg2: str) -> bool:
    return get_domain_family(lg1) == get_domain_family(lg2)


def is_same_competition_type(lg1: str, lg2: str) -> bool:
    return get_competition_type(lg1) == get_competition_type(lg2)


ALL_DOMAIN_FAMILIES = sorted(set(
    t["domain_family"] for t in LEAGUE_TAXONOMY.values()
))
