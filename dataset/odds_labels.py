"""
OddsMind Label Contract — centralized label encoding/decoding.

Single source of truth for all label mappings.
No silent fallbacks — unknown labels raise ValueError.
"""

# ── Euro ──────────────────────────────────────────────────────────────

EURO_RESULT_TO_ID = {"home": 0, "draw": 1, "away": 2}
EURO_ID_TO_RESULT = {0: "home", 1: "draw", 2: "away"}

# ── Asian 3-class ─────────────────────────────────────────────────────

ASIAN_3CLASS_TO_ID = {"upper": 0, "push": 1, "lower": 2}
ASIAN_3CLASS_ID_TO_NAME = {0: "upper", 1: "push", 2: "lower"}

# ── Asian 5-class ─────────────────────────────────────────────────────

ASIAN_5CLASS_TO_ID = {
    "full_win": 0,
    "half_win": 1,
    "push": 2,
    "half_loss": 3,
    "full_loss": 4,
    # Backward compat aliases (old format)
    "upper_full_win": 0,
    "upper_half_win": 1,
    "upper_full_loss": 4,
    "upper_half_loss": 3,
}
ASIAN_5CLASS_ID_TO_NAME = {0: "full_win", 1: "half_win", 2: "push", 3: "half_loss", 4: "full_loss"}

# ── 5→3 collapse ─────────────────────────────────────────────────────

ASIAN_5_TO_3 = {
    "full_win": "upper",
    "half_win": "upper",
    "push": "push",
    "half_loss": "lower",
    "full_loss": "lower",
    # Backward compat
    "upper_full_win": "upper",
    "upper_half_win": "upper",
    "upper_full_loss": "lower",
    "upper_half_loss": "lower",
}

# ── Backward compat aliases (for existing code that imports these) ────

EURO_MAP = EURO_RESULT_TO_ID
ASIAN_MAP = ASIAN_3CLASS_TO_ID
ASIAN_MAP_5CLASS = ASIAN_5CLASS_TO_ID
ASIAN_REV_5CLASS = ASIAN_5CLASS_ID_TO_NAME
ASIAN_REV = ASIAN_3CLASS_ID_TO_NAME


# ── Encode functions ──────────────────────────────────────────────────

def encode_euro_label(label: str) -> int:
    """Encode euro result string to int. Raises ValueError on unknown."""
    if label not in EURO_RESULT_TO_ID:
        raise ValueError(
            f"Unknown euro label '{label}'. "
            f"Allowed: {list(EURO_RESULT_TO_ID.keys())}"
        )
    return EURO_RESULT_TO_ID[label]


def encode_asian_label(label: str, mode: str = "3class") -> int:
    """Encode asian result string to int.
    
    In 3class mode, accepts both 3-class names (upper/push/lower) and
    5-class names (full_win/half_win/push/half_loss/full_loss), collapsing
    5→3 via ASIAN_5_TO_3.
    
    In 5class mode, accepts only 5-class names.
    
    Raises ValueError on unknown label or invalid mode.
    """
    if mode not in ("3class", "5class"):
        raise ValueError(f"Invalid asian label mode '{mode}'. Allowed: '3class', '5class'")
    
    if mode == "3class":
        # Direct 3-class match
        if label in ASIAN_3CLASS_TO_ID:
            return ASIAN_3CLASS_TO_ID[label]
        # 5-class → 3-class collapse
        if label in ASIAN_5_TO_3:
            return ASIAN_3CLASS_TO_ID[ASIAN_5_TO_3[label]]
        raise ValueError(
            f"Unknown asian label '{label}' for 3class mode. "
            f"Allowed: {list(ASIAN_3CLASS_TO_ID.keys())} or {list(ASIAN_5_TO_3.keys())}"
        )
    else:
        # 5-class mode
        if label in ASIAN_5CLASS_TO_ID:
            return ASIAN_5CLASS_TO_ID[label]
        raise ValueError(
            f"Unknown asian label '{label}' for 5class mode. "
            f"Allowed: {list(ASIAN_5CLASS_TO_ID.keys())}"
        )


def normalize_asian_label_3class(label: str) -> str:
    """Normalize a 5-class asian label to its 3-class name.
    
    Returns the 3-class name (upper/push/lower).
    Raises ValueError on unknown label.
    """
    if label in ASIAN_3CLASS_TO_ID:
        return label
    if label in ASIAN_5_TO_3:
        return ASIAN_5_TO_3[label]
    raise ValueError(f"Unknown asian label '{label}'")


def get_asian_num_classes(mode: str = "3class") -> int:
    """Return number of classes for the given asian label mode."""
    if mode == "3class":
        return 3
    elif mode == "5class":
        return 5
    raise ValueError(f"Invalid asian label mode '{mode}'")


# ── Bookmaker embedding ──────────────────────────────────────────────

BOOKMAKER_MAP = {k: i for i, k in enumerate(sorted([
    "Bet365", "Pinnacle", "William Hill", "1XBet", "Bwin", "Betfair Exchange"
]))}
BOOKMAKER_COUNT = len(BOOKMAKER_MAP)
