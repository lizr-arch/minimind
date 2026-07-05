"""
OddsEventCollator — Event-driven collator for v6_event schema.

Features:
    - Variable-length sequence padding with attention mask
    - Lead-lag feature computation (cross-bookmaker Pinnacle comparison)
    - Euro-Asian alignment feature computation
    - Single-bookmaker event sequence batch assembly

Output batch:
    features:        [B, T_max, 33]  (31 + 2 new features)
    missing_mask:    [B, T_max, 31]  (mask excludes new features)
    attention_mask:  [B, T_max]      (1=valid, 0=pad)
    euro_labels:     [B]             long
    asian_labels:    [B]             long
    match_ids:       list[str]
    league_ids:      list[str]
    event_counts:    [B]             (per-sample actual event count)
"""

from typing import Dict, List, Optional

import torch


def build_pinnacle_index(samples: List[dict]) -> Dict[str, List[dict]]:
    """Build Pinnacle timeline index from dataset samples.
    
    Args:
        samples: list of sample dicts, each containing 'match_id', 'bookmaker_id', 'odds_timeline'
    
    Returns:
        dict mapping match_id -> Pinnacle's timeline (sorted by minutes_before_kickoff descending)
    """
    index = {}
    for sample in samples:
        if sample.get("bookmaker_id") == "Pinnacle":
            match_id = sample["match_id"]
            timeline = sample.get("odds_timeline", [])
            # Sort descending by minutes_before_kickoff (earliest first)
            sorted_timeline = sorted(
                timeline,
                key=lambda e: e.get("minutes_before_kickoff", 0),
                reverse=True,
            )
            index[match_id] = sorted_timeline
    return index


def compute_leadlag_feature(
    event: dict,
    pinnacle_timeline: Optional[List[dict]],
    prev_event: Optional[dict],
    bookmaker_id: str,
) -> float:
    """Compute lead-lag feature for a single event.
    
    Algorithm:
        1. If bookmaker is Pinnacle, return 0.0
        2. If pinnacle_timeline is None, return 0.0
        3. Find Pinnacle events at or before this moment (minutes_before_kickoff >= T)
        4. Compute Pinnacle's euro_h change direction
        5. Compute current bookmaker's euro_h change direction
        6. Return +1.0 if same direction, -1.0 if opposite, 0.0 otherwise
    
    Information leak control: Only use Pinnacle info at or before moment T.
    """
    # Rule 1: Pinnacle itself has no lead-lag
    if bookmaker_id == "Pinnacle":
        return 0.0
    
    # Rule 2: No Pinnacle data
    if pinnacle_timeline is None or len(pinnacle_timeline) == 0:
        return 0.0
    
    # Current event's time
    current_minutes = event.get("minutes_before_kickoff", 0)
    
    # Find Pinnacle events at or before this moment (>= current_minutes means earlier in time)
    # Pinnacle timeline is sorted descending, so events with larger minutes_before_kickoff are earlier
    pinnacle_before = []
    pinnacle_after = []
    for p_event in pinnacle_timeline:
        p_minutes = p_event.get("minutes_before_kickoff", 0)
        if p_minutes >= current_minutes:
            pinnacle_before.append(p_event)
        else:
            pinnacle_after.append(p_event)
    
    # Need at least one Pinnacle event before and one after (or at) this moment
    if len(pinnacle_before) == 0:
        return 0.0
    
    # Get Pinnacle's last event before this moment
    p_last_before = pinnacle_before[0]  # First in list = most recent before T (largest minutes)
    
    # Get Pinnacle's first event at or after this moment
    p_first_after = pinnacle_before[-1] if len(pinnacle_before) > 0 else None
    
    # If we only have one Pinnacle event before T, we can't compute direction
    if p_first_after is None or p_first_after == p_last_before:
        return 0.0
    
    # Compute Pinnacle direction: from last_before to first_after
    p_euro_h_before = float(p_last_before.get("euro_h", 0) or 0)
    p_euro_h_after = float(p_first_after.get("euro_h", 0) or 0)
    
    if p_euro_h_before <= 0 or p_euro_h_after <= 0:
        pinnacle_direction = 0
    elif p_euro_h_after < p_euro_h_before:
        pinnacle_direction = 1  # euro_h decreased = favoring home
    elif p_euro_h_after > p_euro_h_before:
        pinnacle_direction = -1  # euro_h increased = favoring away
    else:
        pinnacle_direction = 0
    
    # Compute current bookmaker direction (vs previous event)
    if prev_event is None:
        return 0.0
    
    curr_euro_h = float(event.get("euro_h", 0) or 0)
    prev_euro_h = float(prev_event.get("euro_h", 0) or 0)
    
    if curr_euro_h <= 0 or prev_euro_h <= 0:
        our_direction = 0
    elif curr_euro_h < prev_euro_h:
        our_direction = 1  # euro_h decreased = favoring home
    elif curr_euro_h > prev_euro_h:
        our_direction = -1  # euro_h increased = favoring away
    else:
        our_direction = 0
    
    # Compare directions
    if pinnacle_direction != 0 and pinnacle_direction == our_direction:
        return 1.0
    elif pinnacle_direction != 0 and pinnacle_direction != our_direction:
        return -1.0
    else:
        return 0.0


def compute_euro_asian_alignment(
    event: dict,
    prev_event: Optional[dict],
) -> float:
    """Compute Euro-Asian alignment feature.
    
    Algorithm:
        1. If prev_event is None, return 0.0
        2. Compute euro direction: euro_h decrease -> +1 (favor home)
        3. Compute asian direction: asian_line decrease -> +1 (favor home)
           If asian_line unchanged, use upper_water
        4. Return +1.0 if same direction, -1.0 if opposite, 0.0 otherwise
    """
    if prev_event is None:
        return 0.0
    
    # Euro direction
    curr_euro_h = float(event.get("euro_h", 0) or 0)
    prev_euro_h = float(prev_event.get("euro_h", 0) or 0)
    
    if curr_euro_h <= 0 or prev_euro_h <= 0:
        euro_dir = 0
    elif curr_euro_h < prev_euro_h:
        euro_dir = 1  # favor home
    elif curr_euro_h > prev_euro_h:
        euro_dir = -1  # favor away
    else:
        euro_dir = 0
    
    # Asian direction
    curr_asian = float(event.get("asian_line", 0) or 0)
    prev_asian = float(prev_event.get("asian_line", 0) or 0)
    
    if curr_asian < prev_asian:
        asian_dir = 1  # favor home (more handicap)
    elif curr_asian > prev_asian:
        asian_dir = -1  # favor away (less handicap)
    else:
        # Asian line unchanged, check water
        curr_upper = float(event.get("upper_water", 0) or 0)
        prev_upper = float(prev_event.get("upper_water", 0) or 0)
        
        if curr_upper <= 0 or prev_upper <= 0:
            asian_dir = 0
        elif curr_upper < prev_upper:
            asian_dir = 1  # favor home
        elif curr_upper > prev_upper:
            asian_dir = -1  # favor away
        else:
            asian_dir = 0
    
    # Compare directions
    if euro_dir != 0 and euro_dir == asian_dir:
        return 1.0
    elif euro_dir != 0 and euro_dir != asian_dir:
        return -1.0
    else:
        return 0.0


class OddsEventCollator:
    """Collate function for v6_event schema with lead-lag and alignment features.
    
    Args:
        pinnacle_index: dict mapping match_id -> Pinnacle timeline
        pad_value: value for padding (default 0.0)
    """
    
    def __init__(
        self,
        pinnacle_index: Optional[Dict[str, List[dict]]] = None,
        pad_value: float = 0.0,
    ):
        self.pinnacle_index = pinnacle_index or {}
        self.pad_value = pad_value
    
    def __call__(self, batch: List[dict]) -> dict:
        """Collate a batch of v6_event samples.
        
        Args:
            batch: list of dicts from OddsDataset.__getitem__(), each containing:
                - features: [T_i, 31] tensor
                - missing_mask: [T_i, 31] tensor
                - attention_mask: [T_i] tensor
                - euro_label: int
                - asian_label: int
                - match_id: str
                - league_id: str
                - bookmaker_id: str (from raw_timeline)
                - raw_timeline: list[dict] (for lead-lag computation)
        
        Returns:
            dict with batched tensors
        """
        batch_size = len(batch)
        
        # Find max sequence length in this batch
        max_len = max(item["features"].shape[0] for item in batch)
        feature_dim = 33  # 31 + 2 new features
        
        # Allocate padded tensors
        features = torch.full(
            (batch_size, max_len, feature_dim),
            self.pad_value,
            dtype=torch.float32,
        )
        missing_mask = torch.zeros(batch_size, max_len, 33, dtype=torch.float32)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        euro_labels = torch.zeros(batch_size, dtype=torch.long)
        asian_labels = torch.zeros(batch_size, dtype=torch.long)
        event_counts = torch.zeros(batch_size, dtype=torch.long)
        match_ids = []
        league_ids = []
        
        for i, item in enumerate(batch):
            seq_len = item["features"].shape[0]
            match_id = item["match_id"]
            bookmaker_id = item.get("bookmaker_id", "Bet365")
            
            # Get Pinnacle timeline for this match
            pinnacle_tl = self.pinnacle_index.get(match_id)
            
            # Compute lead-lag and alignment features for each event
            raw_timeline = item.get("raw_timeline", [])
            leadlag_features = []
            alignment_features = []
            
            for t in range(seq_len):
                event = raw_timeline[t] if t < len(raw_timeline) else {}
                prev_event = raw_timeline[t - 1] if t > 0 and t < len(raw_timeline) else None
                
                ll = compute_leadlag_feature(event, pinnacle_tl, prev_event, bookmaker_id)
                al = compute_euro_asian_alignment(event, prev_event)
                
                leadlag_features.append(ll)
                alignment_features.append(al)
            
            # Stack new features: [T, 2]
            new_features = torch.tensor(
                [leadlag_features, alignment_features],
                dtype=torch.float32,
            ).T  # [T, 2]
            
            # Concatenate with original features: [T, 31] + [T, 2] = [T, 33]
            full_features = torch.cat([item["features"], new_features], dim=1)
            
            # Fill batch tensors
            features[i, :seq_len] = full_features
            # missing_mask: base 31 + 2 zeros for lead-lag/alignment
            missing_mask[i, :seq_len, :31] = item["missing_mask"]
            attention_mask[i, :seq_len] = item["attention_mask"]
            euro_labels[i] = item["euro_label"]
            asian_labels[i] = item["asian_label"]
            event_counts[i] = seq_len
            match_ids.append(match_id)
            league_ids.append(item.get("league_id", ""))
        
        return {
            "features": features,
            "missing_mask": missing_mask,
            "attention_mask": attention_mask,
            "euro_labels": euro_labels,
            "asian_labels": asian_labels,
            "match_ids": match_ids,
            "league_ids": league_ids,
            "event_counts": event_counts,
        }
