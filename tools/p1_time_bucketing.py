"""P1: Time bucketing — convert 128 padded events to 10 fixed time buckets."""
import sys, os, json, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from collections import defaultdict

# 10 time buckets (hours before kickoff → label)
BUCKET_EDGES = [
    (float('inf'), 168, 'opening'),      # > 7 days
    (168, 72, '7d'),                     # 7d to 3d
    (72, 24, '3d'),                      # 3d to 1d
    (24, 12, '1d'),                      # 1d to 12h
    (12, 6, '12h'),                      # 12h to 6h
    (6, 3, '6h'),                        # 6h to 3h
    (3, 1, '3h'),                        # 3h to 1h
    (1, 0.5, '1h'),                      # 1h to 30m
    (0.5, 0, '30m'),                     # 30m to 0
    (0, -float('inf'), 'closing'),       # exactly closing (or < 0)
]

N_BUCKETS = len(BUCKET_EDGES)
# Per-event feature indices we bucket (same 33 as v6_event + leadlag/alignment)
# For now, use the first 31 (raw features) — skip leadlag/alignment for bucketing
N_FEATURES = 31  
# Aggregation: last, mean, std, delta (from first in bucket), valid_count
AGG_DIM = 5
BUCKET_FEATURE_DIM = N_BUCKETS * N_FEATURES * AGG_DIM  # 10*31*5 = 1550


def bucket_timeline(events, feature_names=None):
    """
    Convert raw timeline events into bucketed features.
    
    Args:
        events: list of dicts from odds_timeline (sorted by minutes_before_kickoff desc)
        feature_names: optional list of 31 feature names to extract from each event
        
    Returns:
        torch.Tensor [N_BUCKETS * N_FEATURES * AGG_DIM] flattened
    """
    # Group events into buckets
    buckets = defaultdict(list)  # bucket_idx → list of events
    for e in events:
        mbk = e.get('minutes_before_kickoff', 0)
        hours = mbk / 60.0
        for idx, (hi, lo, name) in enumerate(BUCKET_EDGES):
            if hours <= hi and hours > lo:
                buckets[idx].append(e)
                break
    
    # For each bucket, aggregate
    result = torch.zeros(N_BUCKETS, N_FEATURES, AGG_DIM)
    
    for bucket_idx in range(N_BUCKETS):
        evts = buckets.get(bucket_idx, [])
        if not evts:
            continue
        
        # Sort by mbk descending (oldest first in bucket)
        evts_sorted = sorted(evts, key=lambda e: e.get('minutes_before_kickoff', 0), reverse=True)
        
        for feat_idx in range(N_FEATURES):
            vals = []
            for e in evts_sorted:
                v = _extract_feature(e, feat_idx)
                if v is not None:
                    vals.append(v)
            if not vals:
                continue
            
            vals_t = torch.tensor(vals, dtype=torch.float32)
            result[bucket_idx, feat_idx, 0] = vals_t[-1]  # last
            result[bucket_idx, feat_idx, 1] = vals_t.mean()  # mean
            if len(vals) > 1:
                result[bucket_idx, feat_idx, 2] = vals_t.std()  # std
            result[bucket_idx, feat_idx, 3] = vals_t[-1] - vals_t[0]  # delta (last - first)
            result[bucket_idx, feat_idx, 4] = len(vals)  # valid_count
    
    return result.flatten()


def _extract_feature(event, feat_idx):
    """Extract the feat_idx-th feature from a raw timeline event (matching v6_event schema)."""
    try:
        # Features 0-8: raw odds
        if feat_idx == 0: return float(event.get('euro_h', 0) or 0)
        if feat_idx == 1: return float(event.get('euro_d', 0) or 0)
        if feat_idx == 2: return float(event.get('euro_a', 0) or 0)
        if feat_idx == 3: return float(event.get('asian_line', 0) or 0)
        if feat_idx == 4: return float(event.get('upper_water', 0) or 0)
        if feat_idx == 5: return float(event.get('lower_water', 0) or 0)
        if feat_idx == 6: return float(event.get('over_under_line', 0) or 0)
        if feat_idx == 7: return float(event.get('over_water', 0) or 0)
        if feat_idx == 8: return float(event.get('under_water', 0) or 0)
        
        # Features 9-11: euro implied probs (computed)
        eh = float(event.get('euro_h', 0) or 0)
        ed = float(event.get('euro_d', 0) or 0)
        ea = float(event.get('euro_a', 0) or 0)
        if eh > 0 and ed > 0 and ea > 0:
            rh, rd, ra = 1.0/eh, 1.0/ed, 1.0/ea
            total = rh + rd + ra
            if total > 0:
                if feat_idx == 9: return rh / total
                if feat_idx == 10: return rd / total
                if feat_idx == 11: return ra / total
        if feat_idx in (9, 10, 11): return 1.0/3.0
        
        if feat_idx == 12:
            if eh > 0 and ed > 0 and ea > 0:
                return (1.0/eh + 1.0/ed + 1.0/ea) - 1.0
            return 0.0
        
        # Asian probs
        uw = float(event.get('upper_water', 0) or 0)
        lw = float(event.get('lower_water', 0) or 0)
        if feat_idx == 13:
            if uw > 0 and lw > 0: return (1.0/uw) / (1.0/uw + 1.0/lw)
            return 0.5
        if feat_idx == 14:
            if uw > 0 and lw > 0: return (1.0/uw + 1.0/lw) - 1.0
            return 0.0
        
        # OU probs
        ow = float(event.get('over_water', 0) or 0)
        uw2 = float(event.get('under_water', 0) or 0)
        if feat_idx == 15:
            if ow > 0 and uw2 > 0: return (1.0/ow) / (1.0/ow + 1.0/uw2)
            return 0.5
        if feat_idx == 16:
            if ow > 0 and uw2 > 0: return (1.0/ow + 1.0/uw2) - 1.0
            return 0.0
        
        # Time
        if feat_idx == 17:
            return math.log1p(float(event.get('minutes_before_kickoff', 0)))
        
        # Snapshot/market flags — skip for bucketing (always same within bucket)
        if feat_idx in (18, 19, 20, 21, 22, 23):
            return float(event.get('snapshot_type') == 'opening') if feat_idx == 19 else 1.0
        
        # Availability
        if feat_idx == 24: return 1.0  # has_euro always true
        if feat_idx == 25: return 1.0 if event.get('has_asian') else 0.0
        if feat_idx == 26: return 1.0 if event.get('has_over_under') else 0.0
        
        # Change features — set to 0 (meaningless without prev in bucket context)
        if feat_idx in (27, 28, 29, 30): return 0.0
        
        return 0.0
    except Exception:
        return 0.0


# ═══ Quick test ═══
if __name__ == '__main__':
    from trainer.train_phase1b_event import *
    from torch.utils.data import DataLoader, Subset
    from dataset.odds_dataset import OddsDataset
    from dataset.odds_collator_v6_event import build_pinnacle_index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    
    train_ids = load_split_ids('data/odds_real/splits_v6/train_match_ids.txt')
    ds = OddsDataset('data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl',
        max_seq_len=128, feature_schema_version='v6_event', allowed_match_ids=train_ids, min_events=1)
    
    # Test bucketing on a few samples
    for i in range(3):
        s = ds.samples[i]
        tl = s.get('odds_timeline', [])
        bucketed = bucket_timeline(tl)
        print(f'Sample {i}: {len(tl)} events → bucket dim={bucketed.shape[0]} nonzeros={(bucketed!=0).sum().item()}')
    
    # Build full training set with bucketed features
    print('\nBuilding bucketed train set...')
    Xb, yb = [], []
    for i in range(min(4096, len(ds))):
        tl = ds.samples[i].get('odds_timeline', [])
        Xb.append(bucket_timeline(tl))
        yb.append(ds[i]['euro_label'])
    Xb = torch.stack(Xb)
    yb = torch.tensor(yb)
    print(f'Bucketed train: {Xb.shape} — {Xb.shape[1]} features')
    
    # Test: simple MLP on bucketed features
    class MLP(torch.nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.BatchNorm1d(d_in),
                torch.nn.Linear(d_in, 128), torch.nn.SiLU(), torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64), torch.nn.SiLU(), torch.nn.Dropout(0.2),
                torch.nn.Linear(64, 3))
        def forward(self, x): return self.net(x)
    
    # Quick train/val split
    n_train = 3072
    Xtr, ytr = Xb[:n_train], yb[:n_train]
    Xva, yva = Xb[n_train:], yb[n_train:]
    
    m = MLP(Xb.shape[1]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
    best_vl = 99
    for ep in range(30):
        m.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i+64]
            loss = torch.nn.functional.cross_entropy(m(Xtr[idx].to(device)), ytr[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            vp = torch.nn.functional.softmax(m(Xva.to(device)), dim=-1).clamp(1e-9, 1-1e-9)
            va = (vp.argmax(-1) == yva.to(device)).float().mean().item()
            vl = -torch.log(vp[torch.arange(len(yva)), yva.to(device)]).mean().item()
        if vl < best_vl: best_vl = vl
        if ep % 10 == 0:
            print(f'E{ep:2d}: val_acc={va:.4f} val_logloss={vl:.4f}')
    print(f'\nBest val_logloss: {best_vl:.4f}')
    print(f'Baseline: RawPooledMLP ~0.930')
