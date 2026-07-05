"""P1 core bucketing test — 9 euro features × 10 buckets × 3 aggs = 270 dims"""
import sys, os, math, torch
sys.path.insert(0, '.')
from trainer.train_phase1b_event import *
from dataset.odds_dataset import OddsDataset
from tools.p1_time_bucketing import BUCKET_EDGES

device = torch.device('cuda')
torch.manual_seed(42)
train_ids = load_split_ids('data/odds_real/splits_v6/train_match_ids.txt')
ds = OddsDataset('data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl',
    max_seq_len=128, feature_schema_version='v6_event', allowed_match_ids=train_ids, min_events=1)

CORE_FEAT = [0,1,2,9,10,11,12,17,24]  # euro raw + implied + margin + time + has_euro
N_BKT = 10
N_AGG = 3

def bucket_core(events):
    bkts = {}
    for e in events:
        mbk = e.get('minutes_before_kickoff', 0)
        h = mbk / 60.0
        for idx, (hi, lo, _) in enumerate(BUCKET_EDGES):
            if h <= hi and h > lo:
                bkts.setdefault(idx, []).append(e)
                break
    feats = torch.zeros(N_BKT, len(CORE_FEAT), N_AGG)
    for bi in range(N_BKT):
        evts = sorted(bkts.get(bi, []), key=lambda e: e.get('minutes_before_kickoff',0), reverse=True)
        if not evts:
            continue
        for fi, feat_idx in enumerate(CORE_FEAT):
            vals = []
            for e in evts:
                v = _extract(e, feat_idx)
                if v is not None:
                    vals.append(v)
            if not vals:
                continue
            vt = torch.tensor(vals, dtype=torch.float32)
            feats[bi, fi, 0] = vt[-1]           # last
            feats[bi, fi, 1] = vt.mean()        # mean
            feats[bi, fi, 2] = vt[-1] - vt[0]   # delta
    return feats.flatten()

def _extract(e, fi):
    try:
        if fi == 0: return float(e.get('euro_h',0) or 0)
        if fi == 1: return float(e.get('euro_d',0) or 0)
        if fi == 2: return float(e.get('euro_a',0) or 0)
        eh = float(e.get('euro_h',0) or 0)
        ed = float(e.get('euro_d',0) or 0)
        ea = float(e.get('euro_a',0) or 0)
        if eh>0 and ed>0 and ea>0:
            rh, rd, ra = 1.0/eh, 1.0/ed, 1.0/ea
            t = rh + rd + ra
            if t > 0:
                if fi == 9: return rh/t
                if fi == 10: return rd/t
                if fi == 11: return ra/t
        if fi in (9,10,11): return 1.0/3.0
        if fi == 12:
            if eh>0 and ed>0 and ea>0: return (1.0/eh+1.0/ed+1.0/ea)-1.0
            return 0.0
        if fi == 17: return math.log1p(float(e.get('minutes_before_kickoff',0)))
        if fi == 24: return 1.0
        return 0.0
    except Exception:
        return 0.0

Xb, yb = [], []
for i in range(min(4096, len(ds))):
    tl = ds.samples[i].get('odds_timeline', [])
    Xb.append(bucket_core(tl))
    yb.append(ds[i]['euro_label'])
Xb = torch.stack(Xb)
yb = torch.tensor(yb)
n_train = 3072
Xtr, ytr = Xb[:n_train], yb[:n_train]
Xva, yva = Xb[n_train:], yb[n_train:]
print(f'Bucketed core: {Xb.shape[1]} features (expected {N_BKT*len(CORE_FEAT)*N_AGG})')

class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(d),
            torch.nn.Linear(d, 64), torch.nn.SiLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 3))
    def forward(self, x):
        return self.net(x)

m = MLP(Xb.shape[1]).to(device)
opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
best = 99
for ep in range(30):
    m.train()
    perm = torch.randperm(len(Xtr))
    for i in range(0, len(Xtr), 64):
        idx = perm[i:i+64]
        loss = torch.nn.functional.cross_entropy(m(Xtr[idx].to(device)), ytr[idx].to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    sched.step()
    m.eval()
    with torch.no_grad():
        vp = torch.nn.functional.softmax(m(Xva.to(device)), dim=-1).clamp(1e-9, 1-1e-9)
        va = (vp.argmax(-1) == yva.to(device)).float().mean().item()
        vl = -torch.log(vp[torch.arange(len(yva)), yva.to(device)]).mean().item()
    if vl < best:
        best = vl
    if ep % 10 == 0:
        print(f'E{ep:2d}: val_acc={va:.4f} val_logloss={vl:.4f}')
print(f'Best: {best:.4f} | RawPooledMLP ~0.930')
