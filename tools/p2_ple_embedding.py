"""P2: PLE numerical embedding — vectorized, fast version"""
import sys, os, math, torch
sys.path.insert(0, '.')
from trainer.train_phase1b_event import *
from dataset.odds_dataset import OddsDataset
from tools.p1_time_bucketing import BUCKET_EDGES

device = torch.device('cuda')
torch.manual_seed(42)

CORE_FEAT = [0,1,2,9,10,11,12,17,24]
FEAT_NAMES = ['euro_h','euro_d','euro_a','imp_h','imp_d','imp_a','margin','time_log','has_euro']
N_BKT, T_BINS = 10, 8

def extract_feature(e, fi):
    try:
        if fi==0: return float(e.get('euro_h',0) or 0)
        if fi==1: return float(e.get('euro_d',0) or 0)
        if fi==2: return float(e.get('euro_a',0) or 0)
        eh=float(e.get('euro_h',0) or 0);ed=float(e.get('euro_d',0) or 0);ea=float(e.get('euro_a',0) or 0)
        if eh>0 and ed>0 and ea>0:
            rh,rd,ra=1.0/eh,1.0/ed,1.0/ea;t=rh+rd+ra
            if t>0:
                if fi==9: return rh/t
                if fi==10: return rd/t
                if fi==11: return ra/t
        if fi in(9,10,11): return 1.0/3.0
        if fi==12:
            if eh>0 and ed>0 and ea>0: return (1.0/eh+1.0/ed+1.0/ea)-1.0
            return 0.0
        if fi==17: return math.log1p(float(e.get('minutes_before_kickoff',0)))
        if fi==24: return 1.0
        return 0.0
    except: return 0.0

class PLE:
    def __init__(self, values, n_bins=8):
        sv = values.sort().values; n = len(sv)
        edges = torch.tensor([sv[min(int(n*i/n_bins), n-1)].item() for i in range(1, n_bins)])
        # edges: [T-1]
        self.lo = torch.cat([torch.tensor([-float('inf')]), edges])  # [T]
        self.hi = torch.cat([edges, torch.tensor([float('inf')])])    # [T]
        self.T = n_bins
    
    def encode(self, x):
        """Vectorized PLE: x [*] → [*shape, T]"""
        x = torch.as_tensor(x, dtype=torch.float32)
        shape = x.shape
        x = x.reshape(-1, 1)  # [N, 1]
        lo = self.lo.to(x.device).unsqueeze(0)  # [1, T]
        hi = self.hi.to(x.device).unsqueeze(0)  # [1, T]
        
        # For interior bins (t < T-1): linear interpolation
        # et = 0 if x < lo[t]; et = 1 if x >= hi[t]; else (x-lo)/(hi-lo)
        denom = hi - lo
        denom = denom.clamp(min=1e-8)
        raw = (x - lo) / denom  # [N, T]
        
        below = (x < lo).float()
        above = (x >= hi).float()
        
        # Last bin (t == T-1): clamp to [0,1]
        result = raw * (1 - below) * (1 - above) + above
        # Clamp last bin
        last_mask = torch.zeros(1, self.T, device=x.device)
        last_mask[0, -1] = 1.0
        result = torch.where(last_mask.bool(), result.clamp(0, 1), result)
        
        return result.reshape(*shape, self.T)

# Fit PLE on training data
train_ids = load_split_ids('data/odds_real/splits_v6/train_match_ids.txt')
ds = OddsDataset('data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl',
    max_seq_len=128, feature_schema_version='v6_event', allowed_match_ids=train_ids, min_events=1)
n_samp = min(2048, len(ds))
feat_vals = [[] for _ in CORE_FEAT]
for i in range(n_samp):
    for e in ds.samples[i].get('odds_timeline',[]):
        for j,fi in enumerate(CORE_FEAT):
            v=extract_feature(e,fi)
            if v is not None and math.isfinite(v): feat_vals[j].append(v)
ples = [PLE(torch.tensor(v), T_BINS) for v in feat_vals]
for j,p in enumerate(ples): print(f'PLE {FEAT_NAMES[j]}: {T_BINS} bins edges={p.lo.shape[0]}')

# Build features: raw bucketed (no PLE) for baseline
def bucket_raw(events):
    bkts={}
    for e in events:
        h=e.get('minutes_before_kickoff',0)/60.0
        for idx,(hi,lo,_) in enumerate(BUCKET_EDGES):
            if h<=hi and h>lo: bkts.setdefault(idx,[]).append(e); break
    r=torch.zeros(N_BKT,len(CORE_FEAT),2)  # last, mean
    for bi in range(N_BKT):
        evts=sorted(bkts.get(bi,[]),key=lambda e:e.get('minutes_before_kickoff',0),reverse=True)
        if not evts: continue
        for fi in range(len(CORE_FEAT)):
            vals=[extract_feature(e,CORE_FEAT[fi]) for e in evts]
            vals=[v for v in vals if v is not None and math.isfinite(v)]
            if not vals: continue
            vt=torch.tensor(vals,dtype=torch.float32)
            r[bi,fi,0]=vt[-1]
            r[bi,fi,1]=vt.mean()
    return r.flatten()

# Build features: PLE bucketed
def bucket_ple(events):
    bkts={}
    for e in events:
        h=e.get('minutes_before_kickoff',0)/60.0
        for idx,(hi,lo,_) in enumerate(BUCKET_EDGES):
            if h<=hi and h>lo: bkts.setdefault(idx,[]).append(e); break
    r=torch.zeros(N_BKT,len(CORE_FEAT)*T_BINS*2)
    for bi in range(N_BKT):
        evts=sorted(bkts.get(bi,[]),key=lambda e:e.get('minutes_before_kickoff',0),reverse=True)
        if not evts: continue
        for fi in range(len(CORE_FEAT)):
            vals=[extract_feature(e,CORE_FEAT[fi]) for e in evts]
            vals=[v for v in vals if v is not None and math.isfinite(v)]
            if not vals: continue
            lt=ples[fi].encode(torch.tensor(vals[-1]))
            mt=ples[fi].encode(torch.tensor(sum(vals)/len(vals)))
            base=fi*T_BINS*2
            r[bi,base:base+T_BINS]=lt
            r[bi,base+T_BINS:base+2*T_BINS]=mt
    return r.flatten()

print('Building features...')
Xs = {}
ys = None
for label, fn in [('RAW', bucket_raw), ('PLE', bucket_ple)]:
    Xl, yl = [], []
    for i in range(n_samp):
        Xl.append(fn(ds.samples[i].get('odds_timeline',[])))
        yl.append(ds[i]['euro_label'])
    Xs[label] = torch.stack(Xl)
    if ys is None: ys = torch.tensor(yl)
    print(f'  {label}: {Xs[label].shape[1]} dims')

# MLP comparison
class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(d),
            torch.nn.Linear(d, 64), torch.nn.SiLU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 3))
    def forward(self, x):
        return self.net(x)

for label in ['RAW', 'PLE']:
    X = Xs[label]; y = ys
    nt = int(len(X) * 0.75)
    Xtr, Xva = X[:nt], X[nt:]
    ytr, yva = y[:nt], y[nt:]
    m = MLP(X.shape[1]).to(device)
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
            vl = -torch.log(vp[torch.arange(len(yva)), yva.to(device)]).mean().item()
        if vl < best:
            best = vl
    va = (torch.nn.functional.softmax(m(Xva.to(device)), dim=-1).argmax(-1) == yva.to(device)).float().mean().item()
    print(f'{label}: best_val_logloss={best:.4f} val_acc={va:.4f} dims={X.shape[1]}')
