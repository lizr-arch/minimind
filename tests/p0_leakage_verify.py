"""P0: Leakage verification for iTransformer"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from trainer.train_phase1b_event import *
from torch.utils.data import DataLoader, Subset
from dataset.odds_dataset import OddsDataset
from dataset.odds_collator_v6_event import build_pinnacle_index

device = torch.device('cuda')
torch.manual_seed(42)
tid = load_split_ids('data/odds_real/splits_v6/train_match_ids.txt')
ds = OddsDataset('data/odds_real/titan007_pure_v4_market_semantics_v2.jsonl',
    max_seq_len=64, feature_schema_version='v6_event', allowed_match_ids=tid, min_events=1)

def get_batch(n):
    ss = [ds.samples[i] for i in range(min(n, len(ds)))]
    pi = build_pinnacle_index(ss)
    sub = Subset(ds, range(min(n, len(ds))))
    dl = DataLoader(sub, batch_size=n, collate_fn=OddsEventCollatorWithPrior(pi))
    b = next(iter(dl))
    return b['features'], b['euro_labels']

X, y = get_batch(4096)
print(f'Data: {X.shape}')

F, d = 33, 64
class IT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tp = torch.nn.Sequential(torch.nn.LayerNorm(64), torch.nn.Linear(64,d),
            torch.nn.SiLU(), torch.nn.Linear(d,d))
        self.fid = torch.nn.Embedding(F, d)
        self.cls = torch.nn.Parameter(torch.zeros(1,1,d))
        el = torch.nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=256,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.enc = torch.nn.TransformerEncoder(el, num_layers=2)
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d,3))
    def forward(self, x):
        x = x.transpose(1,2); z = self.tp(x)
        z = z + self.fid(torch.arange(F, device=z.device))[None,:,:]
        z = torch.cat([self.cls.expand(z.size(0),-1,-1), z], dim=1)
        return self.head(self.enc(z)[:,0])

def train_it(X, y, ep=20):
    m = IT().to(device); opt = torch.optim.AdamW(m.parameters(), lr=0.001)
    for _ in range(ep):
        m.train(); perm = torch.randperm(len(X))
        for i in range(0, len(X), 64):
            idx = perm[i:i+64]
            loss = torch.nn.functional.cross_entropy(m(X[idx].to(device)), y[idx].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.nn.functional.softmax(m(X.to(device)), dim=-1)
        return (p.argmax(-1) == y.to(device)).float().mean().item()

# P0.1: Shuffled label
ys = y[torch.randperm(len(y))]
a_s = train_it(X, ys)
a_t = train_it(X, y)
print(f'P0.1: shuffled={a_s:.4f} true={a_t:.4f} ratio={a_s/a_t:.2f} {"PASS" if a_s<0.5 else "FAIL"}')

# P0.2: Match_id overlap
tms = set()
for i in range(min(3000, len(ds))):
    tms.add(ds.samples[i]['match_id'])
vid = load_split_ids('data/odds_real/splits_v6/val_match_ids.txt')
ol = len(tms & vid)
print(f'P0.2: overlap={ol} {"PASS" if ol==0 else "FAIL"}')

# P0.3: Closing-only
Xc, yc = [], []
for i in range(min(4096, len(ds))):
    s = ds[i]; f = s['features']; m = s['attention_mask']
    li = m.sum().long().item() - 1
    Xc.append(f[li]); yc.append(s['euro_label'])
Xc = torch.stack(Xc); yc = torch.tensor(yc)

class MLP(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.BatchNorm1d(d_in),
            torch.nn.Linear(d_in,64), torch.nn.SiLU(), torch.nn.Linear(64,3))
    def forward(self, x):
        return self.net(x)

mlp = MLP(31).to(device); opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
for _ in range(30):
    perm = torch.randperm(len(Xc))
    for i in range(0, len(Xc), 64):
        idx = perm[i:i+64]
        loss = torch.nn.functional.cross_entropy(mlp(Xc[idx].to(device)), yc[idx].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    ac = (mlp(Xc.to(device)).argmax(-1) == yc.to(device)).float().mean().item()
print(f'P0.3: closing-only={ac:.4f} vs iTfm={a_t:.4f} ratio={ac/a_t:.2f} {"WARN" if ac>0.85 else "OK"}')
