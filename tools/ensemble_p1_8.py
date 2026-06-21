"""P1.8 Ensemble: average Transformer + Tabular consensus probabilities."""
import json, sys, torch
sys.path.insert(0, '.')
from dataset.odds_dataset import OddsDataset, EURO_MAP, ASIAN_MAP_5CLASS
from dataset.odds_collator import OddsCollator
from dataset.odds_split import load_match_ids_from_file
from model.model_oddsmind import OddsMindConfig, OddsMindModel
from torch.utils.data import DataLoader
from eval.odds_metrics import accuracy_from_probs, logloss_from_probs, brier_from_probs

DEVICE = 'cuda'
DATA = 'data/odds_real/enriched_no_consensus.jsonl'
TEST_IDS = 'data/odds_real/splits_p1_7/test_match_ids.txt'
TRANSFORMER_CKP = 'runs/p1_8_transformer_baseline/oddsmind_smoke.pth'

# ── Load test data ──
test_ids = load_match_ids_from_file(TEST_IDS)
ds = OddsDataset(DATA, cutoffs=[0], cutoff_mode='exhaustive', asian_label_mode='5class', allowed_match_ids=test_ids)
collator = OddsCollator()
loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collator)
print(f'Test samples: {len(ds)}')

# ── Transformer predictions ──
cfg = OddsMindConfig(hidden_size=512, num_hidden_layers=16, num_attention_heads=8, asian_num_classes=5)
model = OddsMindModel(cfg).to(DEVICE)
ckp = torch.load(TRANSFORMER_CKP, map_location=DEVICE)
if 'model_state_dict' in ckp: ckp = ckp['model_state_dict']
model.load_state_dict(ckp, strict=False)
model.eval()

transformer_euro_probs = []
transformer_asian_probs = []
all_euro_labels = []
all_asian_labels = []
all_consensus = []
all_match_ids = []

with torch.no_grad():
    for batch in loader:
        f = batch['features'].to(DEVICE)
        m = batch['attention_mask'].to(DEVICE)
        out = model(f, attention_mask=m)
        transformer_euro_probs.append(torch.softmax(out['euro_logits'], -1).cpu())
        transformer_asian_probs.append(torch.softmax(out['asian_logits'], -1).cpu())
        all_euro_labels.append(batch['euro_labels'])
        all_asian_labels.append(batch['asian_labels'])
        all_consensus.append(batch.get('consensus_feats', torch.zeros(len(f),6)))
        all_match_ids.extend(batch['match_ids'])

t_euro = torch.cat(transformer_euro_probs, 0)
t_asian = torch.cat(transformer_asian_probs, 0)
eu_labels = torch.cat(all_euro_labels, 0)
ah_labels = torch.cat(all_asian_labels, 0)
consensus = torch.cat(all_consensus, 0)

# ── Tabular predictions (from consensus features) ──
# Simple logistic: softmax of consensus → prediction
# Use a tiny logistic model trained on consensus features
from model.odds_tabular_baselines import OddsLogisticRegression
import torch.nn as nn
tab_model = OddsLogisticRegression(asian_num_classes=5)
# Replace linear heads to take 6 consensus features
tab_model.euro_head = nn.Linear(6, 3)
tab_model.asian_head = nn.Linear(6, 5)
# Load tabular weights from previous training
# Quick train on consensus features
X_train_list, y_eu_list, y_ah_list = [], [], []
train_ids = load_match_ids_from_file('data/odds_real/splits_p1_7/train_match_ids.txt')
ds_train = OddsDataset(DATA, cutoffs=[0], cutoff_mode='exhaustive', asian_label_mode='5class', allowed_match_ids=train_ids)
loader_train = DataLoader(ds_train, batch_size=64, shuffle=False, collate_fn=collator)
with torch.no_grad():
    for batch in loader_train:
        f = batch['features'].to(DEVICE)
        m = batch['attention_mask'].to(DEVICE)
        out = model(f, attention_mask=m)
        eu_p = torch.softmax(out['euro_logits'], -1).cpu()
        ah_p = torch.softmax(out['asian_logits'], -1).cpu()
    # Actually, we need the raw consensus features from the dataset
    # Simpler: just train logistic directly on consensus features

# Let me build a quick train set for tabular
X_tab, y_eu_tab, y_ah_tab = [], [], []
for i in range(len(ds_train)):
    item = ds_train._odds_ds.samples[i]
    # Get consensus features from the sample's bookmaker_features
    bk_feats = item.get('bookmaker_features', {})
    if bk_feats:
        vec = [bk_feats.get('home_prob_avg',0), bk_feats.get('home_prob_median',0),
               bk_feats.get('draw_prob_avg',0), bk_feats.get('draw_prob_median',0),
               bk_feats.get('away_prob_avg',0), bk_feats.get('away_prob_median',0)]
    else:
        vec = [0]*6
    X_tab.append(vec)
    y_eu_tab.append(item.get('euro_label',0) if isinstance(item.get('euro_label'),int) else EURO_MAP.get(item['label']['euro_result'],0))
    y_ah_tab.append(item.get('asian_label',0) if isinstance(item.get('asian_label'),int) else ASIAN_MAP_5CLASS.get(item['label']['asian_result'],0))

import torch.optim as optim
X_tab_t = torch.tensor(X_tab, dtype=torch.float32)
y_eu_t = torch.tensor(y_eu_tab)
y_ah_t = torch.tensor(y_ah_tab)

tab_model_2 = nn.Sequential(
    nn.Linear(6, 24), nn.SiLU(),
    nn.Linear(24, 3)
)
tab_ah = nn.Sequential(
    nn.Linear(6, 24), nn.SiLU(),
    nn.Linear(24, 5)
)
opt = optim.AdamW(list(tab_model_2.parameters()) + list(tab_ah.parameters()), lr=1e-2)
for ep in range(20):
    opt.zero_grad()
    eu_logits = tab_model_2(X_tab_t)
    ah_logits = tab_ah(X_tab_t)
    loss = torch.nn.functional.cross_entropy(eu_logits, y_eu_t) + torch.nn.functional.cross_entropy(ah_logits, y_ah_t)
    loss.backward()
    opt.step()
print(f'Tabular trained: loss={loss.item():.4f}')

# Tabular predictions on test
X_test = []
for i in range(len(ds)):
    item = ds._odds_ds.samples[i]
    bk_feats = item.get('bookmaker_features', {})
    if bk_feats:
        vec = [bk_feats.get('home_prob_avg',0), bk_feats.get('home_prob_median',0),
               bk_feats.get('draw_prob_avg',0), bk_feats.get('draw_prob_median',0),
               bk_feats.get('away_prob_avg',0), bk_feats.get('away_prob_median',0)]
    else:
        vec = [0]*6
    X_test.append(vec)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    tab_euro = torch.softmax(tab_model_2(X_test_t), -1)
    tab_asian = torch.softmax(tab_ah(X_test_t), -1)

# ── Ensemble: simple average ──
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ens_euro = w * t_euro + (1-w) * tab_euro
    ens_asian = w * t_asian + (1-w) * tab_asian
    eu_acc = accuracy_from_probs(ens_euro, eu_labels)
    eu_ll = logloss_from_probs(ens_euro, eu_labels)
    eu_br = brier_from_probs(ens_euro, eu_labels, 3)
    ah_acc = accuracy_from_probs(ens_asian, ah_labels)
    ah_ll = logloss_from_probs(ens_asian, ah_labels)
    label = 'Ensemble' if 0 < w < 1 else ('Transformer' if w == 1 else 'Tabular')
    print(f'w={w:.1f} ({label:12s}): euro_acc={eu_acc:.4f} euro_ll={eu_ll:.4f} euro_brier={eu_br:.4f} | asian_acc={ah_acc:.4f} asian_ll={ah_ll:.4f}')
