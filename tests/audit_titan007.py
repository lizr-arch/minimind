"""Quick audit of titan007_pure_v2.jsonl"""
import json, sys
from collections import Counter

path = "data/odds_real/titan007_pure_v2.jsonl"
with open(path, 'r', encoding='utf-8') as f:
    lines = [l for l in f if l.strip()]

print(f"Total matches: {len(lines)}")

first = json.loads(lines[0])
print(f"Top-level keys: {sorted(first.keys())}")
print(f"Label keys: {sorted(first.get('label', {}).keys())}")
print(f"Label sample: {json.dumps(first.get('label', {}))}")

tl = first.get('odds_timeline', [])
if tl:
    t0 = tl[0]
    print(f"Timeline keys: {sorted(t0.keys())}")
    print(f"First event: {json.dumps(t0)}")
    print(f"Last event: {json.dumps(tl[-1])}")
    print(f"Timeline length: {len(tl)}")

# Check for over/under
ou_top = [k for k in first.keys() if 'over' in k.lower() or 'under' in k.lower() or 'goal' in k.lower() or 'total' in k.lower()]
print(f"OU top-level keys: {ou_top}")
if tl:
    ou_tl = [k for k in t0.keys() if 'over' in k.lower() or 'under' in k.lower() or 'goal' in k.lower() or 'total' in k.lower()]
    print(f"OU timeline keys: {ou_tl}")

# Leagues
leagues = Counter()
for line in lines[:1000]:
    d = json.loads(line)
    leagues[d.get('league_id', d.get('league', 'unknown'))] += 1
print(f"Leagues (first 1000): {dict(leagues)}")

# Score stats
home_goals, away_goals = [], []
for line in lines:
    d = json.loads(line)
    hg = d.get('label', {}).get('home_goals', -1)
    ag = d.get('label', {}).get('away_goals', -1)
    if hg >= 0: home_goals.append(hg)
    if ag >= 0: away_goals.append(ag)
print(f"Score labels: {len(home_goals)} matches with home_goals")
print(f"Home: mean={sum(home_goals)/len(home_goals):.2f} range=[{min(home_goals)},{max(home_goals)}]")
print(f"Away: mean={sum(away_goals)/len(away_goals):.2f} range=[{min(away_goals)},{max(away_goals)}]")

# Check if score labels present for all
missing = sum(1 for l in lines if json.loads(l).get('label', {}).get('home_goals', -1) < 0)
print(f"Missing score labels: {missing}")
