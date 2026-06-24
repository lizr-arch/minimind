"""Convert live odds events to OddsMind-compatible JSON."""
import json, sys
from datetime import datetime
from collections import defaultdict

path = r"D:\code\git\odds-data-probe\odds-data-probe\data\temp\2906969\events.jsonl"

events = []
match_info = None
with open(path, encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        if d.get('type') == 'match_info':
            match_info = d
        elif 'market' in d:
            events.append(d)

kickoff = datetime.strptime('2026-06-23 01:00', '%Y-%m-%d %H:%M')

timeline = defaultdict(lambda: {
    'euro_h': 0.0, 'euro_d': 0.0, 'euro_a': 0.0,
    'asian_line': 0.0, 'upper_water': 0.0, 'lower_water': 0.0,
    'over_under_line': 0.0, 'over_water': 0.0, 'under_water': 0.0,
})

for e in events:
    t_str = e['change_time']
    t = datetime.strptime(f'2026-{t_str}', '%Y-%m-%d %H:%M')
    mins = max(0, int((kickoff - t).total_seconds() / 60))
    
    mkt = e['market']
    if mkt == 'euro':
        timeline[mins]['euro_h'] = e.get('home_odds', 0)
        timeline[mins]['euro_d'] = e.get('draw_odds', 0)
        timeline[mins]['euro_a'] = e.get('away_odds', 0)
    elif mkt == 'ah':
        timeline[mins]['asian_line'] = e.get('line', 0)
        timeline[mins]['upper_water'] = e.get('home_water', 0.9)
        timeline[mins]['lower_water'] = e.get('away_water', 0.9)
    elif mkt == 'ou':
        timeline[mins]['over_under_line'] = e.get('line', 0)
        timeline[mins]['over_water'] = e.get('home_water', 0.9)
        timeline[mins]['under_water'] = e.get('away_water', 0.9)

sorted_mins = sorted(timeline.keys(), reverse=True)
odds_timeline = []
for mins in sorted_mins:
    e = dict(timeline[mins])
    e['minutes_before_kickoff'] = float(mins)
    odds_timeline.append(e)

print(f"Events: {len(odds_timeline)}")
print(f"Range: {odds_timeline[0]['minutes_before_kickoff']:.0f} -> {odds_timeline[-1]['minutes_before_kickoff']:.0f} min")

# Take last 64 closest to kickoff
used = odds_timeline[-64:] if len(odds_timeline) > 64 else odds_timeline

match_json = {
    'match_id': str(match_info['match_id']),
    'league_id': 'worldcup',
    'bookmaker_id': 'titan007',
    'odds_timeline': used,
    'label': {'euro_result': 'home', 'asian_result': 'full_win', 'home_goals': 0, 'away_goals': 0},
}

out_path = 'data/odds_real/live_2906969.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(match_json, f, ensure_ascii=False)
print(f"Saved: {out_path}, timeline: {len(used)} events")
print(f"Last event: {json.dumps(used[-1])}")
