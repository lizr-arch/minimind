"""Convert new format live odds to OddsMind JSON."""
import json
from datetime import datetime

with open(r"D:\code\git\odds-data-probe\odds-data-probe\data\temp\2906969\events.jsonl", encoding='utf-8') as f:
    data = json.load(f)

kickoff = datetime.strptime(data['kickoff'], '%Y-%m-%d %H:%M')

# Initialize state from summary latest values
s = data['summary']
state = {
    'euro_h': s['euro']['latest']['home'],
    'euro_d': s['euro']['latest']['draw'],
    'euro_a': s['euro']['latest']['away'],
    'asian_line': s['ah']['latest']['line'],
    'upper_water': s['ah']['latest']['home_water'],
    'lower_water': s['ah']['latest']['away_water'],
    'over_under_line': s['ou']['latest']['line'],
    'over_water': s['ou']['latest']['over'],
    'under_water': s['ou']['latest']['under'],
}

# Parse timeline events
parsed = []
for e in data.get('timeline', []):
    t_str = e['change_time'].strip()
    parts = t_str.split()
    if len(parts) != 2:
        continue
    md = parts[0].split('-')
    if len(md) != 2:
        continue
    month = md[0].zfill(2)
    day = md[1].zfill(2)
    try:
        t = datetime.strptime(f'2026-{month}-{day} {parts[1]}', '%Y-%m-%d %H:%M')
    except ValueError:
        continue
    mins = max(0, int((kickoff - t).total_seconds() / 60))
    parsed.append((mins, e))

parsed.sort(key=lambda x: x[0])

# Forward-fill into timeline (last 48h only)
MAX_MINS = 48 * 60
timeline = []
prev_mins = None
for mins, e in parsed:
    if mins > MAX_MINS:
        continue
    if mins < 0:  # skip post-kickoff events
        continue
    mkt = e['market']
    if mkt == 'euro':
        state['euro_h'] = e.get('home', state['euro_h'])
        state['euro_d'] = e.get('draw', state['euro_d'])
        state['euro_a'] = e.get('away', state['euro_a'])
    elif mkt == 'ah':
        if 'line' in e:
            state['asian_line'] = e['line']
        state['upper_water'] = e.get('home_water', state['upper_water'])
        state['lower_water'] = e.get('away_water', state['lower_water'])
    elif mkt == 'ou':
        if 'line' in e:
            state['over_under_line'] = e['line']
        state['over_water'] = e.get('over', state['over_water'])
        state['under_water'] = e.get('under', state['under_water'])
    
    if prev_mins is None or mins != prev_mins:
        entry = dict(state)
        entry['minutes_before_kickoff'] = float(mins)
        timeline.append(entry)
        prev_mins = mins

# Take 64 most recent (smallest minutes)
used = timeline[:64] if len(timeline) > 64 else timeline

match_json = {
    'match_id': str(data['match_id']),
    'league_id': 'worldcup',
    'bookmaker_id': data.get('bookmaker', 'Bet365'),
    'odds_timeline': used,
    'label': {'euro_result': 'home', 'asian_result': 'full_win', 'home_goals': 0, 'away_goals': 0},
}

out_path = 'data/odds_real/live_2906969.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(match_json, f, ensure_ascii=False)

# Summary
print(f"Summary: euro={data['summary']['euro']['latest']}")
print(f"         ah={data['summary']['ah']['latest']}")
print(f"         ou={data['summary']['ou']['latest']}")
print(f"Timeline events: {len(timeline)} -> using {len(used)}")
mins_used = [e['minutes_before_kickoff'] for e in used]
print(f"Minutes range: {min(mins_used):.0f} -> {max(mins_used):.0f} before kickoff")
last = used[0]
print(f"Most recent event ({last['minutes_before_kickoff']:.0f}min): euro={last['euro_h']}/{last['euro_d']}/{last['euro_a']} ah={last['asian_line']} ou={last['over_under_line']}")
print(f"Saved: {out_path}")
