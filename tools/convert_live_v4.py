"""Convert live odds v4 format — filter post-kickoff events."""
import json
from datetime import datetime

with open(r"D:\code\git\odds-data-probe\odds-data-probe\data\temp\2906969\events.jsonl", encoding='utf-8') as f:
    data = json.load(f)

kickoff = datetime.strptime(data['kickoff'], '%Y-%m-%d %H:%M')

# Parse all events, compute minutes-before-kickoff
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
    mins = (kickoff - t).total_seconds() / 60  # positive = before kickoff, negative = after
    parsed.append((mins, e))

# Sort by time DESCENDING (oldest first, newest last) for forward-fill
parsed.sort(key=lambda x: x[0], reverse=True)

# Forward-fill, only pre-kickoff
state = {
    'euro_h': 0.0, 'euro_d': 0.0, 'euro_a': 0.0,
    'asian_line': 0.0, 'upper_water': 0.0, 'lower_water': 0.0,
    'over_under_line': 0.0, 'over_water': 0.0, 'under_water': 0.0,
}

timeline = []
prev_mins = None
pre_count = 0
post_count = 0

for mins, e in parsed:
    if mins < 0:  # after kickoff — skip
        post_count += 1
        continue
    pre_count += 1
    # Only last 48h
    if mins > 2880:
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

# Take 64 most recent pre-kickoff (end of list = closest to kickoff)
used = timeline[-64:] if len(timeline) > 64 else timeline

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

print(f"Events: {pre_count} pre-kickoff, {post_count} post-kickoff (skipped)")
print(f"Timeline: {len(timeline)} points -> using {len(used)}")
mins_used = [e['minutes_before_kickoff'] for e in used]
if mins_used:
    print(f"Minutes range: {min(mins_used):.0f} -> {max(mins_used):.0f} before kickoff")
    last = used[-1]
    print(f"Latest pre-kickoff ({last['minutes_before_kickoff']:.0f}min): euro={last['euro_h']}/{last['euro_d']}/{last['euro_a']} ah={last['asian_line']} ou={last['over_under_line']}")
print(f"Saved: {out_path}")
