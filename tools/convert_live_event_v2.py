"""Convert live odds events to OddsMind-compatible JSON — forward-fill version."""
import json
from datetime import datetime

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

# Parse all events with minutes-before-kickoff
def parse_change_time(t_str: str):
    """Handle inconsistent formats: '06-22 21:32', '6-22 21:32', '6-8 18:52'."""
    t_str = t_str.strip()
    # Normalize: split into date and time parts
    parts = t_str.split()
    if len(parts) != 2:
        return None
    date_part, time_part = parts
    # Normalize month-day: split by '-', pad month to 2 digits
    md = date_part.split('-')
    if len(md) != 2:
        return None
    month = md[0].zfill(2)
    day = md[1].zfill(2)
    try:
        return datetime.strptime(f'2026-{month}-{day} {time_part}', '%Y-%m-%d %H:%M')
    except ValueError:
        return None

parsed = []
for e in events:
    t = parse_change_time(e['change_time'])
    if t is None:
        continue
    mins = max(0, int((kickoff - t).total_seconds() / 60))
    parsed.append((mins, e))

# Sort by time ascending (earliest first)
parsed.sort(key=lambda x: x[0])

# Forward-fill: maintain latest values for each market
state = {
    'euro_h': 0.0, 'euro_d': 0.0, 'euro_a': 0.0,
    'asian_line': 0.0, 'upper_water': 0.0, 'lower_water': 0.0,
    'over_under_line': 0.0, 'over_water': 0.0, 'under_water': 0.0,
}

# Build timeline: record state at each change point (within 48h of kickoff)
MAX_HOURS = 48
MAX_MINUTES = MAX_HOURS * 60
timeline = []
prev_mins = None
for mins, e in parsed:
    if mins > MAX_MINUTES:
        continue  # skip events older than 48h
    mkt = e['market']
    if mkt == 'euro':
        state['euro_h'] = e.get('home_odds', state['euro_h'])
        state['euro_d'] = e.get('draw_odds', state['euro_d'])
        state['euro_a'] = e.get('away_odds', state['euro_a'])
    elif mkt == 'ah':
        if 'line' in e:
            state['asian_line'] = e['line']
        state['upper_water'] = e.get('home_water', state['upper_water'])
        state['lower_water'] = e.get('away_water', state['lower_water'])
    elif mkt == 'ou':
        if 'line' in e:
            state['over_under_line'] = e['line']
        state['over_water'] = e.get('home_water', state['over_water'])
        state['under_water'] = e.get('away_water', state['under_water'])
    
    # Only record if this is a new minute bucket (deduplicate)
    if prev_mins is None or mins != prev_mins:
        entry = dict(state)
        entry['minutes_before_kickoff'] = float(mins)
        timeline.append(entry)
        prev_mins = mins

# Reverse so most recent first (closest to kickoff last)
# Already in ascending time order, so closest to kickoff is at the end

# Take up to 64 events CLOSEST to kickoff (smallest minutes = beginning of ascending list)
used = timeline[:64] if len(timeline) > 64 else timeline
# The dataset expects events; it will re-sort by minutes_before_kickoff descending internally

print(f"Timeline: {len(timeline)} points -> using {len(used)}")
print(f"Range: {used[0]['minutes_before_kickoff']:.0f} -> {used[-1]['minutes_before_kickoff']:.0f} min before kickoff")
print(f"First: {json.dumps(used[0])}")
print(f"Last: {json.dumps(used[-1])}")

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
print(f"Saved: {out_path}")
