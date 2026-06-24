"""Check for missing/zero odds in timeline events."""
import json

for path in ['data/odds_real/titan007_pure_v2.jsonl', 'data/odds_real/v5_b365.jsonl']:
    print(f"\n=== {path} ===")
    total_events = 0
    zero_euro = 0  # all three euro fields are 0
    zero_asian = 0  # asian_line is 0
    zero_ou = 0     # over_under_line is 0
    samples_zero = []
    
    for line in open(path, encoding='utf-8'):
        d = json.loads(line)
        for e in d.get('odds_timeline', []):
            total_events += 1
            eh = e.get('euro_h', -1)
            ed = e.get('euro_d', -1)
            ea = e.get('euro_a', -1)
            al = e.get('asian_line', -1)
            ou = e.get('over_under_line', -1)
            
            if eh <= 0.001 and ed <= 0.001 and ea <= 0.001:
                zero_euro += 1
            if al == 0:
                zero_asian += 1
            if ou == 0:
                zero_ou += 1
            
            if zero_euro <= 3 and eh <= 0.001:
                samples_zero.append((d.get('match_id','?'), e))
    
    print(f"  Total events: {total_events}")
    print(f"  All euro=0: {zero_euro} ({100*zero_euro/max(1,total_events):.1f}%)")
    print(f"  asian_line=0: {zero_asian} ({100*zero_asian/max(1,total_events):.1f}%)")
    print(f"  over_under_line=0: {zero_ou} ({100*zero_ou/max(1,total_events):.1f}%)")
    
    if samples_zero:
        print(f"  Sample zero-euro event: {json.dumps(samples_zero[0][1])}")
    
    # Also check: any event that has euro but NOT asian, or vice versa
    has_euro_no_asian = 0
    has_asian_no_euro = 0
    for line in open(path, encoding='utf-8'):
        d = json.loads(line)
        for e in d.get('odds_timeline', []):
            has_e = (e.get('euro_h', 0) > 0.01)
            has_a = (e.get('asian_line', -99) != 0)
            if has_e and not has_a: has_euro_no_asian += 1
            if has_a and not has_e: has_asian_no_euro += 1
    print(f"  Has Euro but no Asian: {has_euro_no_asian}")
    print(f"  Has Asian but no Euro: {has_asian_no_euro}")
