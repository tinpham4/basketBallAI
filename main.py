import pandas as pd
import numpy as np

# ============================================================================
# SETUP - Change these to match your needs
# ============================================================================

csv_file = "basketball_stats.csv"
position = "C"  # PG, SG, SF, PF, or C

# ============================================================================
# BETTING LINES - Set to None if you don't want to bet on it
# ============================================================================

# Point Guard
PG_lines = {
    'points': 25.5,
    'assists': 7.5,
    'rebounds': 4.5,
    'three_pointers': 2.5,
    'steals': 1.5,
    'turnovers': 3.5,
    'field_goal_attempts': 18.5,
    'pts_reb_ast': 35.5,
    'pts_ast': 30.5,
    'pts_reb': 28.5,
    'reb_ast': 11.5,
}

# Shooting Guard
SG_lines = {
    'points': 22.5,
    'assists': 4.5,
    'rebounds': 4.5,
    'three_pointers': 3.5,
    'steals': 1.5,
    'turnovers': 2.5,
    'field_goal_attempts': 16.5,
    'pts_reb_ast': 32.5,
    'pts_ast': 26.5,
    'pts_reb': 26.5,
    'reb_ast': 8.5,
}

# Small Forward
SF_lines = {
    'points': 20.5,
    'assists': 3.5,
    'rebounds': 6.5,
    'three_pointers': 2.5,
    'steals': 1.5,
    'blocks': 0.5,
    'turnovers': 2.5,
    'field_goal_attempts': 15.5,
    'pts_reb_ast': 30.5,
    'pts_ast': 23.5,
    'pts_reb': 26.5,
    'reb_ast': 9.5,
}

# Power Forward
PF_lines = {
    'points': 18.5,
    'assists': 3.5,
    'rebounds': 8.5,
    'three_pointers': 1.5,
    'blocks': 1.5,
    'steals': 0.5,
    'turnovers': 2.5,
    'field_goal_attempts': 14.5,
    'pts_reb_ast': 30.5,
    'pts_ast': 21.5,
    'pts_reb': 26.5,
    'reb_ast': 11.5,
}

# Center
C_lines = {
    'points': 16.5,
    'rebounds': 10.5,
    'assists': 2.5,
    'blocks': 2.5,
    'steals': 0.5,
    'turnovers': 2.5,
    'field_goal_attempts': 12.5,
    'pts_reb_ast': 29.5,
    'pts_ast': 18.5,
    'pts_reb': 26.5,
    'reb_ast': 12.5,
    'double_double': 0.5,
}

# ============================================================================
# Column name mappings
# ============================================================================

columns = {
    'points': ['PTS', 'pts', 'points'],
    'assists': ['AST', 'ast', 'assists'],
    'rebounds': ['REB', 'reb', 'rebounds', 'TRB'],
    'three_pointers': ['3PT', '3P', '3PM'],
    'steals': ['STL', 'stl', 'steals'],
    'blocks': ['BLK', 'blk', 'blocks'],
    'turnovers': ['TO', 'to', 'turnovers', 'TOV'],
    'field_goal_attempts': ['FGA', 'fga'],
}

min_games = 5
hit_rate_threshold = 0.60

# ============================================================================
# Helper functions
# ============================================================================

def find_col(df, names):
    cols = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in cols:
            return cols[name.lower()]
    return None

def get_combo_stats(df):
    combos = {}
    
    pts_col = find_col(df, columns['points'])
    reb_col = find_col(df, columns['rebounds'])
    ast_col = find_col(df, columns['assists'])
    
    if pts_col and reb_col and ast_col:
        pts = pd.to_numeric(df[pts_col], errors='coerce')
        reb = pd.to_numeric(df[reb_col], errors='coerce')
        ast = pd.to_numeric(df[ast_col], errors='coerce')
        combos['pts_reb_ast'] = (pts + reb + ast).dropna().values
    
    if pts_col and ast_col:
        pts = pd.to_numeric(df[pts_col], errors='coerce')
        ast = pd.to_numeric(df[ast_col], errors='coerce')
        combos['pts_ast'] = (pts + ast).dropna().values
    
    if pts_col and reb_col:
        pts = pd.to_numeric(df[pts_col], errors='coerce')
        reb = pd.to_numeric(df[reb_col], errors='coerce')
        combos['pts_reb'] = (pts + reb).dropna().values
    
    if reb_col and ast_col:
        reb = pd.to_numeric(df[reb_col], errors='coerce')
        ast = pd.to_numeric(df[ast_col], errors='coerce')
        combos['reb_ast'] = (reb + ast).dropna().values
    
    if pts_col and reb_col:
        pts = pd.to_numeric(df[pts_col], errors='coerce')
        reb = pd.to_numeric(df[reb_col], errors='coerce')
        ast = pd.to_numeric(df[ast_col], errors='coerce') if ast_col else pd.Series([0] * len(df))
        
        dd = []
        for i in range(len(df)):
            count = 0
            if pts.iloc[i] >= 10: count += 1
            if reb.iloc[i] >= 10: count += 1
            if ast.iloc[i] >= 10: count += 1
            dd.append(1 if count >= 2 else 0)
        
        combos['double_double'] = np.array(dd)
    
    return combos

def load_data(file, lines):
    try:
        df = pd.read_csv(file)
        print(f"✓ Loaded {len(df)} games")
        print(f"✓ Columns: {list(df.columns)}")
        
        data = {}
        combos = get_combo_stats(df)
        
        for stat, line in lines.items():
            if line is None:
                continue
            
            if stat in combos:
                vals = combos[stat]
                if len(vals) > 0:
                    data[stat] = {'values': vals, 'line': line, 'col': f'{stat} (calc)'}
                    print(f"✓ Calculated '{stat}' ({len(vals)} games)")
                continue
            
            names = columns.get(stat, [stat])
            col = find_col(df, names)
            
            if col:
                vals = pd.to_numeric(df[col], errors='coerce').dropna().values
                if len(vals) > 0:
                    data[stat] = {'values': vals, 'line': line, 'col': col}
                    print(f"✓ Found '{stat}' in '{col}' ({len(vals)} games)")
            else:
                print(f"✗ Couldn't find '{stat}'")
        
        return data
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ============================================================================
# Analysis class
# ============================================================================

class Analyzer:
    def __init__(self, values, line, name):
        self.values = np.array(values)
        self.line = line
        self.name = name
        self.games = len(values)
    
    def stats(self):
        return {
            'games': self.games,
            'avg': float(np.mean(self.values)),
            'median': float(np.median(self.values)),
            'max': float(np.max(self.values)),
            'min': float(np.min(self.values)),
            'std': float(np.std(self.values)),
        }
    
    def hit_rate(self):
        over = int(np.sum(self.values > self.line))
        push = int(np.sum(self.values == self.line))
        under = int(np.sum(self.values < self.line))
        rate = over / self.games if self.games > 0 else 0
        return over, push, under, rate
    
    def recent(self, n):
        if self.games < n:
            return None
        
        last = self.values[-n:]
        w = np.exp(np.linspace(0, 1, n))
        w = w / w.sum()
        
        return {
            'avg': float(np.mean(last)),
            'weighted': float(np.average(last, weights=w)),
            'rate': float(np.sum(last > self.line) / n)
        }
    
    def trend(self, n):
        if self.games < n:
            return None
        last = self.values[-n:]
        x = np.arange(len(last))
        return float(np.polyfit(x, last, 1)[0])
    
    def streak(self):
        over = (self.values > self.line).astype(int)
        length = 0
        direction = None
        
        for val in reversed(over):
            if length == 0:
                length = 1
                direction = 'OVER' if val == 1 else 'UNDER'
            elif (val == 1 and direction == 'OVER') or (val == 0 and direction == 'UNDER'):
                length += 1
            else:
                break
        
        return length, direction
    
    def confidence(self, rate, recent_rate):
        score = 50
        score += (rate - 0.5) * 40
        
        if recent_rate is not None:
            score += (recent_rate - 0.5) * 30
        
        if self.games >= 15:
            score += 10
        elif self.games >= 10:
            score += 5
        elif self.games < min_games:
            score -= 15
        
        s = self.stats()
        cv = s['std'] / s['avg'] if s['avg'] > 0 else 0
        if cv < 0.2:
            score += 10
        elif cv > 0.4:
            score -= 10
        
        dist = abs(s['avg'] - self.line) / self.line if self.line > 0 else 0
        if dist > 0.15:
            score += 5
        
        return max(0, min(100, score))
    
    def ev(self, rate):
        win = 100 / 110
        return (rate * win - (1 - rate) * 1) * 100
    
    def decide(self, rate, conf, ev, recent_rate):
        if self.games < min_games:
            return "NO BET", f"Need {min_games}+ games", 0
        
        signals = 0
        total = 4
        
        if rate >= hit_rate_threshold:
            signals += 1
        if recent_rate is not None:
            if recent_rate > 0.55:
                signals += 1
        else:
            total -= 1
        if ev > 2:
            signals += 1
        if conf >= 55:
            signals += 1
        
        if signals >= total - 1 and conf >= 55:
            bet = "BET OVER" if rate >= hit_rate_threshold else "BET UNDER"
            reason = f"Strong ({signals}/{total} signals)"
            strength = conf
        elif signals >= total - 2 and conf >= 45:
            bet = "LEAN OVER" if rate >= hit_rate_threshold else "LEAN UNDER"
            reason = f"Moderate ({signals}/{total} signals)"
            strength = conf
        else:
            bet = "NO BET"
            reason = f"Weak ({signals}/{total} signals)"
            strength = 0
        
        return bet, reason, strength
    
    def analyze(self):
        s = self.stats()
        over, push, under, rate = self.hit_rate()
        last3 = self.recent(3)
        last5 = self.recent(5)
        t = self.trend(5)
        streak_len, streak_dir = self.streak()
        
        recent_rate = last5['rate'] if last5 else None
        conf = self.confidence(rate, recent_rate)
        ev = self.ev(rate)
        decision, reason, strength = self.decide(rate, conf, ev, recent_rate)
        
        return {
            'stats': s, 'over': over, 'push': push, 'under': under, 'rate': rate,
            'last3': last3, 'last5': last5, 'trend': t, 
            'streak_len': streak_len, 'streak_dir': streak_dir,
            'conf': conf, 'ev': ev, 'decision': decision, 'reason': reason, 'strength': strength
        }

def show_results(name, r, line):
    print("\n" + "="*70)
    print(f"{name.upper().replace('_', ' '):^70}")
    print("="*70)
    
    s = r['stats']
    print(f"\n📊 LINE: {line}")
    print(f"🎮 GAMES: {s['games']}")
    
    print(f"\n{'STATS':-^70}")
    print(f"Average:           {s['avg']:.2f}")
    print(f"Median:            {s['median']:.2f}")
    print(f"Range:             {s['min']:.2f} - {s['max']:.2f}")
    print(f"Std Dev:           {s['std']:.2f}")
    
    print(f"\n{'HIT RATE':-^70}")
    print(f"Over {line}:          {r['over']} ({r['rate']:.1%})")
    print(f"Push at {line}:       {r['push']}")
    print(f"Under {line}:         {r['under']} ({r['under']/s['games']:.1%})")
    print(f"Streak:            {r['streak_len']} games {r['streak_dir']}")
    
    if r['last3']:
        print(f"\n{'LAST 3':-^70}")
        print(f"Average:           {r['last3']['avg']:.2f}")
        print(f"Weighted:          {r['last3']['weighted']:.2f}")
        print(f"Hit Rate:          {r['last3']['rate']:.1%}")
    
    if r['last5']:
        print(f"\n{'LAST 5':-^70}")
        print(f"Average:           {r['last5']['avg']:.2f}")
        print(f"Weighted:          {r['last5']['weighted']:.2f}")
        print(f"Hit Rate:          {r['last5']['rate']:.1%}")
    
    if r['trend'] is not None:
        direction = "↗️ Up" if r['trend'] > 0 else "↘️ Down"
        print(f"\n{'TREND':-^70}")
        print(f"5-Game:            {direction} ({r['trend']:+.2f} per game)")
    
    print(f"\n{'METRICS':-^70}")
    print(f"Confidence:        {r['conf']:.0f}/100")
    print(f"Expected Value:    {r['ev']:+.2f}%")
    
    print(f"\n{'CALL':-^70}")
    emoji = "🟢" if "BET" in r['decision'] and "NO" not in r['decision'] else "🟡" if "LEAN" in r['decision'] else "🔴"
    print(f"{emoji} {r['decision']}")
    print(f"📝 {r['reason']}")
    if r['strength'] > 0:
        print(f"💪 {r['strength']:.0f}/100")

# ============================================================================
# Run it
# ============================================================================

if position == "PG":
    lines = PG_lines
elif position == "SG":
    lines = SG_lines
elif position == "SF":
    lines = SF_lines
elif position == "PF":
    lines = PF_lines
elif position == "C":
    lines = C_lines
else:
    print(f"❌ Bad position: {position}")
    exit()

print("\n" + "="*70)
print(f"{'BASKETBALL BETTING ANALYZER':^70}")
print(f"{position:^70}")
print("="*70 + "\n")

data = load_data(csv_file, lines)
if not data:
    print("\n❌ No data")
    exit()

print("\n" + "="*70)
print("ANALYZING...")
print("="*70)

bets = []
for name, info in data.items():
    a = Analyzer(info['values'], info['line'], name)
    r = a.analyze()
    show_results(name, r, info['line'])
    
    if r['decision'] != "NO BET":
        bets.append({
            'name': name.replace('_', ' ').title(),
            'decision': r['decision'],
            'conf': r['conf'],
            'rate': r['rate'],
            'ev': r['ev']
        })

if bets:
    print("\n" + "="*70)
    print(f"{'SUMMARY':^70}")
    print("="*70)
    
    bets.sort(key=lambda x: x['conf'], reverse=True)
    
    for b in bets:
        emoji = "🟢" if "BET" in b['decision'] else "🟡"
        print(f"{emoji} {b['name']}: {b['decision']}")
        print(f"   Conf: {b['conf']:.0f} | Rate: {b['rate']:.1%} | EV: {b['ev']:+.1f}%")
        print()