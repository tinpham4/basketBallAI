import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================
PLAYER_NAME = "Tyrese Maxey"
# CSV File Settings
CSV_PATH = "basketball_stats.csv"

# Player Position: "PG", "SG", "SF", "PF", "C"
POSITION = "SF"

# ============================================================================
# BETTING LINES - Only set lines for stats you want to analyze
# Set to None if you don't want to analyze that stat
# ============================================================================

# POINT GUARD LINES
PG_LINES = {
    'points': 30.5,
    'assists': 7.0,
    'rebounds': 4.5,
    'three_pointers': 3.5,
    'steals': 0.5,
    'turnovers': 2.5,
    'field_goal_attempts': 23.5,
    'pts_reb_ast': 41.5,  # Combined stat
    'pts_ast': 37.5,
    'pts_reb': 34.5,
    'reb_ast': 11.5,
}

# SHOOTING GUARD LINES
SG_LINES = {
    'points': 17.5,
    'assists': 5.5,
    'rebounds': 4.0,
    'three_pointers': 3.0,
    'steals': 1.5,
    'turnovers': 1.5,
    'field_goal_attempts': 14.5,
    'pts_reb_ast': 27.5,
    'pts_ast': 23.5,
    'pts_reb': 22.5,
    'reb_ast': 9.5,
}

# SMALL FORWARD LINES
SF_LINES = {
    'points': 25.5,
    'assists': 5.5,
    'rebounds': 6.5,
    'three_pointers': 3.5,
    'steals': 1.5,
    'blocks': 1.5,
    'turnovers': None,
    'field_goal_attempts': 17.5,
    'pts_reb_ast': 38.5,
    'pts_ast': 31.5,
    'pts_reb': 32.5,
    'reb_ast': 12.5,
}

# POWER FORWARD LINES
PF_LINES = {
    'points': 22.5,
    'assists': 5.5,
    'rebounds': 6.5,
    'three_pointers': 1.5,
    'blocks': 0.5,
    'steals': 1.5,
    'turnovers': 2.5,
    'field_goal_attempts': 14.5,
    'pts_reb_ast': 36.5,
    'pts_ast': 28.5,
    'pts_reb': 30.5,
    'reb_ast': 12.5,
    'double_double': 0.5,
}

# CENTER LINES
C_LINES = {
    'points': 28.5,
    'rebounds': 10.5,
    'assists': 10.5,
    'blocks': 1.5,
    'steals': 1.5,
    'turnovers': 2.5,
    'field_goal_attempts': 17.5,
    'pts_reb_ast': 51.5,
    'pts_ast': 39.5,
    'pts_reb': 41.5,
    'reb_ast': 23.5,
    'double_double': None,
}

# ============================================================================
# COLUMN MAPPINGS - Map your CSV columns to stat types
# ============================================================================

COLUMN_MAPPINGS = {
    'points': ['PTS', 'pts', 'points', 'Points'],
    'assists': ['AST', 'ast', 'assists', 'Assists'],
    'rebounds': ['REB', 'reb', 'rebounds', 'Rebounds', 'TRB'],
    'three_pointers': ['3PT', '3P', '3PM', 'threes', 'three_pointers'],
    'steals': ['STL', 'stl', 'steals', 'Steals'],
    'blocks': ['BLK', 'blk', 'blocks', 'Blocks'],
    'turnovers': ['TO', 'to', 'turnovers', 'Turnovers', 'TOV'],
    'field_goals': ['FG', 'fg', 'field_goals', 'FGM'],
    'field_goal_attempts': ['FGA', 'fga', 'field_goal_attempts', 'fg_att'],
    'free_throws': ['FT', 'ft', 'free_throws', 'FTM'],
    'minutes': ['MIN', 'min', 'minutes', 'MP'],
    'offensive_rebounds': ['OR', 'OREB', 'off_reb'],
    'defensive_rebounds': ['DR', 'DREB', 'def_reb'],
    'personal_fouls': ['PF', 'pf', 'fouls', 'personal_fouls'],
}

# ============================================================================
# SETTINGS
# ============================================================================
THRESHOLD = 0.60  # Minimum hit rate for strong recommendation
MIN_GAMES = 5
CONFIDENCE_HIGH = 65
CONFIDENCE_MEDIUM = 55

# ============================================================================
# ANALYSIS ENGINE
# ============================================================================

def find_column(df, possible_names):
    """Find a column in the dataframe from a list of possible names."""
    df_columns_lower = {col.lower(): col for col in df.columns}
    for name in possible_names:
        if name.lower() in df_columns_lower:
            return df_columns_lower[name.lower()]
    return None

def calculate_combined_stats(df):
    """Calculate combined stats like PTS+REB+AST."""
    combined = {}
    
    # Find necessary columns
    pts_col = find_column(df, COLUMN_MAPPINGS['points'])
    reb_col = find_column(df, COLUMN_MAPPINGS['rebounds'])
    ast_col = find_column(df, COLUMN_MAPPINGS['assists'])
    
    # PTS + REB + AST
    if pts_col and reb_col and ast_col:
        pts = pd.to_numeric(df[pts_col], errors='coerce')
        reb = pd.to_numeric(df[reb_col], errors='coerce')
        ast = pd.to_numeric(df[ast_col], errors='coerce')
        combined['pts_reb_ast'] = (pts + reb + ast).dropna().values
    
    # PTS + AST
    if pts_col and ast_col:
        pts = pd.to_numeric(df[pts_col], errors='coerce')
        ast = pd.to_numeric(df[ast_col], errors='coerce')
        combined['pts_ast'] = (pts + ast).dropna().values
    
    # PTS + REB
    if pts_col and reb_col:
        pts = pd.to_numeric(df[pts_col], errors='coerce')
        reb = pd.to_numeric(df[reb_col], errors='coerce')
        combined['pts_reb'] = (pts + reb).dropna().values
    
    # REB + AST
    if reb_col and ast_col:
        reb = pd.to_numeric(df[reb_col], errors='coerce')
        ast = pd.to_numeric(df[ast_col], errors='coerce')
        combined['reb_ast'] = (reb + ast).dropna().values
    
    # Double-Double (10+ points and 10+ rebounds, or 10+ points and 10+ assists, etc.)
    if pts_col and reb_col:
        pts = pd.to_numeric(df[pts_col], errors='coerce')
        reb = pd.to_numeric(df[reb_col], errors='coerce')
        ast = pd.to_numeric(df[ast_col], errors='coerce') if ast_col else pd.Series([0] * len(df))
        
        double_double = []
        for i in range(len(df)):
            stats_over_10 = 0
            if pts.iloc[i] >= 10:
                stats_over_10 += 1
            if reb.iloc[i] >= 10:
                stats_over_10 += 1
            if ast.iloc[i] >= 10:
                stats_over_10 += 1
            double_double.append(1 if stats_over_10 >= 2 else 0)
        
        combined['double_double'] = np.array(double_double)
    
    return combined

def load_stats_from_csv(csv_path, lines_dict):
    """Load stats from CSV based on the lines provided."""
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV with {len(df)} games")
        print(f"✓ Available columns: {list(df.columns)}")
        
        stats_data = {}
        
        # Calculate combined stats first
        combined_stats = calculate_combined_stats(df)
        
        for stat_name, line in lines_dict.items():
            if line is None:
                continue
            
            # Check if it's a combined stat
            if stat_name in combined_stats:
                data = combined_stats[stat_name]
                if len(data) > 0:
                    stats_data[stat_name] = {
                        'data': data,
                        'line': line,
                        'column': f'{stat_name} (calculated)'
                    }
                    print(f"✓ Calculated '{stat_name}' ({len(data)} games)")
                continue
            
            # Find the column in the CSV
            possible_columns = COLUMN_MAPPINGS.get(stat_name, [stat_name])
            col_name = find_column(df, possible_columns)
            
            if col_name:
                # Convert to numeric and remove invalid entries
                data = pd.to_numeric(df[col_name], errors='coerce')
                data = data.dropna().values
                
                if len(data) > 0:
                    stats_data[stat_name] = {
                        'data': data,
                        'line': line,
                        'column': col_name
                    }
                    print(f"✓ Found '{stat_name}' in column '{col_name}' ({len(data)} games)")
                else:
                    print(f"⚠ Column '{col_name}' exists but has no valid data")
            else:
                print(f"✗ Could not find '{stat_name}' in CSV")
        
        return stats_data, df
    
    except FileNotFoundError:
        print(f"❌ Error: File '{csv_path}' not found")
        return None, None
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None, None

class StatAnalyzer:
    def __init__(self, data, line, stat_name):
        self.data = np.array(data)
        self.line = line
        self.stat_name = stat_name
        self.games = len(data)
        
    def calculate_basic_stats(self):
        return {
            'games': self.games,
            'avg': float(np.mean(self.data)),
            'median': float(np.median(self.data)),
            'max': float(np.max(self.data)),
            'min': float(np.min(self.data)),
            'std': float(np.std(self.data)),
        }
    
    def calculate_hit_rate(self):
        hits = int(np.sum(self.data > self.line))
        pushes = int(np.sum(self.data == self.line))
        misses = int(np.sum(self.data < self.line))
        hit_rate = hits / self.games if self.games > 0 else 0
        return hits, pushes, misses, hit_rate
    
    def analyze_recent(self, n_games):
        if self.games < n_games:
            return None
        
        recent = self.data[-n_games:]
        weights = np.exp(np.linspace(0, 1, n_games))
        weights = weights / weights.sum()
        
        return {
            'simple_avg': float(np.mean(recent)),
            'weighted_avg': float(np.average(recent, weights=weights)),
            'hit_rate': float(np.sum(recent > self.line) / n_games)
        }
    
    def calculate_trend(self, n_games):
        if self.games < n_games:
            return None
        recent = self.data[-n_games:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope)
    
    def calculate_streak(self):
        """Calculate current streak over/under the line."""
        over = (self.data > self.line).astype(int)
        
        # Current streak
        current_streak = 0
        streak_type = None
        for val in reversed(over):
            if current_streak == 0:
                current_streak = 1
                streak_type = 'OVER' if val == 1 else 'UNDER'
            elif (val == 1 and streak_type == 'OVER') or (val == 0 and streak_type == 'UNDER'):
                current_streak += 1
            else:
                break
        
        return current_streak, streak_type
    
    def calculate_confidence(self, hit_rate, recent_hit_rate):
        confidence = 50
        confidence += (hit_rate - 0.5) * 40
        
        if recent_hit_rate is not None:
            confidence += (recent_hit_rate - 0.5) * 30
        
        if self.games >= 15:
            confidence += 10
        elif self.games >= 10:
            confidence += 5
        elif self.games < MIN_GAMES:
            confidence -= 15
        
        stats = self.calculate_basic_stats()
        cv = stats['std'] / stats['avg'] if stats['avg'] > 0 else 0
        if cv < 0.2:
            confidence += 10
        elif cv > 0.4:
            confidence -= 10
        
        line_distance = abs(stats['avg'] - self.line) / self.line if self.line > 0 else 0
        if line_distance > 0.15:
            confidence += 5
        
        return max(0, min(100, confidence))
    
    def calculate_ev(self, hit_rate, odds=-110):
        if odds < 0:
            win_amount = 100 / abs(odds)
        else:
            win_amount = odds / 100
        ev = (hit_rate * win_amount) - ((1 - hit_rate) * 1)
        return ev * 100
    
    def make_decision(self, hit_rate, confidence, ev, recent_hit_rate):
        if self.games < MIN_GAMES:
            return "NO BET", f"Need {MIN_GAMES}+ games", 0
        
        signals = 0
        total_signals = 4
        
        if hit_rate >= THRESHOLD:
            signals += 1
        
        if recent_hit_rate is not None:
            if recent_hit_rate > 0.55:
                signals += 1
        else:
            total_signals -= 1
        
        if ev > 2:
            signals += 1
        
        if confidence >= CONFIDENCE_MEDIUM:
            signals += 1
        
        if signals >= total_signals - 1 and confidence >= CONFIDENCE_MEDIUM:
            recommendation = "BET OVER" if hit_rate >= THRESHOLD else "BET UNDER"
            reason = f"Strong edge ({signals}/{total_signals} signals)"
            strength = confidence
        elif signals >= total_signals - 2 and confidence >= CONFIDENCE_MEDIUM - 10:
            recommendation = "LEAN OVER" if hit_rate >= THRESHOLD else "LEAN UNDER"
            reason = f"Moderate edge ({signals}/{total_signals} signals)"
            strength = confidence
        else:
            recommendation = "NO BET"
            reason = f"Weak edge ({signals}/{total_signals} signals)"
            strength = 0
        
        return recommendation, reason, strength
    
    def full_analysis(self):
        stats = self.calculate_basic_stats()
        hits, pushes, misses, hit_rate = self.calculate_hit_rate()
        
        last3 = self.analyze_recent(3)
        last5 = self.analyze_recent(5)
        
        trend5 = self.calculate_trend(5)
        streak_count, streak_type = self.calculate_streak()
        
        recent_hr = last5['hit_rate'] if last5 else None
        confidence = self.calculate_confidence(hit_rate, recent_hr)
        ev = self.calculate_ev(hit_rate)
        
        decision, reason, strength = self.make_decision(hit_rate, confidence, ev, recent_hr)
        
        return {
            'stats': stats,
            'hits': hits,
            'pushes': pushes,
            'misses': misses,
            'hit_rate': hit_rate,
            'last3': last3,
            'last5': last5,
            'trend5': trend5,
            'streak_count': streak_count,
            'streak_type': streak_type,
            'confidence': confidence,
            'ev': ev,
            'decision': decision,
            'reason': reason,
            'strength': strength
        }

def print_stat_analysis(stat_name, results, line):
    """Print analysis for a single stat."""
    print("\n" + "="*70)
    print(f"{stat_name.upper().replace('_', ' '):^70}")
    print("="*70)
    
    stats = results['stats']
    print(f"\n📊 LINE: {line}")
    print(f"🎮 GAMES: {stats['games']}")
    
    print(f"\n{'STATISTICS':-^70}")
    print(f"Average:           {stats['avg']:.2f}")
    print(f"Median:            {stats['median']:.2f}")
    print(f"Range:             {stats['min']:.2f} - {stats['max']:.2f}")
    print(f"Std Deviation:     {stats['std']:.2f}")
    
    print(f"\n{'HIT RATE':-^70}")
    print(f"Hits over {line}:     {results['hits']} ({results['hit_rate']:.1%})")
    print(f"Pushes at {line}:     {results['pushes']}")
    print(f"Misses under {line}:  {results['misses']} ({results['misses']/stats['games']:.1%})")
    print(f"Current Streak:    {results['streak_count']} games {results['streak_type']}")
    
    if results['last3']:
        print(f"\n{'LAST 3 GAMES':-^70}")
        print(f"Average:           {results['last3']['simple_avg']:.2f}")
        print(f"Weighted Avg:      {results['last3']['weighted_avg']:.2f}")
        print(f"Hit Rate:          {results['last3']['hit_rate']:.1%}")
    
    if results['last5']:
        print(f"\n{'LAST 5 GAMES':-^70}")
        print(f"Average:           {results['last5']['simple_avg']:.2f}")
        print(f"Weighted Avg:      {results['last5']['weighted_avg']:.2f}")
        print(f"Hit Rate:          {results['last5']['hit_rate']:.1%}")
    
    if results['trend5'] is not None:
        trend_dir = "↗️ Up" if results['trend5'] > 0 else "↘️ Down"
        print(f"\n{'TREND':-^70}")
        print(f"5-Game Trend:      {trend_dir} ({results['trend5']:+.2f} per game)")
    
    print(f"\n{'METRICS':-^70}")
    print(f"Confidence:        {results['confidence']:.0f}/100")
    print(f"Expected Value:    {results['ev']:+.2f}%")
    
    print(f"\n{'RECOMMENDATION':-^70}")
    color = "🟢" if "BET" in results['decision'] and "NO" not in results['decision'] else "🟡" if "LEAN" in results['decision'] else "🔴"
    print(f"{color} {results['decision']}")
    print(f"📝 {results['reason']}")
    if results['strength'] > 0:
        print(f"💪 Strength: {results['strength']:.0f}/100")

# ============================================================================
# RUN ANALYSIS
# ============================================================================

# Select lines based on position
if POSITION == "PG":
    selected_lines = PG_LINES
elif POSITION == "SG":
    selected_lines = SG_LINES
elif POSITION == "SF":
    selected_lines = SF_LINES
elif POSITION == "PF":
    selected_lines = PF_LINES
elif POSITION == "C":
    selected_lines = C_LINES
else:
    print(f"❌ Invalid position: {POSITION}")
    print("Valid positions: PG, SG, SF, PF, C")
    exit()

print("\n" + "="*70)
print(f"{'BASKETBALL BETTING ANALYSIS':^70}")
print(f"{POSITION + ' POSITION':^70}")
print("="*70 + "\n")

# Load stats from CSV
stats_data, df = load_stats_from_csv(CSV_PATH, selected_lines)

if not stats_data:
    print("\n❌ No valid stats found to analyze")
    exit()

# Analyze each available stat
print("\n" + "="*70)
print("ANALYZING STATS...")
print("="*70)

recommendations = []
for stat_name, stat_info in stats_data.items():
    analyzer = StatAnalyzer(stat_info['data'], stat_info['line'], stat_name)
    results = analyzer.full_analysis()
    print_stat_analysis(stat_name, results, stat_info['line'])
    
    if results['decision'] != "NO BET":
        recommendations.append({
            'stat': stat_name.replace('_', ' ').title(),
            'decision': results['decision'],
            'confidence': results['confidence'],
            'hit_rate': results['hit_rate'],
            'ev': results['ev']
        })

# Summary
if recommendations:
    print("\n" + "="*70)
    print(f"{'SUMMARY - ALL RECOMMENDATIONS':^70}")
    print(PLAYER_NAME)
    print("="*70)
    
    recommendations.sort(key=lambda x: x['confidence'], reverse=True)
    
    for rec in recommendations:
        emoji = "🟢" if "BET" in rec['decision'] else "🟡"
        print(f"{emoji} {rec['stat']}: {rec['decision']}")
        print(f"   Confidence: {rec['confidence']:.0f}/100 | Hit Rate: {rec['hit_rate']:.1%} | EV: {rec['ev']:+.1f}%")