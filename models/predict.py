import fastf1
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
from collections import defaultdict

fastf1.Cache.enable_cache("data/raw")

#The race one wants to predict 
SEASON      = 2026
ROUND       = 1

N_SIMULATIONS = 1000
MODEL_DIR     = Path("models")
SESSIONS      = ['FP1', 'FP2', 'FP3', 'Q']


#Data Fetching

def fetch_session_times(season, round_num):
    all_laps = []

    for session_name in SESSIONS:
        try:
            print(f"  Fetching {session_name}...", end=" ")
            session = fastf1.get_session(season, round_num, session_name)
            session.load(telemetry=False, laps=True, weather=False)

            laps = session.laps[['Driver', 'LapTime']].copy()
            laps['LapTime'] = laps['LapTime'].dt.total_seconds()
            best = laps.groupby('Driver')['LapTime'].min().reset_index()
            best.columns = ['Driver', session_name]
            all_laps.append(best)

        except Exception as e:
            print(f" ({e})")
            all_laps.append(None)

    df = None
    for i, session_name in enumerate(SESSIONS):
        if all_laps[i] is None:
            continue
        df = all_laps[i] if df is None else df.merge(all_laps[i], on='Driver', how='outer')

    if df is None:
        raise ValueError("No session data could be fetched.")

    df = df.rename(columns={
        'FP1': 'fp1_time',
        'FP2': 'fp2_time',
        'FP3': 'fp3_time',
        'Q':   'quali_time'
    })

    return df


def fetch_grid_position(season, round_num):
    try:
        print(f"  Fetching grid positions...", end=" ")
        session = fastf1.get_session(season, round_num, 'Q')
        session.load(telemetry=False, laps=False, weather=False)

        grid = session.results[['Abbreviation', 'Position']].copy()
        grid = grid.rename(columns={
            'Abbreviation': 'Driver',
            'Position':     'GridPosition'
        })
        grid['GridPosition'] = pd.to_numeric(grid['GridPosition'], errors='coerce')

        return grid

    except Exception as e:
        print(f" ({e})")
        return None


def get_track_id(season, round_num):
    try:
        schedule   = fastf1.get_event_schedule(season, include_testing=False)
        event_row  = schedule[schedule['RoundNumber'] == round_num].iloc[0]
        event_name = event_row['EventName']

        df_train  = pd.read_csv("data/processed/f1_dataset.csv")
        track_map = dict(enumerate(df_train['EventName'].astype('category').cat.categories))
        rev_map   = {v: k for k, v in track_map.items()}
        track_id  = rev_map.get(event_name, -1)

        if track_id == -1:
            print(f" '{event_name}' not in training data — TrackID set to -1")
        else:
            print(f"  Track: {event_name} → TrackID {track_id} ✓")

        return event_name, track_id

    except Exception as e:
        print(f"  Could not resolve TrackID: {e}")
        return "Unknown", -1


#Feature Engineering

def compute_deltas(df):
    session_map = {
        'fp1_time':   'fp1_delta',
        'fp2_time':   'fp2_delta',
        'fp3_time':   'fp3_delta',
        'quali_time': 'quali_delta',
    }
    for raw_col, delta_col in session_map.items():
        if raw_col in df.columns:
            best = df[raw_col].min()
            df[delta_col] = (df[raw_col] - best).round(4)

    return df


def compute_grid_penalty(df):
    if 'quali_delta' in df.columns and 'GridPosition' in df.columns:
        # Use Int64 (nullable integer) instead of int to handle NaN rows
        df['quali_rank'] = (
            df['quali_delta']
            .rank(method='min', na_option='bottom')  # push NaN drivers to back
            .astype('Int64')
        )
        df['grid_penalty'] = (
            (df['GridPosition'] - df['quali_rank'])
            .fillna(0)
            .astype(int)
        )
    else:
        df['quali_rank']   = pd.NA
        df['grid_penalty'] = 0

    return df


def build_features(times_df, grid_df, track_id):
    df = times_df.copy()

    if grid_df is not None:
        df = df.merge(grid_df, on='Driver', how='left')
    else:
        df['GridPosition'] = np.nan

    df['TrackID'] = track_id

    # Fill missing practice times with session median
    for col in ['fp1_time', 'fp2_time', 'fp3_time']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = compute_deltas(df)
    df = compute_grid_penalty(df)

    return df


#Monte Carlo Simulation - to get a better predicted value

def get_raw_probs(feature_df, model_path, label):
    """Get raw model probabilities for each driver."""
    if not model_path.exists():
        print(f"  Model not found: {model_path} — run train.py first")
        return None

    pipeline = joblib.load(model_path)
    features = ['fp1_delta', 'fp2_delta', 'fp3_delta', 'quali_delta',
                'GridPosition', 'grid_penalty', 'TrackID']
    available = [f for f in features if f in feature_df.columns]

    probs = pipeline.predict_proba(feature_df[available])[:, 1]

    result = feature_df[['Driver', 'GridPosition']].copy()
    result[f'{label}_raw_prob'] = probs

    return result


def normalise_probs(probs_array):
    """
    Normalise raw probabilities so they sum to 1 across all drivers.
    This treats the race as a single competition where probabilities
    are dependent — one driver winning reduces others' chances.
    """
    total = probs_array.sum()
    if total == 0:
        return np.ones(len(probs_array)) / len(probs_array)
    return probs_array / total


def simulate_race(winner_probs, podium_probs, drivers, n_sims):
    win_counts    = defaultdict(int)
    podium_counts = defaultdict(int)
    pos_counts    = defaultdict(lambda: defaultdict(int)) 

    n_drivers = len(drivers)

    # Normalise so probabilities sum to 1 across the field
    norm_win    = normalise_probs(winner_probs)
    norm_podium = normalise_probs(podium_probs)

    for _ in range(n_sims):
        remaining_drivers  = list(range(n_drivers))
        remaining_win      = norm_win.copy()
        remaining_podium   = norm_podium.copy()
        finishing_order    = []

        #Position 1: sample from win probabilities
        win_probs_norm = normalise_probs(remaining_win[remaining_drivers])
        p1_idx         = np.random.choice(remaining_drivers, p=win_probs_norm)
        finishing_order.append(p1_idx)
        win_counts[drivers[p1_idx]] += 1
        podium_counts[drivers[p1_idx]] += 1
        pos_counts[drivers[p1_idx]][1] += 1
        remaining_drivers.remove(p1_idx)

        #Position 2: sample from podium probabilities (winner excluded)
        p2_probs_norm = normalise_probs(remaining_podium[remaining_drivers])
        p2_idx        = np.random.choice(remaining_drivers, p=p2_probs_norm)
        finishing_order.append(p2_idx)
        podium_counts[drivers[p2_idx]] += 1
        pos_counts[drivers[p2_idx]][2] += 1
        remaining_drivers.remove(p2_idx)

        #Position 3: same pool minus P1 and P2
        p3_probs_norm = normalise_probs(remaining_podium[remaining_drivers])
        p3_idx        = np.random.choice(remaining_drivers, p=p3_probs_norm)
        finishing_order.append(p3_idx)
        podium_counts[drivers[p3_idx]] += 1
        pos_counts[drivers[p3_idx]][3] += 1
        remaining_drivers.remove(p3_idx)

        #Positions 4–N: uniform draw from remaining drivers
        np.random.shuffle(remaining_drivers)
        for pos, d_idx in enumerate(remaining_drivers, start=4):
            finishing_order.append(d_idx)
            pos_counts[drivers[d_idx]][pos] += 1

    # Convert counts to percentages
    win_pct    = {d: round(c / n_sims * 100, 1) for d, c in win_counts.items()}
    podium_pct = {d: round(c / n_sims * 100, 1) for d, c in podium_counts.items()}
    avg_pos    = {
        d: round(sum(p * c for p, c in positions.items()) / n_sims, 2)
        for d, positions in pos_counts.items()
    }

    return win_pct, podium_pct, avg_pos



def print_predictions(event_name, drivers, win_pct, podium_pct, avg_pos, grid_df, n_sims):

    #Summary table
    results = pd.DataFrame({
        'Driver':    drivers,
        'Win%':      [win_pct.get(d, 0.0)    for d in drivers],
        'Podium%':   [podium_pct.get(d, 0.0) for d in drivers],
        'Avg Finish':[avg_pos.get(d, 20.0)   for d in drivers],
    })

    if grid_df is not None:
        results = results.merge(grid_df[['Driver', 'GridPosition']], on='Driver', how='left')
        results['GridPosition'] = results['GridPosition'].fillna('?')
    else:
        results['GridPosition'] = '?'

    results = results.sort_values('Win%', ascending=False).reset_index(drop=True)
    results.index += 1

    print(f"\n{'='*62}")
    print(f"  🏁  {event_name} — Race Prediction ({n_sims:,} simulations)")
    print(f"{'='*62}")
    print(f"\n  {'#':<4} {'Driver':<8} {'Grid':<6} {'Win %':<10} {'Podium %':<12} {'Avg Finish'}")
    print(f"  {'-'*55}")

    for rank, row in results.iterrows():
        grid = int(row['GridPosition']) if str(row['GridPosition']).isdigit() else '?'
        print(
            f"  {rank:<4} {row['Driver']:<8} {str(grid):<6} "
            f"{row['Win%']:<10} {row['Podium%']:<12} {row['Avg Finish']}"
        )

    top3 = results.head(3)
    print(f"\n  Predicted Winner:  {results.iloc[0]['Driver']}  ({results.iloc[0]['Win%']}%)")
    print(f" Predicted Podium:  {', '.join(top3['Driver'].tolist())}")
    print(f"  Predicted Top 10:  {', '.join(results.head(10)['Driver'].tolist())}")
    print(f"\n    Probabilities are race-dependent — one driver's win")
    print(f"     reduces all others' chances in each simulation.")
    print(f"{'='*62}\n")

    return results


#Main

def main():
    print(f"\n{'='*62}")
    print(f"  F1 Predictor — Season {SEASON} Round {ROUND}")
    print(f"{'='*62}\n")

    print("Fetching session data...")
    times_df = fetch_session_times(SEASON, ROUND)

    print("\n Fetching grid positions...")
    grid_df = fetch_grid_position(SEASON, ROUND)

    print("\n  Resolving track...")
    event_name, track_id = get_track_id(SEASON, ROUND)

    print("\n Building features...")
    feature_df = build_features(times_df, grid_df, track_id)

    print("\n Getting model probabilities...")
    winner_df = get_raw_probs(feature_df, MODEL_DIR / "rf_winner.pkl", "winner")
    podium_df = get_raw_probs(feature_df, MODEL_DIR / "rf_podium.pkl", "podium")

    if winner_df is None or podium_df is None:
        return

    merged  = winner_df.merge(podium_df[['Driver', 'podium_raw_prob']], on='Driver')
    drivers = merged['Driver'].tolist()
    win_p   = merged['winner_raw_prob'].values
    pod_p   = merged['podium_raw_prob'].values

    print(f"\n Running {N_SIMULATIONS:,} Monte Carlo simulations...")
    np.random.seed(42)
    win_pct, podium_pct, avg_pos = simulate_race(win_p, pod_p, drivers, N_SIMULATIONS)

    results = print_predictions(event_name, drivers, win_pct, podium_pct, avg_pos, grid_df, N_SIMULATIONS)

    # Save
    out_path = Path(f"predictions/{SEASON}_round{ROUND:02d}.csv")
    out_path.parent.mkdir(exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
