import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

DATA_PATH  = Path("data/processed/f1_dataset.csv")
MODEL_DIR  = Path("models")
REPORT_DIR = Path("reports")

FEATURES = [
    'fp1_delta',
    'fp2_delta',
    'fp3_delta',
    'quali_delta',
    'GridPosition',
    'grid_penalty',
    'TrackID',
]

MODELS = {
    'podium': ('rf_podium.pkl', 'Podium'),
    'winner': ('rf_winner.pkl', 'Winner'),
}


def temporal_split(df, cutoff_season):
    train = df[df['Season'] < cutoff_season]
    test  = df[df['Season'] >= cutoff_season]
    return train, test


def evaluate_model(df, model_path, target_col, label):

    print(f"\n{'='*50}")
    print(f"Evaluating: {label}")
    print(f"{'='*50}")

    if not model_path.exists():
        print(f"  Model not found at {model_path} — run train.py first")
        return

    pipeline = joblib.load(model_path)

    cutoff = df['Season'].max()
    _, test_df = temporal_split(df, cutoff)

    if test_df.empty:
        print("  Test set is empty — need at least 2 seasons of data")
        return

    delta_cols = ['fp1_delta', 'fp2_delta', 'fp3_delta', 'quali_delta']
    missing_deltas = [c for c in delta_cols if c not in df.columns]
    if missing_deltas:
        print(f"  Delta columns missing: {missing_deltas}")
        print("    Rebuild dataset with build_dataset.py before evaluating.")
        return

    available = [f for f in FEATURES if f in df.columns]
    missing   = set(FEATURES) - set(available)
    if missing:
        print(f"  Some features not found, skipping: {missing}")

    X_test = test_df[available]
    y_test = test_df[target_col]

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    #Key metrics
    auc = roc_auc_score(y_test, probs)
    print(f"\n  ROC-AUC: {auc:.3f}  (0.5 = random, 1.0 = perfect)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=['No', 'Yes']))
    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, preds)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Positives  (correctly called {label}): {tp}")
    print(f"  False Positives (wrongly called {label}):   {fp}")
    print(f"  False Negatives (missed {label}):           {fn}")
    print(f"  True Negatives  (correctly ruled out):      {tn}")

    #Feature importance
    importances = pd.Series(
        pipeline.named_steps['model'].feature_importances_,
        index=available
    ).sort_values(ascending=False)
    print(f"\n  Feature Importances:")
    print(importances.to_string())

    #to ensure quali delta is helping not just grid position bias
    quali_imp = importances.get('quali_delta', 0)
    grid_imp  = importances.get('GridPosition', 0)
    if grid_imp > quali_imp:
        print(f"\n  GridPosition ({grid_imp:.3f}) still outweighs quali_delta ({quali_imp:.3f})")
        print("    Consider adding more seasons of data.")
    else:
        print(f"\n quali_delta ({quali_imp:.3f}) leading GridPosition ({grid_imp:.3f})")

    #Per-race breakdown
    print(f"\n  Per-race breakdown (test season: {cutoff}):")
    test_copy = test_df.copy()
    test_copy['Predicted']   = preds
    test_copy['Probability'] = probs

    for round_num in sorted(test_copy['Round'].unique()):
        race      = test_copy[test_copy['Round'] == round_num]
        event     = race['EventName'].iloc[0] if 'EventName' in race.columns else f"Round {round_num}"
        actual    = race[race[target_col] == 1]['Driver'].tolist()
        predicted = race[race['Predicted'] == 1]['Driver'].tolist()
        top_pick  = race.sort_values('Probability', ascending=False).iloc[0]['Driver']
        print(f"    {event:<30} actual={actual}  predicted={predicted}  top_prob={top_pick}")

    #Plots
    REPORT_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{label} Model — Season {cutoff} Test Set")

    # ROC curve
    RocCurveDisplay.from_predictions(y_test, probs, ax=axes[0], name=label)
    axes[0].set_title("ROC Curve")

    # Feature importance bar chart
    importances.plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title("Feature Importances")
    axes[1].set_ylabel("Importance")
    axes[1].tick_params(axis='x', rotation=30)

    # Probability distribution — positive vs negative class
    axes[2].hist(probs[y_test == 0], bins=30, alpha=0.6, label='No',  color='steelblue')
    axes[2].hist(probs[y_test == 1], bins=30, alpha=0.6, label='Yes', color='tomato')
    axes[2].set_title("Predicted Probability Distribution")
    axes[2].set_xlabel("Predicted Probability")
    axes[2].set_ylabel("Count")
    axes[2].legend()

    plt.tight_layout()
    plot_path = REPORT_DIR / f"eval_{label.lower()}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n Plot saved → {plot_path}")


def evaluate():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(['Season', 'Round']).reset_index(drop=True)

    for label, (model_file, target_col) in MODELS.items():
        if target_col not in df.columns:
            print(f"⚠ Skipping {label} — '{target_col}' column not in dataset")
            continue
        evaluate_model(df, MODEL_DIR / model_file, target_col, label)


if __name__ == "__main__":
    evaluate()