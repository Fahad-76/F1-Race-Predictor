import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

DATA_PATH = "data/processed/f1_dataset.csv"
MODEL_DIR  = Path("models")

FEATURES = [
    'fp1_delta',
    'fp2_delta',
    'fp3_delta',
    'quali_delta',
    'GridPosition',
    'grid_penalty',
    'TrackID',
]

TARGETS = {
    'podium': 'Podium',
    'winner': 'Winner',
}


def temporal_split(df, cutoff_season):
    """Split by season — no future data leaks into training."""
    train = df[df['Season'] < cutoff_season]
    test  = df[df['Season'] >= cutoff_season]
    return train, test


def train_model(df, target_col, model_path):

    print(f"\n{'='*50}")
    print(f"Training: {target_col}")
    print(f"{'='*50}")

    cutoff   = df['Season'].max()
    train_df, test_df = temporal_split(df, cutoff)

    print(f"Train seasons: {sorted(train_df['Season'].unique().tolist())}")
    print(f"Test season:   {sorted(test_df['Season'].unique().tolist())}")
    print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

    # Only use features that exist in the dataset
    available = [f for f in FEATURES if f in df.columns]
    missing   = set(FEATURES) - set(available)
    if missing:
        print(f"⚠ Features not in dataset, skipping: {missing}")

    X_train = train_df[available]
    y_train = train_df[target_col]
    X_test  = test_df[available]
    y_test  = test_df[target_col]

    # Sanity check class balance
    print(f"Positive class rate — train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    #Evaluation metrics
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]
    auc   = roc_auc_score(y_test, probs)

    print(f"\nROC-AUC: {auc:.3f}  (0.5 = random, 1.0 = perfect)")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=['No', 'Yes']))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, preds))

    #Importance of the different features use
    importances = pd.Series(
        pipeline.named_steps['model'].feature_importances_,
        index=available
    ).sort_values(ascending=False)
    print(f"\nFeature Importances:")
    print(importances.to_string())

    #to ensure quali delta is helping not just grid position bias
    quali_imp = importances.get('quali_delta', 0)
    grid_imp  = importances.get('GridPosition', 0)
    if grid_imp > quali_imp:
        print(f"\n GridPosition ({grid_imp:.3f}) still outweighs quali_delta ({quali_imp:.3f})")
        print("  This may mean delta normalisation isn't adding signal yet.")
        print("  Consider adding more seasons of data.")
    else:
        print(f"\n quali_delta ({quali_imp:.3f}) leading over GridPosition ({grid_imp:.3f})")
        print("  Relative pace is contributing meaningfully.")

    #Save
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\n Model saved → {model_path}")


def train():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(['Season', 'Round']).reset_index(drop=True)

    delta_cols = ['fp1_delta', 'fp2_delta', 'fp3_delta', 'quali_delta']
    missing_deltas = [c for c in delta_cols if c not in df.columns]
    if missing_deltas:
        print(f"Delta columns not found: {missing_deltas}")
        print("  Did you run build_dataset.py after adding add_relative_pace()?")
        print("  Rebuild your dataset before retraining.\n")

    for name, target_col in TARGETS.items():
        if target_col not in df.columns:
            print(f"Skipping {name} — '{target_col}' column not in dataset")
            continue
        train_model(df, target_col, MODEL_DIR / f"rf_{name}.pkl")

    print("\n All models trained.")


if __name__ == "__main__":
    train()