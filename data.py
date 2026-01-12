import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


def generate_dataset(n_samples=2000, random_state=42):
    rng = np.random.RandomState(random_state)

    data = {}
    data["study_hours_per_day"] = rng.uniform(0, 10, n_samples).round(1)
    data["assignments_per_week"] = rng.randint(0, 11, n_samples)
    data["exam_prep_days"] = rng.randint(0, 31, n_samples)
    data["academic_pressure"] = rng.randint(1, 11, n_samples)      # 1–10
    data["anxiety_level"] = rng.randint(1, 11, n_samples)           # 1–10
    data["motivation_level"] = rng.randint(1, 11, n_samples)        # 1–10
    data["sleep_hours"] = rng.uniform(3, 10, n_samples).round(1)
    data["exercise_days_per_week"] = rng.randint(0, 8, n_samples)   # 0–7
    data["screen_time_hours"] = rng.uniform(0, 12, n_samples).round(1)
    data["family_support"] = rng.randint(1, 11, n_samples)          # 1–10
    data["peer_support"] = rng.randint(1, 11, n_samples)            # 1–10
    data["commute_time_minutes"] = rng.randint(0, 121, n_samples)   # 0–120

    df = pd.DataFrame(data)

    # Synthetic stress formula (higher = more stress)
    raw = (
        4 * df["study_hours_per_day"] +
        5 * df["assignments_per_week"] +
        2 * df["exam_prep_days"] +
        6 * df["academic_pressure"] +
        7 * df["anxiety_level"] -
        4 * df["motivation_level"] -
        3 * df["sleep_hours"] -
        2 * df["exercise_days_per_week"] +
        2 * df["screen_time_hours"] -
        3 * df["family_support"] -
        2 * df["peer_support"] +
        0.4 * df["commute_time_minutes"]
    )

    rng = np.random.RandomState(random_state)
    noise = rng.normal(0, raw.std() * 0.1, len(df))
    raw = raw + noise

    # Scale to 0–100
    stress = (raw - raw.min()) / (raw.max() - raw.min()) * 100
    df["stress_level"] = stress.round(2)

    return df


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("✅ Generating synthetic dataset...")
    df = generate_dataset()
    dataset_path = os.path.join("data", "student_stress_dataset.csv")
    df.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path} with shape {df.shape}")

    FEATURE_COLUMNS = [
        "study_hours_per_day",
        "assignments_per_week",
        "exam_prep_days",
        "academic_pressure",
        "anxiety_level",
        "motivation_level",
        "sleep_hours",
        "exercise_days_per_week",
        "screen_time_hours",
        "family_support",
        "peer_support",
        "commute_time_minutes",
    ]

    X = df[FEATURE_COLUMNS].values
    y = df["stress_level"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=150, random_state=42
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=42
        ),
        "Support Vector Regressor": SVR(kernel="rbf"),
    }

    metrics = {}
    best_model_name = None
    best_r2 = -1e9
    best_model = None

    print("\n✅ Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        accuracy_pct = max(0.0, float(r2)) * 100.0  # convert R² to "accuracy" style

        metrics[name] = {
            "r2": float(r2),
            "mae": float(mae),
            "accuracy": float(accuracy_pct),
        }

        print(
            f"{name} -> R²: {r2:.3f}, MAE: {mae:.3f}, Accuracy (R²%): {accuracy_pct:.2f}%"
        )

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model = model

    # Save best model
    model_path = os.path.join("models", "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n🏆 Best model: {best_model_name} (R²={best_r2:.3f})")
    print(f"Model saved to {model_path}")

    # Save metrics for dashboard
    metrics_path = os.path.join("models", "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

    # Save feature order metadata (for safety)
    meta_path = os.path.join("models", "meta.json")
    with open(meta_path, "w") as f:
        json.dump({"feature_order": FEATURE_COLUMNS}, f, indent=4)
    print(f"Meta info saved to {meta_path}")


if __name__ == "__main__":
    main()
