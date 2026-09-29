from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def create_synthetic_dataset(n_samples=3000, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    income = rng.normal(60000, 22000, n_samples).clip(18000, 180000)
    debt = rng.normal(18000, 12000, n_samples).clip(500, 90000)
    payment_history = rng.normal(82, 12, n_samples).clip(35, 100)
    credit_utilization = (rng.beta(2.2, 4.0, n_samples) * 100).clip(1, 100)
    credit_age_years = rng.normal(7, 4, n_samples).clip(0.5, 25)
    late_payments = rng.poisson(1.5, n_samples).clip(0, 12)
    open_accounts = rng.poisson(6, n_samples).clip(1, 20)
    inquiries = rng.poisson(1.5, n_samples).clip(0, 8)
    debt_to_income = debt / income
    risk_score = (-0.025*(payment_history-80) + 2.5*debt_to_income +
                  0.035*(credit_utilization-30) + 0.30*late_payments +
                  0.18*inquiries - 0.035*credit_age_years -
                  0.000012*(income-60000))
    probability_risky = 1 / (1 + np.exp(-risk_score))
    credit_risk = (rng.random(n_samples) < probability_risky).astype(int)
    return pd.DataFrame({
        "income": income.round(2), "debt": debt.round(2),
        "payment_history": payment_history.round(2),
        "credit_utilization": credit_utilization.round(2),
        "credit_age_years": credit_age_years.round(2),
        "late_payments": late_payments.astype(int),
        "open_accounts": open_accounts.astype(int),
        "inquiries": inquiries.astype(int),
        "credit_risk": credit_risk})

def engineer_features(df):
    data = df.copy()
    data["debt_to_income"] = data["debt"] / data["income"]
    data["income_after_debt"] = data["income"] - data["debt"]
    data["late_payment_ratio"] = data["late_payments"] / data["open_accounts"].clip(lower=1)
    return data

def evaluate_model(name, model, x_test, y_test, output_dir):
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-Score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities)}
    print("\n" + "="*60 + "\n" + name + "\n" + "="*60)
    print(classification_report(y_test, predictions, target_names=["Good", "Risky"], zero_division=0))
    for key, value in metrics.items():
        if key != "Model":
            print(f"{key}: {value:.4f}")
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, predictions),
                                     display_labels=["Good", "Risky"])
    display.plot(cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / (name.lower().replace(" ", "_") + "_confusion_matrix.png"), dpi=160)
    plt.close()
    return metrics

def main():
    project_dir = Path(__file__).resolve().parent
    data_dir, output_dir = project_dir/"data", project_dir/"outputs"
    data_dir.mkdir(exist_ok=True); output_dir.mkdir(exist_ok=True)
    df = create_synthetic_dataset()
    df.to_csv(data_dir/"credit_scoring_dataset.csv", index=False)
    data = engineer_features(df)
    x, y = data.drop(columns=["credit_risk"]), data["credit_risk"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=RANDOM_STATE, stratify=y)
    logistic = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])
    forest = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=3, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    logistic.fit(x_train, y_train); forest.fit(x_train, y_train)
    results = [evaluate_model("Logistic Regression", logistic, x_test, y_test, output_dir),
               evaluate_model("Random Forest", forest, x_test, y_test, output_dir)]
    pd.DataFrame(results).to_csv(output_dir/"model_metrics.csv", index=False)
    ax = plt.gca()
    RocCurveDisplay.from_estimator(logistic, x_test, y_test, ax=ax, name="Logistic Regression")
    RocCurveDisplay.from_estimator(forest, x_test, y_test, ax=ax, name="Random Forest")
    ax.set_title("Credit Scoring - ROC Curves")
    plt.tight_layout(); plt.savefig(output_dir/"roc_curves.png", dpi=160); plt.close()
    importances = pd.Series(forest.feature_importances_, index=x.columns).sort_values(ascending=False)
    plt.figure(figsize=(9,5)); importances.head(10).sort_values().plot(kind="barh")
    plt.title("Top Feature Importances - Random Forest"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(output_dir/"feature_importance.png", dpi=160); plt.close()
    print("\nPROJECT COMPLETED SUCCESSFULLY")
    print("Metrics:", output_dir/"model_metrics.csv")

if __name__ == "__main__":
    main()
