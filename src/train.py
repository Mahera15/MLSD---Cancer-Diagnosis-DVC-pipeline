
import sys
import json
import pandas as pd
import yaml
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def train(model_name):
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train_df = pd.read_csv("data/processed/train.csv")
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    if model_name == "random_forest":
        p = params["train_random_forest"]
        model = RandomForestClassifier(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            random_state=p["random_state"],
        )
        model.fit(X_train, y_train)

    elif model_name == "logistic_regression":
        p = params["train_logistic_regression"]

        # Scale features so every column is on a similar numeric range
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        joblib.dump(scaler, "models/scaler_logistic_regression.pkl")

        model = LogisticRegression(
            max_iter=p["max_iter"],
            random_state=p["random_state"],
        )
        model.fit(X_train_scaled, y_train)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    out_path = f"models/model_{model_name}.pkl"
    joblib.dump(model, out_path)
    print(f"Model ({model_name}) trained and saved to {out_path}")


def evaluate(model_name):
    model = joblib.load(f"models/model_{model_name}.pkl")
    test_df = pd.read_csv("data/processed/test.csv")

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    if model_name == "logistic_regression":
        # Use the SAME scaler that was fit on the training data
        scaler = joblib.load("models/scaler_logistic_regression.pkl")
        X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    metrics_path = f"metrics/metrics_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["malignant", "benign"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix ({model_name})")
    plt.savefig(f"metrics/confusion_matrix_{model_name}.png", bbox_inches="tight")

    print(f"Evaluation metrics ({model_name}):", metrics)


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("train", "evaluate") or sys.argv[2] not in ("random_forest", "logistic_regression"):
        print("Usage: python src/train.py [train|evaluate] [random_forest|logistic_regression]")
        sys.exit(1)

    action, model_name = sys.argv[1], sys.argv[2]

    if action == "train":
        train(model_name)
    else:
        evaluate(model_name)