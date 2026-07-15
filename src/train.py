from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TARGET = "stroke"
ID_COLUMN = "id"


def load_processed_dataset(processed_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    paths = {
        "X_train": processed_dir / "X_train.csv",
        "y_train": processed_dir / "y_train.csv",
        "X_test": processed_dir / "X_test.csv",
        "y_test": processed_dir / "y_test.csv",
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Dados pre-processados ausentes. Execute o pre-processamento antes do treino: "
            + ", ".join(missing_files)
        )

    X_train = pd.read_csv(paths["X_train"])
    y_train = pd.read_csv(paths["y_train"]).squeeze("columns").astype(int)
    X_test = pd.read_csv(paths["X_test"])
    y_test = pd.read_csv(paths["y_test"]).squeeze("columns").astype(int)

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("As features e os rótulos pre-processados possuem tamanhos diferentes.")

    return X_train, y_train, X_test, y_test


def split_training_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_val, y_train, y_val


def build_model() -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="liblinear",
    )


def predict_scores(model: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def choose_threshold(y_true: pd.Series, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    return float(thresholds[int(np.argmax(f1_values))])


def evaluate_predictions(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, object]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def evaluate_model(
    model: LogisticRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[float, dict[str, object]]:
    model.fit(X_train, y_train)
    y_val_score = predict_scores(model, X_val)
    threshold = choose_threshold(y_val, y_val_score)
    validation_metrics = evaluate_predictions(y_val, y_val_score, threshold)
    return threshold, validation_metrics


def save_roc_curve(y_true: pd.Series, y_score: np.ndarray, output_path: Path) -> None:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_df = pd.DataFrame(
        {
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "threshold": thresholds,
        }
    )
    roc_df.to_csv(output_path, index=False)


def write_markdown_report(
    output_path: Path,
    class_distribution: dict[str, dict[str, int]],
    model_name: str,
    validation_metrics: dict[str, object],
    test_metrics: dict[str, object],
    model_path: Path,
    roc_curve_path: Path,
) -> None:
    lines = [
        "# Relatorio de desempenho - Predicao de AVC",
        "",
        "## Separacao dos dados",
        "",
        "| Particao | Classe 0 | Classe 1 | Total |",
        "| --- | ---: | ---: | ---: |",
    ]

    for split_name, counts in class_distribution.items():
        total = counts.get("0", 0) + counts.get("1", 0)
        lines.append(
            f"| {split_name} | {counts.get('0', 0)} | {counts.get('1', 0)} | {total} |"
        )

    lines.extend(
        [
            "",
            "## Validacao",
            "",
            f"Modelo: `{model_name}`",
            "",
            "| Threshold | Precision | Recall | F1-Score | ROC AUC | Accuracy |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {validation_metrics['threshold']:.4f} | "
                f"{validation_metrics['precision']:.4f} | "
                f"{validation_metrics['recall']:.4f} | "
                f"{validation_metrics['f1_score']:.4f} | "
                f"{validation_metrics['roc_auc']:.4f} | "
                f"{validation_metrics['accuracy']:.4f} |"
            ),
        ]
    )

    cm = test_metrics["confusion_matrix"]
    lines.extend(
        [
            "",
            "## Teste final",
            "",
            "| Threshold | Precision | Recall | F1-Score | ROC AUC | Accuracy |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {test_metrics['threshold']:.4f} | "
                f"{test_metrics['precision']:.4f} | "
                f"{test_metrics['recall']:.4f} | "
                f"{test_metrics['f1_score']:.4f} | "
                f"{test_metrics['roc_auc']:.4f} | "
                f"{test_metrics['accuracy']:.4f} |"
            ),
            "",
            "## Matriz de confusao no teste",
            "",
            "|  | Previsto 0 | Previsto 1 |",
            "| --- | ---: | ---: |",
            f"| Real 0 | {cm['tn']} | {cm['fp']} |",
            f"| Real 1 | {cm['fn']} | {cm['tp']} |",
            "",
            "## Artefatos",
            "",
            f"- Modelo final: `{model_path.as_posix()}`",
            f"- Curva ROC do teste: `{roc_curve_path.as_posix()}`",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def class_counts(y: pd.Series) -> dict[str, int]:
    counts = y.value_counts().sort_index()
    return {str(label): int(counts.get(label, 0)) for label in [0, 1]}


def run_training(
    processed_dir: str = "data/processed",
    models_dir: str = "models",
    reports_dir: str = "reports",
) -> dict[str, object]:
    processed_path = Path(processed_dir)
    models_path = Path(models_dir)
    reports_path = Path(reports_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    X_train_processed, y_train_processed, X_test, y_test = load_processed_dataset(processed_path)
    X_train, X_val, y_train, y_val = split_training_data(
        X_train_processed, y_train_processed
    )

    model_name = "logistic_regression_balanced"
    validation_model = build_model()
    threshold, validation_metrics = evaluate_model(
        validation_model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    final_model = build_model()
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    final_model.fit(X_train_val, y_train_val)

    y_test_score = predict_scores(final_model, X_test)
    test_metrics = evaluate_predictions(y_test, y_test_score, threshold)

    model_path = models_path / "stroke_risk_model.joblib"
    roc_curve_path = reports_path / "roc_curve_test.csv"
    summary_path = reports_path / "metrics_summary.csv"
    json_path = reports_path / "metrics.json"
    markdown_path = reports_path / "metrics_report.md"

    artifact = {
        "model": final_model,
        "threshold": threshold,
        "target": TARGET,
        "model_name": model_name,
        "features": list(X_train.columns),
        "data_source": processed_path.as_posix(),
    }
    joblib.dump(artifact, model_path)
    save_roc_curve(y_test, y_test_score, roc_curve_path)

    summary_rows = [
        {"split": "validation", "model": model_name, **validation_metrics},
        {"split": "test", "model": model_name, **test_metrics},
    ]
    pd.json_normalize(summary_rows).to_csv(summary_path, index=False)

    class_distribution = {
        "treino": class_counts(y_train),
        "validacao": class_counts(y_val),
        "teste": class_counts(y_test),
    }

    serializable_results = {
        "class_distribution": class_distribution,
        "model": model_name,
        "validation": validation_metrics,
        "test": test_metrics,
        "artifacts": {
            "model": model_path.as_posix(),
            "roc_curve": roc_curve_path.as_posix(),
            "summary": summary_path.as_posix(),
            "report": markdown_path.as_posix(),
        },
    }
    json_path.write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")

    write_markdown_report(
        markdown_path,
        class_distribution,
        model_name,
        validation_metrics,
        test_metrics,
        model_path,
        roc_curve_path,
    )

    return serializable_results


if __name__ == "__main__":
    results = run_training()
    print(f"Modelo treinado: {results['model']}")
    print(f"F1 teste: {results['test']['f1_score']:.4f}")
    print(f"Recall teste: {results['test']['recall']:.4f}")
    print(f"ROC AUC teste: {results['test']['roc_auc']:.4f}")
