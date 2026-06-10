"""
energy-ml-v1_baseline: Базовый классификатор режимов сети.

Запуск:
    python code/run_baseline.py --dataset datasets/telemetry_v1.csv

Зависимости:
    pip install pandas numpy sklearn matplotlib seaborn
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Загрузка датасета."""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    print(f"Загружен датасет: {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Распределение классов:\n{df['mode'].value_counts()}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Подготовка признаков (без утечки из будущего)."""
    feature_cols = [c for c in df.columns if c not in ['timestamp', 'mode', 'alarm_flag']]
    X = df[feature_cols].values
    y = df['mode'].values
    return X, y, feature_cols


def train_baseline(X_train, y_train, X_test, y_test, model_type='rf'):
    """Обучение baseline-модели."""
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_type == 'lr':
        model = LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1
        )
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    ba = balanced_accuracy_score(y_test, y_pred)
    print(f"\n{'='*50}")
    print(f"Модель: {model_type}")
    print(f"Balanced Accuracy: {ba:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    return model, ba


def main():
    parser = argparse.ArgumentParser(description='Baseline классификатор режимов сети')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Путь к CSV-датасету')
    parser.add_argument('--model', type=str, default='rf',
                        choices=['rf', 'lr'],
                        help='Тип модели (rf — RandomForest, lr — LogisticRegression)')
    args = parser.parse_args()

    # Загрузка
    df = load_data(args.dataset)

    # Признаки (без shuffle для временных рядов)
    X, y, feature_cols = prepare_features(df)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Обучение
    model, ba = train_baseline(X_train, y_train, X_test, y_test,
                               model_type=args.model)

    print(f"\n✅ Baseline {args.model}: BA = {ba:.4f}")


if __name__ == '__main__':
    main()
