"""
Полный бенчмарк: сравнение моделей на синтетической телеметрии.
Генерирует таблицы метрик и графики для НИР.

Запуск:
    python code/run_full_benchmark.py --dataset datasets/telemetry_v1_synthetic.csv
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             matthews_corrcoef)
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150


def load_data(path: str) -> pd.DataFrame:
    """Загрузка датасета."""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    print(f"\n{'='*60}")
    print(f"Датасет: {path}")
    print(f"Строк: {df.shape[0]}, Колонок: {df.shape[1]}")
    print(f"Период: {df['timestamp'].min()} — {df['timestamp'].max()}")
    print(f"\nРаспределение классов:")
    class_dist = df['mode'].value_counts().sort_index()
    for k, v in class_dist.items():
        label = {0: 'Норма', 1: 'Предупреждение', 2: 'Авария'}[k]
        print(f"  {k} ({label}): {v} ({v/len(df)*100:.1f}%)")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Подготовка признаков."""
    feature_cols = [c for c in df.columns
                    if c not in ['timestamp', 'mode', 'alarm_flag']]
    X = df[feature_cols].values
    y = df['mode'].values
    print(f"\nПризнаки: {len(feature_cols)}")
    for c in feature_cols:
        print(f"  - {c}")
    return X, y, feature_cols


def train_and_evaluate(X_train, X_test, y_train, y_test,
                       model, model_name: str) -> dict:
    """Обучение и оценка модели."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Метрики
    ba = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    # ROC-AUC (OvR для многоклассовой)
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)

    # Отчёт
    print(f"\n{'='*60}")
    print(f"МОДЕЛЬ: {model_name}")
    print(f"{'='*60}")
    print(f"Balanced Accuracy: {ba:.4f}")
    print(f"F1 (weighted):     {f1:.4f}")
    print(f"MCC:               {mcc:.4f}")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"ROC-AUC (OvR):     {roc_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
          target_names=['Норма', 'Предупреждение', 'Авария']))
    print(f"\nМатрица ошибок:")
    print(cm)

    return {
        'model': model_name,
        'balanced_accuracy': ba,
        'f1_weighted': f1,
        'mcc': mcc,
        'precision': precision,
        'recall': recall,
        'roc_auc_ovr': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'model_obj': model
    }


def plot_confusion_matrix(cm, model_name: str, save_path: str):
    """График матрицы ошибок."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Норма', 'Предупр.', 'Авария'],
                yticklabels=['Норма', 'Предупр.', 'Авария'],
                ax=ax)
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    ax.set_title(f'Матрица ошибок — {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")
    plt.close()


def plot_metrics_comparison(results: list, save_path: str):
    """Сравнение метрик моделей."""
    metrics = ['balanced_accuracy', 'f1_weighted', 'mcc',
               'precision', 'recall', 'roc_auc_ovr']
    metrics_ru = ['Balanced Accuracy', 'F1 (weighted)', 'MCC',
                  'Precision', 'Recall', 'ROC-AUC']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    width = 0.35

    for i, res in enumerate(results):
        values = [res[m] for m in metrics]
        offset = (i - len(results)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=res['model'])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_ru)
    ax.set_ylabel('Значение метрики')
    ax.set_title('Сравнение моделей классификации режимов сети')
    ax.legend()
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")
    plt.close()


def plot_feature_importance(model, feature_cols, model_name: str,
                            save_path: str, top_n: int = 10):
    """График важности признаков."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        print(f"Нет важности признаков для {model_name}")
        return

    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.set_xlabel('Важность')
    ax.set_title(f'Топ-{top_n} признаков — {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")
    plt.close()


def plot_class_distribution(df: pd.DataFrame, save_path: str):
    """Распределение классов."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    labels = ['Норма (0)', 'Предупреждение (1)', 'Авария (2)']
    counts = df['mode'].value_counts().sort_index()
    bars = ax.bar(labels, counts.values, color=colors)
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{v}\n({v/len(df)*100:.1f}%)',
                ha='center', va='bottom')
    ax.set_ylabel('Количество записей')
    ax.set_title('Распределение режимов сети в датасете')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")
    plt.close()


def plot_time_series(df: pd.DataFrame, save_path: str):
    """Временной ряд телеметрии."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    ts = df.head(1000)  # Первые 1000 точек для наглядности

    axes[0].plot(ts['timestamp'], ts['U_10kV'], color='#3498db', linewidth=0.8)
    axes[0].set_ylabel('U, кВ')
    axes[0].set_title('Напряжение 10 кВ')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ts['timestamp'], ts['I_feeder_1'], color='#e74c3c', linewidth=0.8)
    axes[1].set_ylabel('I, А')
    axes[1].set_title('Ток фидера 1')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ts['timestamp'], ts['P_feeder_1'], color='#2ecc71', linewidth=0.8)
    axes[2].set_ylabel('P, кВт')
    axes[2].set_title('Активная мощность')
    axes[2].grid(True, alpha=0.3)

    # Аномалии на графике
    anomalies = ts[ts['mode'] > 0]
    axes[3].plot(ts['timestamp'], ts['f_grid'], color='#9b59b6', linewidth=0.8)
    axes[3].scatter(anomalies['timestamp'], anomalies['f_grid'],
                    c=anomalies['mode'], cmap='autumn_r', s=15, alpha=0.7)
    axes[3].set_ylabel('f, Гц')
    axes[3].set_title('Частота сети (аномалии выделены цветом)')
    axes[3].grid(True, alpha=0.3)

    plt.xlabel('Время')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")
    plt.close()


def save_metrics_table(results: list, save_path: str):
    """Сохранение таблицы метрик в CSV."""
    rows = []
    for res in results:
        rows.append({
            'Модель': res['model'],
            'Balanced Accuracy': f"{res['balanced_accuracy']:.4f}",
            'F1 (weighted)': f"{res['f1_weighted']:.4f}",
            'MCC': f"{res['mcc']:.4f}",
            'Precision': f"{res['precision']:.4f}",
            'Recall': f"{res['recall']:.4f}",
            'ROC-AUC (OvR)': f"{res['roc_auc_ovr']:.4f}"})
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_csv(save_path, index=False)
    print(f"Сохранена таблица метрик: {save_path}")
    return df_metrics


def main():
    parser = argparse.ArgumentParser(description='Полный бенчмарк моделей')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    # Загрузка
    df = load_data(args.dataset)

    # Распределение классов
    plot_class_distribution(df, out_dir / 'class_distribution.png')

    # Временной ряд
    plot_time_series(df, out_dir / 'time_series.png')

    # Признаки
    X, y, feature_cols = prepare_features(df)

    # Сплит без shuffle (временные ряды)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"\nСплит: Train {len(X_train)}, Test {len(X_test)}")

    # Модели
    models = [
        (RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                n_jobs=-1, class_weight='balanced'),
         'Random Forest'),
        (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                            n_jobs=-1, class_weight='balanced'),
         'Logistic Regression'),
        (CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1,
                            random_seed=RANDOM_STATE, verbose=0,
                            early_stopping_rounds=50,
                            auto_class_weights='Balanced'),
         'CatBoost'),
    ]

    results = []
    for model, name in models:
        res = train_and_evaluate(X_train, X_test, y_train, y_test, model, name)
        results.append(res)

        # Матрица ошибок
        plot_confusion_matrix(res['confusion_matrix'], name,
                              out_dir / f'cm_{name.lower().replace(" ", "_")}.png')

        # Важность признаков
        plot_feature_importance(res['model_obj'], feature_cols, name,
                                out_dir / f'feature_importance_{name.lower().replace(" ", "_")}.png')

    # Сравнение метрик
    plot_metrics_comparison(results, out_dir / 'metrics_comparison.png')

    # Таблица метрик
    df_metrics = save_metrics_table(results, out_dir / 'metrics_table.csv')

    # Финальная сводка
    print(f"\n{'='*60}")
    print("ФИНАЛЬНАЯ СВОДКА")
    print(f"{'='*60}")
    print(df_metrics.to_string(index=False))
    print(f"\n✅ Все результаты сохранены в: {out_dir}")


if __name__ == '__main__':
    main()
