"""
Бенчмарк моделей для НИР-2 Art_elect.
Только модели, указанные в НИР: CatBoost, LightGBM, MLP (нейросеть).

Запуск:
    python code/run_art_benchmark.py --dataset datasets/telemetry_v3_synthetic.csv
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
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             matthews_corrcoef)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Инженерия признаков для распределительных сетей (как в НИР-2)."""
    df_feat = df.copy()

    # Лаговые признаки (t-1, t-2, t-3, t-24, t-48)
    lag_hours = [1, 2, 3, 24, 48]
    for col in ['U_10kV', 'I_feeder_1', 'P_feeder_1', 'f_grid']:
        for h in lag_hours:
            df_feat[f'{col}_lag_{h}h'] = df_feat[col].shift(h * 10)  # 10 записей/час

    # Скользящие окна (1ч, 4ч, 8ч, 24ч)
    windows = [10, 40, 80, 240]
    for col in ['I_feeder_1', 'P_feeder_1']:
        for w in windows:
            df_feat[f'{col}_rmean{w//10}h'] = df_feat[col].rolling(w).mean()
            df_feat[f'{col}_rstd_{w//10}h'] = df_feat[col].rolling(w).std()
            df_feat[f'{col}_rmax_{w//10}h'] = df_feat[col].rolling(w).max()
            df_feat[f'{col}_rmin_{w//10}h'] = df_feat[col].rolling(w).min()

    # Отношения (как в НИР: cos_fi, P/Q, I/U)
    df_feat['ratio_P_Q'] = df_feat['P_feeder_1'] / (df_feat['Q_feeder_1'] + 0.01)
    df_feat['ratio_I_U'] = df_feat['I_feeder_1'] / (df_feat['U_10kV'] + 0.01)
    df_feat['U_deviation'] = (df_feat['U_10kV'] - 10.0) / 10.0  # отклонение от номинала
    df_feat['f_deviation'] = df_feat['f_grid'] - 50.0

    # Календарные признаки
    ts = pd.to_datetime(df_feat['timestamp'])
    df_feat['hour'] = ts.dt.hour
    df_feat['day_of_week'] = ts.dt.dayofweek
    df_feat['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)
    df_feat['is_night'] = ((ts.dt.hour >= 23) | (ts.dt.hour <= 5)).astype(int)

    # Заполнение NaN (появились от лагов и окон)
    df_feat = df_feat.bfill().ffill().fillna(0)

    return df_feat


def load_data(path: str) -> pd.DataFrame:
    """Загрузка и первичный анализ."""
    df = pd.read_csv(path, parse_dates=['timestamp'])
    print(f"\n{'='*60}")
    print(f"Датасет: {path}")
    print(f"Строк: {df.shape[0]}, Колонок: {df.shape[1]}")
    print(f"Период: {df['timestamp'].min()} — {df['timestamp'].max()}")

    # Инженерия признаков
    df = add_features(df)
    print(f"После инженерии: {df.shape[1]} колонок")

    dist = df['mode'].value_counts()
    total = len(df)
    print(f"\nРаспределение классов:")
    labels = {0: 'Норма', 1: 'Предупредительный', 2: 'Аварийный', 3: 'Пост-аварийный'}
    for k, v in sorted(dist.items()):
        print(f"  {k} ({labels[k]}): {v:5d} ({v/total*100:.2f}%)")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Подготовка признаков."""
    exclude = ['timestamp', 'mode', 'alarm_flag']
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].values
    y = df['mode'].values
    print(f"\nПризнаков: {len(feature_cols)}")
    return X, y, feature_cols


def train_and_evaluate(X_train, X_test, y_train, y_test,
                       model, model_name: str, scaler=None) -> dict:
    """Обучение и оценка."""
    if scaler:
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
    else:
        X_train_s, X_test_s = X_train, X_test

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    try:
        y_prob = model.predict_proba(X_test_s)
        roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    except Exception:
        y_prob = None
        roc_auc = 0.0

    ba = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

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
          target_names=['Норма', 'Предупр.', 'Авария', 'Пост-авар.'], zero_division=0))
    print(f"\nМатрица ошибок:\n{cm}")

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
        'model_obj': model if not scaler else None,
    }


def plot_confusion_matrix(cm, model_name: str, save_path: str):
    """Матрица ошибок."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu',
                xticklabels=['Норма', 'Предупр.', 'Авария', 'Пост-авар.'],
                yticklabels=['Норма', 'Предупр.', 'Авария', 'Пост-авар.'], ax=ax)
    ax.set_xlabel('Предсказанный класс')
    ax.set_ylabel('Истинный класс')
    ax.set_title(f'Матрица ошибок — {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")
    plt.close()


def plot_metrics_comparison(results: list, save_path: str):
    """Сравнение метрик всех моделей."""
    metrics = ['balanced_accuracy', 'f1_weighted', 'mcc',
               'precision', 'recall', 'roc_auc_ovr']
    metrics_ru = ['BalAcc', 'F1', 'MCC', 'Prec', 'Recall', 'ROC-AUC']

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metrics))
    n_models = len(results)

    for i, res in enumerate(results):
        values = [res[m] for m in metrics]
        width = 0.8 / n_models
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=res['model'])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_ru)
    ax.set_ylabel('Значение метрики')
    ax.set_title('Сравнение моделей (инженерия признаков)')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Сохранено: {save_path}")
    plt.close()


def save_metrics_table(results: list, save_path: str):
    """Таблица метрик в CSV."""
    rows = []
    for res in results:
        rows.append({
            'Модель': res['model'],
            'Balanced Accuracy': f"{res['balanced_accuracy']:.4f}",
            'F1 (weighted)': f"{res['f1_weighted']:.4f}",
            'MCC': f"{res['mcc']:.4f}",
            'Precision': f"{res['precision']:.4f}",
            'Recall': f"{res['recall']:.4f}",
            'ROC-AUC (OvR)': f"{res['roc_auc_ovr']:.4f}",
            'Матрица ошибок': str(res['confusion_matrix'].tolist()),
        })
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"Таблица метрик: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Бенчмарк Art_elect')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)
    plots_dir = out_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Загрузка
    df = load_data(args.dataset)
    X, y, feature_cols = prepare_features(df)

    # Сплит (без shuffle — временные ряды)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f"Сплит: Train {len(X_train)}, Test {len(X_test)}")

    # ---- Модели (только из НИР-2, ничего общего с soda) ----
    from catboost import CatBoostClassifier
    import lightgbm as lgb
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier

    models = [
        (CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                            random_seed=RANDOM_STATE, verbose=0,
                            early_stopping_rounds=50,
                            auto_class_weights='Balanced'),
         'CatBoost', False),
        (lgb.LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.08,
                            random_state=RANDOM_STATE, verbose=-1,
                            class_weight='balanced'),
         'LightGBM', False),
        (GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                    learning_rate=0.1, random_state=RANDOM_STATE),
         'Gradient Boosting', False),
        (MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                       max_iter=500, random_state=RANDOM_STATE, early_stopping=True),
         'MLP (нейросеть)', True),
    ]

    results = []
    for model, name, need_scaler in models:
        res = train_and_evaluate(X_train, X_test, y_train, y_test,
                                  model, name, scaler=StandardScaler() if need_scaler else None)
        results.append(res)
        plot_confusion_matrix(res['confusion_matrix'], name,
                              plots_dir / f'cm_{name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png')

    # Сравнение
    plot_metrics_comparison(results, out_dir / 'metrics_comparison.png')
    save_metrics_table(results, out_dir / 'metrics_table.csv')

    # Финальная сводка
    print(f"\n{'='*60}")
    print("ФИНАЛЬНАЯ СВОДКА")
    print(f"{'='*60}")
    for res in results:
        print(f"  {res['model']:25s} BA={res['balanced_accuracy']:.4f}  "
              f"MCC={res['mcc']:.4f}  ROC-AUC={res['roc_auc_ovr']:.4f}")
    print(f"\n✅ Результаты: {out_dir}")
    print(f"Графики: {plots_dir}")


if __name__ == '__main__':
    main()
