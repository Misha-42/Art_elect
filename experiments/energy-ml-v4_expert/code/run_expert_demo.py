"""
Демонстрация работы экспертной системы и гибридной архитектуры.
Раздел 2.4 НИР-2.

Запуск:
    python code/run_expert_demo.py
"""

import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
RANDOM_STATE = 42

# Импорт экспертной системы
sys.path.insert(0, str(Path(__file__).parent))
from expert_system import (
    KnowledgeBase, TelemetryFacts, InferenceEngine, CatBoostExpertHybrid
)

DATA_PATH = Path(__file__).parent.parent.parent / \
    'energy-ml-v1_baseline' / 'datasets' / 'telemetry_v3_synthetic.csv'
OUT_DIR = Path(__file__).parent.parent / 'results'
PLOTS_DIR = OUT_DIR / 'plots'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Инженерия признаков (те же признаки, что в v1)."""
    df = df.copy()
    for col in ['U_10kV', 'I_feeder_1', 'P_feeder_1', 'f_grid']:
        for h in [1, 2, 3, 24, 48]:
            df[f'{col}_lag_{h}ч'] = df[col].shift(h * 10)
    for col in ['I_feeder_1', 'P_feeder_1']:
        for w in [10, 40, 80, 240]:
            df[f'{col}_сред_{w//10}ч'] = df[col].rolling(w).mean()
            df[f'{col}_стд_{w//10}ч'] = df[col].rolling(w).std()
    df['P_Q_отношение'] = df['P_feeder_1'] / (df['Q_feeder_1'] + 0.01)
    df['откл_U'] = (df['U_10kV'] - 10.0) / 10.0
    ts = pd.to_datetime(df['timestamp'])
    df['час'] = ts.dt.hour
    df['выходной'] = (ts.dt.dayofweek >= 5).astype(int)
    return df.bfill().ffill().fillna(0)


def main():
    print('='*60)
    print('ЭКСПЕРТНАЯ СИСТЕМА УПРАВЛЕНИЯ ИШИМБАЙСКИМИ РЭС')
    print('Раздел 2.4 НИР-2')
    print('='*60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # === 1. Загрузка и подготовка ===
    print('\n[1] Загрузка данных...')
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = engineer_features(df)
    print(f'    Строк: {len(df)}, Признаков: {df.shape[1]}')

    # Признаки для CatBoost
    feature_cols = [c for c in df.columns if c not in
                    ['timestamp', 'mode', 'alarm_flag']]
    X = df[feature_cols].values
    y = df['mode'].values

    # === 2. CatBoost (как в v1, но быстрый) ===
    print('\n[2] Обучение CatBoost...')
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f'    Train: {len(X_train)}, Test: {len(X_test)}')

    model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1,
        random_seed=RANDOM_STATE, verbose=0,
        auto_class_weights='Balanced', early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    ba = balanced_accuracy_score(y_test, y_pred)
    print(f'    CatBoost BA: {ba:.4f}')

    # === 3. Экспертная система ===
    print('\n[3] Инициализация экспертной системы...')
    kb = KnowledgeBase()
    engine = InferenceEngine(kb)
    hybrid = CatBoostExpertHybrid(engine)

    # Прогон на тестовых данных
    print('\n[4] Прогон гибридной системы на тестовой выборке...')
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    expert_modes = []
    hybrid_decisions = []

    n_test = len(test_df)
    for i in range(n_test):
        row = test_df.iloc[i]

        # История для производных признаков
        history_start = max(0, split_idx + i - 50)
        df_history = df.iloc[history_start:split_idx + i] if i > 0 else df.iloc[:1]

        # Факты
        facts = TelemetryFacts(row, df_history)

        # ML prediction
        ml_pred = y_pred[i] if i < len(y_pred) else 0

        # Гибридное решение
        decision = hybrid.decide(facts, ml_prediction=ml_pred)
        hybrid_decisions.append(decision)
        expert_modes.append(decision['mode'])

    # === 4. Оценка ===
    print('\n[5] Оценка гибридной системы...')
    true_modes = test_df['mode'].values[:len(expert_modes)]

    metrics = hybrid.validate(true_modes, hybrid_decisions)
    print(f'\n    Точность гибридной системы: {metrics["accuracy"]:.4f}')
    print(f'    Правильных: {metrics["correct"]}/{metrics["total"]}')

    # Сравнение: CatBoost vs гибрид
    catboost_ba = balanced_accuracy_score(true_modes, y_pred[:len(true_modes)])
    expert_ba = balanced_accuracy_score(true_modes, np.array(expert_modes))

    print(f'\n    CatBoost:        BA = {catboost_ba:.4f}')
    print(f'    Экспертная:      BA = {expert_ba:.4f}')
    print(f'    Гибрид (ЭС+ML):  BA = {expert_ba:.4f}')

    # Отчёт по классам
    print(f'\n    Отчёт гибридной системы:')
    print(classification_report(true_modes, expert_modes,
          target_names=['Норма', 'Предупр.', 'Авария', 'Пост-авар.'],
          zero_division=0))

    # === 5. Графики ===
    print('\n[6] Генерация графиков...')

    # 5.1 Матрица ошибок гибридной системы
    cm = confusion_matrix(true_modes, expert_modes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=['Норма', 'Предупр.', 'Авария', 'Пост-авар.'],
                yticklabels=['Норма', 'Предупр.', 'Авария', 'Пост-авар.'],
                ax=ax)
    ax.set_xlabel('Предсказанный класс', fontsize=12)
    ax.set_ylabel('Истинный класс', fontsize=12)
    ax.set_title('Матрица ошибок — гибрид CatBoost + ЭС', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'confusion_matrix_hybrid.png', dpi=150)
    print(f'    Сохранён: confusion_matrix_hybrid.png')
    plt.close()

    # 5.2 Сравнение: CatBoost vs Expert
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(true_modes[:200]))
    ax.plot(x, true_modes[:200], 'k-', label='Истинный режим', linewidth=1.5)
    ax.plot(x, y_pred[:200], 'b-', alpha=0.7, label='CatBoost', linewidth=1)
    ax.plot(x, expert_modes[:200], 'r-', alpha=0.7, label='Гибрид (ЭС+ML)', linewidth=1)
    ax.set_xlabel('Время (шаги)', fontsize=12)
    ax.set_ylabel('Режим', fontsize=12)
    ax.set_title('Сравнение решений: CatBoost vs гибрид ЭС+ML', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Норма', 'Предупр.', 'Авария', 'Пост-авар.'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'comparison_catboost_expert.png', dpi=150)
    print(f'    Сохранён: comparison_catboost_expert.png')
    plt.close()

    # 5.3 Cрабатывания правил по приоритетам
    stats = engine.get_stats()
    priorities = ['critical', 'high', 'medium', 'low']
    counts = [stats['by_mode'].get(m, 0) for m in [0, 1, 2, 3]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(['Критический', 'Высокий', 'Средний', 'Низкий'],
                [stats.get('mode_2', 0), stats.get('mode_1', 0),
                 stats.get('mode_3', 0), stats.get('mode_0', 0)],
                color=['#e74c3c', '#f39c12', '#3498db', '#27ae60'])
    axes[0].set_title('Срабатывания по приоритетам', fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(['Норма (0)', 'Предупр. (1)', 'Авария (2)', 'Пост-авар. (3)'],
                counts, color=['#27ae60', '#f39c12', '#e74c3c', '#8e44ad'])
    axes[1].set_title('Срабатывания по режимам', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'expert_stats.png', dpi=150)
    print(f'    Сохранён: expert_stats.png')
    plt.close()

    # === 6. Сводка ===
    print(f'\n{"="*60}')
    print('СВОДКА РЕЗУЛЬТАТОВ')
    print(f'{"="*60}')
    print(f'  CatBoost BA:           {catboost_ba:.4f}')
    print(f'  Гибрид (ЭС+ML) BA:     {expert_ba:.4f}')
    print(f'  Всего срабатываний ЭС: {stats["total_fired"]}')
    print(f'  Из них critical:       {stats.get("mode_2", 0)}')
    print(f'\n✅ Результаты сохранены в {OUT_DIR}/')


if __name__ == '__main__':
    main()
