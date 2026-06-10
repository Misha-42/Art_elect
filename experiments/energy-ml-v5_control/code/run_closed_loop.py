"""
Замкнутый контур управления Ишимбайскими РЭС.
Раздел 2.5 НИР-2: CatBoost + Экспертная система + автоматическое исполнение.

Запуск:
    python code/run_closed_loop.py
"""

import sys; sys.stdout.reconfigure(encoding='utf-8')
import importlib, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd, numpy as np

# Импорт экспертной системы
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'energy-ml-v4_expert' / 'code'))
import expert_system as es
importlib.reload(es)

from catboost import CatBoostClassifier
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef)
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
RANDOM_STATE = 42

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT.parent / 'energy-ml-v1_baseline' / 'datasets' / 'telemetry_v3_synthetic.csv'
KB_PATH = ROOT.parent / 'energy-ml-v4_expert' / 'knowledge_base' / 'rules.json'
OUT_DIR = ROOT / 'results'
PLOTS_DIR = OUT_DIR / 'plots'


def engineer_features(df):
    """Инженерия признаков."""
    df = df.copy()
    for col in ['U_10kV', 'I_feeder_1', 'P_feeder_1', 'f_grid']:
        for h in [1, 2, 3, 24, 48]:
            df[f'{col}_lag_{h}ch'] = df[col].shift(h*10)
    for col in ['I_feeder_1', 'P_feeder_1']:
        for w in [10, 40, 80, 240]:
            df[f'{col}_sred_{w//10}ch'] = df[col].rolling(w).mean()
            df[f'{col}_std_{w//10}ch'] = df[col].rolling(w).std()
    df['P_Q_ratio'] = df['P_feeder_1'] / (df['Q_feeder_1']+0.01)
    df['U_dev'] = (df['U_10kV']-10.0)/10.0
    ts = pd.to_datetime(df['timestamp'])
    df['hour'] = ts.dt.hour
    df['is_weekend'] = (ts.dt.dayofweek>=5).astype(int)
    return df.bfill().ffill().fillna(0)


def decode_mode(mode):
    return {0:'Норма',1:'Предупредительный',2:'Аварийный',3:'Пост-аварийный'}.get(mode, f'Unknown({mode})')


def main():
    print('=' * 70)
    print('  ЗАМКНУТЫЙ КОНТУР УПРАВЛЕНИЯ ИШИМБАЙСКИМИ РЭС')
    print('  CatBoost + Экспертная система + автоматическое исполнение')
    print('=' * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # === 1. Данные ===
    print('\n[1] Загрузка данных...')
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df_feat = engineer_features(df)

    feature_cols = [c for c in df_feat.columns if c not in ['timestamp', 'mode', 'alarm_flag']]
    X = df_feat[feature_cols].values
    y = df_feat['mode'].values

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    print(f'    Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}')
    dist = pd.Series(y_test).value_counts().sort_index()
    for k, v in dist.items():
        print(f'    {decode_mode(k)}: {v} ({v/len(y_test)*100:.1f}%)')

    # === 2. CatBoost ===
    print('\n[2] Обучение CatBoost...')
    cb = CatBoostClassifier(
        iterations=400, depth=6, learning_rate=0.1,
        random_seed=RANDOM_STATE, verbose=0,
        auto_class_weights='Balanced', early_stopping_rounds=50
    )
    cb.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
    y_pred_cb = cb.predict(X_test).ravel()
    y_prob_cb = cb.predict_proba(X_test)
    ba_cb = balanced_accuracy_score(y_test, y_pred_cb)
    mcc_cb = matthews_corrcoef(y_test, y_pred_cb)
    print(f'    CatBoost BA: {ba_cb:.4f}, MCC: {mcc_cb:.4f}')

    # === 3. Экспертная система ===
    print('\n[3] Инициализация экспертной системы...')
    kb = es.KnowledgeBase(str(KB_PATH))
    engine = es.InferenceEngine(kb)
    hybrid = es.CatBoostExpertHybrid(engine)

    # === 4. Замкнутый контур ===
    print('\n[4] Запуск замкнутого контура управления...')
    test_df = df_feat.iloc[split_idx:].reset_index(drop=True)
    closed_loop_modes = []
    control_actions = []
    audit_log = []

    n_test = len(test_df)
    for i in range(n_test):
        row = test_df.iloc[i]
        h_start = max(0, split_idx + i - 50)
        df_hist = df_feat.iloc[h_start:split_idx + i] if i > 0 else df_feat.iloc[:1]
        facts = es.TelemetryFacts(row, df_hist)
        ml_pred = int(y_pred_cb[i])

        # Гибридное решение
        decision = hybrid.decide(facts, ml_prediction=ml_pred)
        closed_loop_modes.append(decision['mode'])

        # Формирование управляющего воздействия
        action = {
            'step': i,
            'true_mode': int(y_test[i]),
            'ml_mode': ml_pred,
            'final_mode': decision['mode'],
            'source': decision['decision'],
            'rules_fired': decision['n_fired'],
        }
        control_actions.append(action)
        audit_log.append({
            'step': i, 'true_mode': int(y_test[i]),
            'ml_mode': ml_pred, 'final_mode': decision['mode'],
            'source': decision['decision']
        })

        if i % 2000 == 0:
            print(f'    Обработано: {i}/{n_test}')

    print(f'    Обработано: {n_test}/{n_test} - Готово!')

    # === 5. Оценка ===
    print('\n[5] Оценка замкнутого контура...')
    y_final = np.array(closed_loop_modes)

    ba_hybrid = balanced_accuracy_score(y_test, y_final)
    mcc_hybrid = matthews_corrcoef(y_test, y_final)
    f1_hybrid = f1_score(y_test, y_final, average='weighted')

    print(f'\n    {"Метрика":25s} {"CatBoost":15s} {"Гибрид (ЭС+ML)":15s}')
    print(f'    {"-"*55}')
    print(f'    {"Balanced Accuracy":25s} {ba_cb:.4f}{"":10s} {ba_hybrid:.4f}')
    print(f'    {"MCC":25s} {mcc_cb:.4f}{"":10s} {mcc_hybrid:.4f}')
    print(f'    {"F1 (weighted)":25s} {"---":>9s}{"":10s} {f1_hybrid:.4f}')
    print()

    # Отчёт по классам
    print(classification_report(y_test, y_final,
          target_names=['Норма','Предупр.','Авария','Пост-авар.'],
          zero_division=0))

    # Сводка управляющих воздействий
    print(f'\n  Сводка управляющих воздействий:')
    df_actions = pd.DataFrame(control_actions)
    print(f'  Всего решений: {len(df_actions)}')
    for mode in [0, 1, 2, 3]:
        count = (df_actions['final_mode'] == mode).sum()
        print(f'    {decode_mode(mode)}: {count} ({count/len(df_actions)*100:.1f}%)')

    # === 6. Сохранение ===
    pd.DataFrame(audit_log).to_csv(OUT_DIR / 'audit_log.csv', index=False)
    df_actions.to_csv(OUT_DIR / 'control_actions.csv', index=False)

    # Таблица метрик
    metrics_data = [
        ['CatBoost', f'{ba_cb:.4f}', f'{mcc_cb:.4f}', '---'],
        ['Гибрид CatBoost+ЭС', f'{ba_hybrid:.4f}', f'{mcc_hybrid:.4f}', f'{f1_hybrid:.4f}'],
    ]
    pd.DataFrame(metrics_data, columns=['Модель','BA','MCC','F1']).to_csv(
        OUT_DIR / 'metrics_table.csv', index=False)

    # === 7. Графики ===
    print('\n[6] Генерация графиков...')

    # 7.1 Матрица ошибок гибрида
    cm = confusion_matrix(y_test, y_final)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=['Норма','Предупр.','Авария','Пост-авар.'],
                yticklabels=['Норма','Предупр.','Авария','Пост-авар.'], ax=ax)
    ax.set_xlabel('Предсказанный класс', fontsize=12)
    ax.set_ylabel('Истинный класс', fontsize=12)
    ax.set_title('Замкнутый контур — матрица ошибок', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'confusion_matrix_closed_loop.png', dpi=150)
    plt.close()

    # 7.2 Сравнение решений (первые 300 шагов)
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(min(300, n_test))
    ax.plot(x, y_test[:300], 'k-', label='Истинный режим', linewidth=1.5)
    ax.plot(x, y_pred_cb[:300], 'b-', alpha=0.6, label='CatBoost', linewidth=1)
    ax.plot(x, y_final[:300], 'r--', alpha=0.7, label='Замкнутый контур', linewidth=1)
    ax.set_xlabel('Шаг', fontsize=12)
    ax.set_ylabel('Режим', fontsize=12)
    ax.set_title('Сравнение: истина vs CatBoost vs замкнутый контур', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Норма','Предупр.','Авария','Пост-авар.'])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'closed_loop_comparison.png', dpi=150)
    plt.close()

    # 7.3 Источники решений
    sources = pd.DataFrame(audit_log)['source'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'CatBoost': '#3498db', 'ЭС (critical)': '#e74c3c',
              'ЭС (high)': '#f39c12', 'ЭС (medium)': '#27ae60',
              'ЭС (low)': '#95a5a6', 'норма (fallback)': '#2ecc71'}
    bar_colors = [colors.get(s, '#7f8c8d') for s in sources.index]
    ax.barh(sources.index, sources.values, color=bar_colors)
    for i, v in enumerate(sources.values):
        ax.text(v+10, i, str(v), va='center', fontsize=10)
    ax.set_xlabel('Количество решений', fontsize=12)
    ax.set_title('Источники управляющих решений', fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'decision_sources.png', dpi=150)
    plt.close()

    # === 8. Итог ===
    print(f'\n{"=" * 70}')
    print('  ИТОГ ЗАМКНУТОГО КОНТУРА')
    print(f'{"=" * 70}')
    print(f'  CatBoost BA:           {ba_cb:.4f}')
    print(f'  Гибрид (ЭС+ML) BA:     {ba_hybrid:.4f}')
    print(f'  MCC:                   {mcc_hybrid:.4f}')
    print(f'  F1 (weighted):         {f1_hybrid:.4f}')
    print(f'  Управляющих решений:   {len(audit_log)}')
    print(f'  Источники решений:')
    for src, cnt in sources.items():
        print(f'    {src:25s}: {cnt} ({cnt/len(audit_log)*100:.1f}%)')
    print(f'\n  ✅ Результаты сохранены в {OUT_DIR}/')


if __name__ == '__main__':
    main()
