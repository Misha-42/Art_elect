"""
Модуль прогнозирования нагрузки на базе CatBoost.
Раздел 2.3 НИР-2: прогнозирование активной мощности P_feeder_1.

Горизонты: 1ч, 4ч, 24ч, 168ч.
Метрики: MAE, RMSE, R², MAPE.

Запуск:
    python code/run_catboost_forecast.py
"""

import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings; warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
RANDOM_STATE = 42
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / '..' / 'energy-ml-v1_baseline' / 'datasets' / 'telemetry_v3_synthetic.csv'
OUT_DIR = ROOT / 'results'
PLOTS_DIR = OUT_DIR / 'plots'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Инженерия признаков для прогнозирования (раздел 2.2.3 НИР-2)."""
    df = df.copy()
    ts = pd.to_datetime(df['timestamp'])

    # Лаговые признаки (t-1, t-2, t-3, t-4, t-6, t-12, t-24, t-48)
    for col in ['U_10kV', 'I_feeder_1', 'P_feeder_1', 'f_grid']:
        for h in [1, 2, 3, 4, 6, 12, 24, 48]:
            df[f'lag_{col}_{h}ч'] = df[col].shift(h * 10)

    # Скользящие окна (1ч, 4ч, 8ч, 24ч)
    for col in ['I_feeder_1', 'P_feeder_1']:
        for w, lbl in [(10, '1ч'), (40, '4ч'), (80, '8ч'), (240, '24ч')]:
            df[f'mean_{col}_{lbl}'] = df[col].rolling(w).mean()
            df[f'std_{col}_{lbl}'] = df[col].rolling(w).std()
            df[f'max_{col}_{lbl}'] = df[col].rolling(w).max()
            df[f'min_{col}_{lbl}'] = df[col].rolling(w).min()

    # Отношения и производные
    df['P_Q_ratio'] = df['P_feeder_1'] / (df['Q_feeder_1'] + 0.01)
    df['I_U_ratio'] = df['I_feeder_1'] / (df['U_10kV'] + 0.01)
    df['U_dev'] = (df['U_10kV'] - 10.0) / 10.0
    df['f_dev'] = df['f_grid'] - 50.0
    df['P_diff'] = df['P_feeder_1'].diff()
    df['I_diff'] = df['I_feeder_1'].diff()

    # Календарные признаки
    df['час'] = ts.dt.hour
    df['день_нед'] = ts.dt.dayofweek
    df['выходной'] = (ts.dt.dayofweek >= 5).astype(int)
    df['ночь'] = ((ts.dt.hour >= 23) | (ts.dt.hour <= 5)).astype(int)
    df['час_sin'] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df['час_cos'] = np.cos(2 * np.pi * ts.dt.hour / 24)

    # Заполнение NaN
    df = df.bfill().ffill().fillna(0)
    return df


def create_targets(df: pd.DataFrame, horizons: list) -> dict:
    """Создание целевых переменных для разных горизонтов."""
    targets = {}
    for h in horizons:
        targets[f'target_{h}ч'] = df['P_feeder_1'].shift(-h * 10)
    return pd.DataFrame(targets)


def prepare_data(horizon_h: int):
    """Загрузка, инженерия и подготовка данных для заданного горизонта."""
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = engineer_features(df)

    # Целевая переменная
    target_col = f'target_{horizon_h}ч'
    df[target_col] = df['P_feeder_1'].shift(-horizon_h * 10)

    # Удаление NaN (образовались от лагов и сдвига)
    df = df.dropna()

    # Признаки
    exclude = ['timestamp', 'mode', 'alarm_flag']
    exclude += [c for c in df.columns if c.startswith('target_')]
    feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].values
    y = df[target_col].values

    # Time-series split: 70/15/15
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f'\n{"="*60}')
    print(f'ГОРИЗОНТ: {horizon_h} ч ({horizon_h*10} шагов)')
    print(f'{"="*60}')
    print(f'Обучающая:     {len(X_train)} строк')
    print(f'Валидационная: {len(X_val)} строк')
    print(f'Тестовая:      {len(X_test)} строк')
    print(f'Признаков:     {len(feature_cols)}')

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def train_catboost(X_train, y_train, X_val, y_val, horizon_h: int):
    """Обучение CatBoost (гиперпараметры из Таблицы 2 НИР-2)."""
    params = {'depth': 6, 'learning_rate': 0.1, 'l2_leaf_reg': 5}

    print(f'\nГиперпараметры: depth={params["depth"]}, '
          f'lr={params["learning_rate"]}, l2={params["l2_leaf_reg"]}')

    model = CatBoostRegressor(
        iterations=500,
        depth=params['depth'],
        learning_rate=params['learning_rate'],
        l2_leaf_reg=params['l2_leaf_reg'],
        random_seed=RANDOM_STATE,
        verbose=50,
        early_stopping_rounds=50,
        loss_function='RMSE',
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Оценка на валидации
    val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)
    print(f'  R² на валидации: {val_r2:.4f}')

    return model, params


def evaluate(model, X_test, y_test, y_train, horizon_h: int):
    """Оценка точности прогноза (раздел 2.3.3 НИР-2)."""
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 0.01))) * 100

    # Относительная ошибка в % от среднего
    mean_load = np.mean(y_train)
    mae_pct = mae / mean_load * 100

    print(f'\nМетрики на тестовой выборке:')
    print(f'  MAE:  {mae:.2f} кВт  ({mae_pct:.1f}% от ср. нагрузки)')
    print(f'  RMSE: {rmse:.2f} кВт')
    print(f'  R²:   {r2:.4f}')
    print(f'  MAPE: {mape:.2f}%')

    return {'горизонт': f'{horizon_h} ч', 'MAE': f'{mae:.2f}',
            'RMSE': f'{rmse:.2f}', 'R²': f'{r2:.4f}',
            'MAPE': f'{mape:.1f}%', 'MAE_%': f'{mae_pct:.1f}%',
            'mean_load': f'{mean_load:.0f}'}


def plot_forecast(model, X_test, y_test, feature_cols, horizon_h: int):
    """Графики фактического vs прогнозного значений."""
    y_pred = model.predict(X_test)

    # 1. Факт vs Прогноз (первые 500 точек)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_test[:500], label='Фактическая нагрузка', color='#2c3e50', linewidth=0.8)
    ax.plot(y_pred[:500], label='Прогноз CatBoost', color='#e74c3c', linewidth=0.8, alpha=0.8)
    ax.set_ylabel('P, кВт', fontsize=12)
    ax.set_title(f'Прогноз нагрузки на {horizon_h} ч — CatBoost', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'forecast_{horizon_h}ч.png', dpi=150)
    print(f'Сохранён: forecast_{horizon_h}ч.png')
    plt.close()

    # 2. Диаграмма рассеяния (факт vs прогноз)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred, s=3, alpha=0.3, color='#2980b9')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.5, label='Идеал')
    ax.set_xlabel('Фактическая нагрузка, кВт', fontsize=12)
    ax.set_ylabel('Прогнозная нагрузка, кВт', fontsize=12)
    ax.set_title(f'Диаграмма рассеяния ({horizon_h} ч)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'scatter_{horizon_h}ч.png', dpi=150)
    print(f'Сохранён: scatter_{horizon_h}ч.png')
    plt.close()

    # 3. График остатков
    residuals = y_test - y_pred
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(residuals[:500], color='#8e44ad', linewidth=0.6, alpha=0.7)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel('Ошибка, кВт', fontsize=12)
    ax.set_title(f'Остатки прогноза ({horizon_h} ч)', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'residuals_{horizon_h}ч.png', dpi=150)
    print(f'Сохранён: residuals_{horizon_h}ч.png')
    plt.close()

    # 4. Топ-15 важности признаков
    if hasattr(model, 'get_feature_importance'):
        importance = model.get_feature_importance()
        top_n = 15
        indices = np.argsort(importance)[-top_n:]
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = plt.cm.RdYlGn(importance[indices] / importance[indices].max())
        ax.barh(range(len(indices)), importance[indices], color=colors)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_cols[i] for i in indices], fontsize=8)
        ax.set_xlabel('Важность', fontsize=12)
        ax.set_title(f'Топ-{top_n} признаков — прогноз на {horizon_h} ч', fontsize=14)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f'feature_importance_{horizon_h}ч.png', dpi=150)
        print(f'Сохранён: feature_importance_{horizon_h}ч.png')
        plt.close()

    # 5. Распределение ошибок
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Ошибка прогноза, кВт', fontsize=12)
    ax.set_ylabel('Количество', fontsize=12)
    ax.set_title(f'Распределение ошибок ({horizon_h} ч)', fontsize=14)
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'error_dist_{horizon_h}ч.png', dpi=150)
    print(f'Сохранён: error_dist_{horizon_h}ч.png')
    plt.close()


def plot_learning_curve(model, horizon_h: int):
    """Кривая обучения CatBoost."""
    evals = model.get_evals_result()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(evals['learn']['RMSE'], label='Обучающая выборка', color='#2980b9')
    ax.plot(evals['validation']['RMSE'], label='Валидационная выборка', color='#e74c3c')
    ax.set_xlabel('Итерация', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'Кривая обучения CatBoost ({horizon_h} ч)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f'learning_curve_{horizon_h}ч.png', dpi=150)
    print(f'Сохранён: learning_curve_{horizon_h}ч.png')
    plt.close()


def generate_tables(results: list):
    """Генерация таблиц 1-3 из НИР-2."""
    print(f'\n{"="*60}')
    print('ТАБЛИЦА 1 — Сравнение алгоритмов прогнозирования')
    print(f'{"="*60}')
    table1 = pd.DataFrame([
        ['CatBoost (Яндекс)', '0.88–0.95', 'Высокая', 'Да', '✅ Выбран'],
        ['XGBoost', '0.85–0.92', 'Высокая', 'Нет', 'Нет категориальных'],
        ['LightGBM', '0.84–0.91', 'Средняя', 'Нет', 'Переобучение на малых'],
        ['LSTM', '0.82–0.90', 'Низкая', 'Да', 'Требует много данных'],
        ['Linear Regression', '0.60–0.75', 'Высокая', 'Нет', 'Не ловит нелинейности'],
    ], columns=['Алгоритм', 'R² (типичный)', 'Скорость', 'Импортозамещение', 'Вывод'])
    print(table1.to_string(index=False))

    print(f'\n{"="*60}')
    print('ТАБЛИЦА 2 — Гиперпараметры CatBoost по типам зон')
    print(f'{"="*60}')
    table2 = pd.DataFrame([
        ['Центральная (город)', 6, 0.10, 5, 500],
        ['Сельская', 4, 0.08, 10, 400],
        ['Промышленная', 8, 0.15, 3, 600],
        ['Смешанная', 6, 0.10, 5, 500],
    ], columns=['Тип зоны', 'Глубина', 'Темп обучения', 'L2 рег.', 'Итераций'])
    print(table2.to_string(index=False))

    print(f'\n{"="*60}')
    print('ТАБЛИЦА 3 — Ожидаемые метрики точности по типам зон')
    print(f'{"="*60}')
    table3 = pd.DataFrame([
        ['Центральная (город)', '85–110', '110–140', '0.88–0.93', '6–8'],
        ['Сельская', '50–70', '65–90', '0.85–0.90', '8–12'],
        ['Промышленная', '200–280', '250–350', '0.85–0', '5–7'],
    ], columns=['Тип зоны', 'MAE, кВт', 'RMSE, кВт', 'R²', 'MAPE, %'])
    print(table3.to_string(index=False))

    # Сохранение таблиц
    table1.to_csv(OUT_DIR / 'table1_algorithms.csv', index=False)
    table2.to_csv(OUT_DIR / 'table2_hyperparams.csv', index=False)
    table3.to_csv(OUT_DIR / 'table3_expected_metrics.csv', index=False)
    print('\nТаблицы сохранены в results/')


def plot_metrics_summary(results: list):
    """Сводный график метрик по горизонтам."""
    df = pd.DataFrame(results)
    for col in ['MAE', 'RMSE']:
        df[col] = df[col].astype(float)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    w = 0.3

    bars1 = ax1.bar(x - w/2, df['MAE'], w, label='MAE', color='#3498db')
    bars2 = ax1.bar(x + w/2, df['RMSE'], w, label='RMSE', color='#e74c3c')

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([r['горизонт'] for r in results])
    ax1.set_ylabel('Ошибка, кВт', fontsize=12)
    ax1.set_title('Сравнение метрик по горизонтам прогноза', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # R² на второй оси
    ax2 = ax1.twinx()
    ax2.plot(x, df['R²'].astype(float), 'o-', color='#27ae60', linewidth=2, label='R²')
    for i, v in enumerate(df['R²']):
        ax2.text(i, float(v) + 0.02, f'{v}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('R²', fontsize=12)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'metrics_summary.png', dpi=150)
    print(f'Сохранён: metrics_summary.png')
    plt.close()


def main():
    print('='*60)
    print('МОДУЛЬ ПРОГНОЗИРОВАНИЯ НАГРУЗКИ — CatBoost')
    print('Раздел 2.3 НИР-2: ИСУ Ишимбайскими РЭС')
    print('='*60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Горизонты прогнозирования
    horizons = [1, 4, 24, 168]
    results = []

    for h in horizons:
        # Подготовка данных
        X_train, X_val, X_test, y_train, y_val, y_test, features = prepare_data(h)

        # Обучение
        model, best_params = train_catboost(X_train, y_train, X_val, y_val, h)

        # Оценка
        metrics = evaluate(model, X_test, y_test, y_train, h)
        results.append(metrics)

        # Графики
        plot_forecast(model, X_test, y_test, features, h)
        plot_learning_curve(model, h)

    # Сводные графики
    plot_metrics_summary(results)

    # Таблицы НИР-2
    generate_tables(results)

    # Сводная таблица метрик
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUT_DIR / 'metrics_table.csv', index=False)
    print(f'\n{"="*60}')
    print('СВОДНАЯ ТАБЛИЦА МЕТРИК')
    print(f'{"="*60}')
    print(df_results.to_string(index=False))
    print(f'\n✅ Все результаты сохранены в {OUT_DIR}/')


if __name__ == '__main__':
    main()
