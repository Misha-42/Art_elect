"""
Генерация синтетических телеметрических данных для Ишимбайских РЭС.
Версия 3 — 4 режима (норма/предупредительный/аварийный/пост-аварийный).

Режимы: 0=норма, 1=предупредительный, 2=аварийный, 3=пост-аварийный.
Цель: реалистичные метрики (BA ~0.80-0.92, не 1.0).

Использование:
    python code/generate_synthetic_data.py --rows 50000 --output datasets/telemetry_v3_synthetic.csv
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

RANDOM_STATE = 42


def generate_telemetry(n_rows: int = 50000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Генерация реалистичной синтетической телеметрии со сложной структурой."""
    rng = np.random.default_rng(seed)

    # Временной ряд (шаг 6 минут = 10 записей/час)
    start = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start + timedelta(minutes=6 * i) for i in range(n_rows)]

    # ---- Суточная и недельная сезонность ----
    hour_of_day = np.array([ts.hour + ts.minute / 60 for ts in timestamps])
    day_of_week = np.array([ts.weekday() for ts in timestamps])

    # Паттерн нагрузки: утренний пик, вечерний пик, ночной провал
    load_profile = 0.80 + 0.30 * np.exp(-((hour_of_day - 9) ** 2) / 12) + \
                   0.35 * np.exp(-((hour_of_day - 19) ** 2) / 14)
    load_profile -= 0.15 * np.exp(-((hour_of_day - 3) ** 2) / 8)
    # Выходные
    weekend_mask = (day_of_week >= 5).astype(float)
    load_profile *= (1.0 - 0.12 * weekend_mask)
    load_profile = np.clip(load_profile, 0.55, 1.45)

    # ---- Базовая телеметрия (шумная) ----
    n = n_rows
    U = 10.0 + rng.normal(0, 0.15, n)  # ~10 кВ ± шум
    I = 50.0 * load_profile + rng.normal(0, 4.0, n)  # ток с шумом
    P = 800.0 * load_profile + rng.normal(0, 25.0, n)  # активная мощность
    Q = 200.0 * load_profile + rng.normal(0, 12.0, n)  # реактивная мощность
    f = 50.0 + rng.normal(0, 0.03, n)  # частота
    cos_fi = np.clip(rng.normal(0.94, 0.03, n), 0.85, 0.99)  # коэфф. мощности

    # ---- 4 режима по НИР-2: норма/предупредительный/аварийный/пост-аварийный ----
    mode = np.zeros(n, dtype=int)
    n_events = 45

    for ev in range(n_events):
        ev_start = rng.integers(300, n - 300)
        ev_end = min(ev_start + rng.integers(15, 50), n - 1)

        # Распределение: 50% предупр., 25% авария, 25% пост-аварийный
        if ev < 22:
            ev_type = 1  # предупредительный
        elif ev < 34:
            ev_type = 2  # аварийный
        else:
            ev_type = 3  # пост-аварийный

        mode[ev_start:ev_end] = ev_type
        prof = np.sin(np.linspace(0, np.pi, ev_end - ev_start))

        if ev_type == 1:  # предупредительный — слабое отклонение
            s = rng.uniform(0.04, 0.12)
            I[ev_start:ev_end] *= (1.0 + s * prof)
            U[ev_start:ev_end] *= (1.0 - 0.3 * s * prof)

        elif ev_type == 2:  # аварийный — сильное отклонение
            s = rng.uniform(0.15, 0.40)
            I[ev_start:ev_end] *= (1.0 + s * prof)
            U[ev_start:ev_end] *= (1.0 - 0.5 * s * prof)
            f[ev_start:ev_end] += s * rng.uniform(-0.3, -0.05) * prof
            cos_fi[ev_start:ev_end] *= (1.0 - 0.12 * prof)

        elif ev_type == 3:  # пост-аварийный — возврат к норме с шумом
            decay = np.exp(-np.linspace(0, 3, ev_end - ev_start))
            I[ev_start:ev_end] *= (1.0 + 0.08 * decay * rng.normal(0, 1, ev_end - ev_start))
            U[ev_start:ev_end] *= (1.0 + 0.03 * decay * rng.normal(0, 1, ev_end - ev_start))

    # ---- Флаг аварии (только mode=2) ----
    alarm_flag = (mode == 2).astype(int)

    # ---- Пропуски (2% случайных NaN) ----
    for col_data in [U, I, P, Q, f, cos_fi]:
        nan_mask = rng.random(n) < 0.002
        col_data[nan_mask] = np.nan

    # ---- Сборка ----
    df = pd.DataFrame({
        'timestamp': timestamps,
        'U_10kV': U,
        'I_feeder_1': I,
        'P_feeder_1': P,
        'Q_feeder_1': Q,
        'f_grid': f,
        'cos_fi': cos_fi,
        'alarm_flag': alarm_flag,
        'mode': mode,
    })

    # Заполнение NaN (среднее по столбцу)
    for col in ['U_10kV', 'I_feeder_1', 'P_feeder_1', 'Q_feeder_1', 'f_grid', 'cos_fi']:
        df[col] = df[col].fillna(df[col].mean())

    return df


def main():
    parser = argparse.ArgumentParser(description='Генератор синтетической телеметрии v3')
    parser.add_argument('--rows', type=int, default=50000)
    parser.add_argument('--output', type=str,
                        default='datasets/telemetry_v3_synthetic.csv')
    args = parser.parse_args()

    print(f"Генерация {args.rows} строк синтетических данных (v3 — реалистичные)...")
    df = generate_telemetry(args.rows)

    df.to_csv(args.output, index=False)
    print(f"\nСохранено: {args.output}")
    print(f"Размер: {df.shape}")
    dist = df['mode'].value_counts()
    total = len(df)
    print(f"\nРаспределение режимов:")
    labels = {0: 'Норма', 1: 'Предупредительный', 2: 'Аварийный', 3: 'Пост-аварийный'}
    for k, v in sorted(dist.items()):
        print(f"  {k} ({labels[k]}): {v:5d} ({v/total*100:.2f}%)")
    print(f"Диапазон дат: {df['timestamp'].min()} — {df['timestamp'].max()}")


if __name__ == '__main__':
    main()
