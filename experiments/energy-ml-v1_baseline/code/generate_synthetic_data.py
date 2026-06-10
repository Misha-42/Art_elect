"""
Генерация синтетических телеметрических данных для Ишимбайских РЭС.

Имитация показаний SCADA: напряжение, ток, мощность, частота.
Режимы: 0=норма, 1=предупреждение, 2=авария.

Использование:
    python code/generate_synthetic_data.py --rows 10000 --output datasets/telemetry_v1_synthetic.csv
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

RANDOM_STATE = 42


def generate_telemetry(n_rows: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Генерация синтетической телеметрии."""
    rng = np.random.default_rng(seed)

    # Временной ряд (шаг 6 минут)
    start = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start + timedelta(minutes=6 * i) for i in range(n_rows)]

    # Базовые параметры (нормальный режим)
    U_base = 10.0  # кВ
    I_base = 50.0  # А
    P_base = 800.0  # кВт
    Q_base = 200.0  # кВАр
    f_base = 50.0  # Гц

    # Генерация с автокорреляцией
    noise_U = rng.normal(0, 0.05, n_rows)
    noise_I = rng.normal(0, 2.0, n_rows)
    noise_P = rng.normal(0, 30.0, n_rows)
    noise_Q = rng.normal(0, 10.0, n_rows)
    noise_f = rng.normal(0, 0.01, n_rows)

    # Суточная и недельная сезонность
    hour = np.array([ts.hour + ts.minute / 60 for ts in timestamps])
    day = np.array([ts.weekday() for ts in timestamps])

    # Дневной паттерн нагрузки (пик в 10-12 и 18-20)
    load_pattern = 0.3 * np.sin(2 * np.pi * (hour - 6) / 24) + 1.0
    load_pattern = np.clip(load_pattern, 0.6, 1.4)
    weekend_factor = np.where(day >= 5, 0.8, 1.0)

    U = U_base + noise_U
    I = I_base * load_pattern * weekend_factor + noise_I
    P = P_base * load_pattern * weekend_factor + noise_P
    Q = Q_base * load_pattern * weekend_factor + noise_Q
    f = f_base + noise_f
    cosφ = rng.uniform(0.92, 0.98, n_rows)

    # Разметка режимов (норма — 90%, предупреждение — 7%, авария — 3%)
    mode = np.zeros(n_rows, dtype=int)
    n_warn = int(n_rows * 0.07)
    n_alarm = int(n_rows * 0.03)

    # Аномалии: резкие скачки тока или падение напряжения
    warn_indices = rng.choice(n_rows, n_warn, replace=False)
    alarm_indices = rng.choice(n_rows, n_alarm, replace=False)

    mode[warn_indices] = 1  # предупреждение
    mode[alarm_indices] = 2  # авария

    # Модифицируем телеметрию для аномальных режимов
    for idx in warn_indices:
        I[idx] *= rng.uniform(1.2, 1.5)
        U[idx] *= rng.uniform(0.85, 0.95)
    for idx in alarm_indices:
        I[idx] *= rng.uniform(1.5, 2.5)
        U[idx] *= rng.uniform(0.6, 0.8)
        f[idx] += rng.uniform(-0.5, -0.1)

    # Флаг аварийного события
    alarm_flag = np.zeros(n_rows, dtype=int)
    alarm_flag[alarm_indices] = 1

    df = pd.DataFrame({
        'timestamp': timestamps,
        'U_10kV': U,
        'I_feeder_1': I,
        'P_feeder_1': P,
        'Q_feeder_1': Q,
        'f_grid': f,
        'cosφ': cosφ,
        'alarm_flag': alarm_flag,
        'mode': mode,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description='Генерация синтетической телеметрии')
    parser.add_argument('--rows', type=int, default=10000,
                        help='Количество строк')
    parser.add_argument('--output', type=str,
                        default='datasets/telemetry_v1_synthetic.csv',
                        help='Путь для сохранения CSV')
    args = parser.parse_args()

    print(f"Генерация {args.rows} строк синтетических данных...")
    df = generate_telemetry(args.rows)

    df.to_csv(args.output, index=False)
    print(f"Сохранено: {args.output}")
    print(f"Размер: {df.shape}")
    print(f"Режимы:\n{df['mode'].value_counts()}")
    print(f"Диапазон дат: {df['timestamp'].min()} — {df['timestamp'].max()}")


if __name__ == '__main__':
    main()
