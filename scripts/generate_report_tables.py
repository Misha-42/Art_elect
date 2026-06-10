from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = ROOT / "plots"


@dataclass(frozen=True)
class TableSpec:
    filename: str
    title: str
    caption: str
    columns: list[str]
    rows: list[dict[str, str]]
    column_weights: list[float]
    wrap_widths: list[int]
    header_wrap_widths: list[int]
    alignments: list[str]
    figsize: tuple[float, float]
    font_size: float = 9.2
    title_size: float = 13.5
    caption_size: float = 8.6


def wrap_text(value: object, width: int) -> str:
    text = "" if value is None else str(value)
    if not text:
        return ""

    parts = []
    for chunk in text.split("\n"):
        wrapped = textwrap.fill(
            chunk,
            width=width,
            break_long_words=False,
            break_on_hyphens=False,
            replace_whitespace=False,
        )
        parts.append(wrapped)
    return "\n".join(parts)


def normalize_widths(widths: list[float]) -> list[float]:
    total = sum(widths)
    if total <= 0:
        raise ValueError("Column widths must sum to a positive value.")
    return [width / total for width in widths]


def render_table(spec: TableSpec) -> Path:
    output_path = PLOTS_DIR / spec.filename
    df = pd.DataFrame.from_records(spec.rows, columns=spec.columns)

    wrapped_df = pd.DataFrame(
        {
            column: df[column].map(lambda value, w=width: wrap_text(value, w))
            for column, width in zip(spec.columns, spec.wrap_widths)
        }
    )
    wrapped_headers = [
        wrap_text(column, width)
        for column, width in zip(spec.columns, spec.header_wrap_widths)
    ]

    fig, ax = plt.subplots(figsize=spec.figsize, dpi=240)
    fig.patch.set_facecolor("white")
    ax.set_axis_off()

    fig.text(
        0.5,
        0.965,
        spec.title,
        ha="center",
        va="top",
        fontsize=spec.title_size,
        fontweight="bold",
        color="#17324d",
    )
    fig.text(
        0.5,
        0.045,
        spec.caption,
        ha="center",
        va="bottom",
        fontsize=spec.caption_size,
        color="#4b5d70",
    )

    table = ax.table(
        cellText=wrapped_df.values,
        colLabels=wrapped_headers,
        colWidths=normalize_widths(spec.column_weights),
        cellLoc="left",
        colLoc="center",
        loc="center",
        bbox=[0.015, 0.12, 0.97, 0.78],
    )
    table.auto_set_font_size(False)

    header_color = "#1f4e79"
    band_color = "#f5f9fd"
    alt_band_color = "#ffffff"
    edge_color = "#8aa1b5"

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.85)
        cell.PAD = 0.035
        cell.get_text().set_fontfamily("DejaVu Sans")
        cell.get_text().set_verticalalignment("center")
        cell.get_text().set_wrap(True)

        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(spec.font_size + 0.4)
            cell.get_text().set_ha("center")
        else:
            cell.set_facecolor(band_color if row % 2 == 1 else alt_band_color)
            cell.get_text().set_fontsize(spec.font_size)
            cell.get_text().set_ha(spec.alignments[col])
            if col == 0:
                cell.get_text().set_weight("bold")

    plt.savefig(output_path, dpi=240, bbox_inches="tight", pad_inches=0.2, facecolor="white")
    plt.close(fig)
    return output_path


def build_specs() -> list[TableSpec]:
    return [
        TableSpec(
            filename="report_table_1.png",
            title="Таблица 1. Сравнение алгоритмов для прогнозирования нагрузки",
            caption="Подпись: сводное сравнение моделей по качеству, интерпретируемости и времени обучения.",
            columns=[
                "Критерий",
                "CatBoost",
                "LSTM",
                "Линейная регрессия",
            ],
            rows=[
                {
                    "Критерий": "Точность прогноза",
                    "CatBoost": "Высокая, R² > 0,90",
                    "LSTM": "Высокая при больших объемах данных",
                    "Линейная регрессия": "Средняя, R² ≈ 0,65–0,75",
                },
                {
                    "Критерий": "Интерпретируемость",
                    "CatBoost": "Высокая (SHAP, Feature Importance)",
                    "LSTM": "Низкая",
                    "Линейная регрессия": "Высокая (коэффициенты)",
                },
                {
                    "Критерий": "Работа с пропусками",
                    "CatBoost": "Встроенная обработка пропусков",
                    "LSTM": "Требует заполнения пропусков",
                    "Линейная регрессия": "Требует удаления или заполнения пропусков",
                },
                {
                    "Критерий": "Время обучения",
                    "CatBoost": "5–15 минут CPU",
                    "LSTM": "1–6 часов GPU",
                    "Линейная регрессия": "< 1 минуты CPU",
                },
                {
                    "Критерий": "Устойчивость к шуму",
                    "CatBoost": "Высокая: L2-регуляризация и shrinkage",
                    "LSTM": "Средняя",
                    "Линейная регрессия": "Низкая",
                },
            ],
            column_weights=[1.75, 2.15, 2.0, 2.5],
            wrap_widths=[18, 24, 24, 24],
            header_wrap_widths=[18, 12, 12, 18],
            alignments=["left", "center", "center", "center"],
            figsize=(12.4, 5.4),
            font_size=9.1,
            title_size=13.5,
            caption_size=8.4,
        ),
        TableSpec(
            filename="report_table_2.png",
            title="Таблица 2. Гиперпараметры модели по типам зон",
            caption="Подпись: ориентировочные настройки модели по типам электропотребляющих зон.",
            columns=[
                "Гиперпараметр",
                "Городская застройка",
                "Промышленная зона",
                "Сельская местность",
            ],
            rows=[
                {
                    "Гиперпараметр": "Глубина деревьев",
                    "Городская застройка": "6–8",
                    "Промышленная зона": "4–6",
                    "Сельская местность": "4–6",
                },
                {
                    "Гиперпараметр": "Скорость обучения",
                    "Городская застройка": "0,05–0,1",
                    "Промышленная зона": "0,03–0,05",
                    "Сельская местность": "0,05–0,1",
                },
                {
                    "Гиперпараметр": "Доля подвыборки",
                    "Городская застройка": "0,8",
                    "Промышленная зона": "0,7",
                    "Сельская местность": "0,6",
                },
                {
                    "Гиперпараметр": "Ранняя остановка, итераций",
                    "Городская застройка": "50",
                    "Промышленная зона": "50",
                    "Сельская местность": "50",
                },
                {
                    "Гиперпараметр": "L2-регуляризация",
                    "Городская застройка": "Средняя",
                    "Промышленная зона": "Высокая",
                    "Сельская местность": "Повышенная",
                },
            ],
            column_weights=[1.65, 1.75, 1.6, 1.6],
            wrap_widths=[18, 18, 18, 18],
            header_wrap_widths=[18, 18, 18, 18],
            alignments=["left", "center", "center", "center"],
            figsize=(11.6, 5.2),
            font_size=9.2,
            title_size=13.5,
            caption_size=8.4,
        ),
        TableSpec(
            filename="report_table_3.png",
            title="Таблица 3. Ожидаемые метрики прогноза по типам зон",
            caption="Подпись: ожидаемые диапазоны MAE, RMSE, R² и горизонта прогноза для разных зон.",
            columns=[
                "Тип зоны",
                "MAE, %",
                "RMSE, %",
                "R²",
                "Горизонт прогноза",
            ],
            rows=[
                {
                    "Тип зоны": "Городская застройка",
                    "MAE, %": "3–5%",
                    "RMSE, %": "5–8%",
                    "R²": "0,90–0,95",
                    "Горизонт прогноза": "1–168 часов",
                },
                {
                    "Тип зоны": "Промышленная зона",
                    "MAE, %": "4–7%",
                    "RMSE, %": "7–10%",
                    "R²": "0,85–0,92",
                    "Горизонт прогноза": "1–72 часа",
                },
                {
                    "Тип зоны": "Сельская местность",
                    "MAE, %": "5–10%",
                    "RMSE, %": "8–15%",
                    "R²": "0,80–0,90",
                    "Горизонт прогноза": "1–168 часов",
                },
            ],
            column_weights=[2.0, 0.9, 0.95, 0.75, 1.55],
            wrap_widths=[20, 8, 8, 10, 16],
            header_wrap_widths=[20, 8, 8, 10, 16],
            alignments=["left", "center", "center", "center", "center"],
            figsize=(10.6, 4.8),
            font_size=9.6,
            title_size=13.5,
            caption_size=8.4,
        ),
        TableSpec(
            filename="report_table_4.png",
            title="Таблица 4. Примеры правил экспертной системы",
            caption="Подпись: меньшие значения в столбце приоритета означают более высокий приоритет правила.",
            columns=[
                "№",
                "ЕСЛИ",
                "ТО",
                "Приоритет",
            ],
            rows=[
                {
                    "№": "1",
                    "ЕСЛИ": "прогноз токовой нагрузки линии превышает 95% от допустимого значения и тренд растет",
                    "ТО": "пометить риск перегрузки, уведомить диспетчера и предложить перераспределение нагрузки",
                    "Приоритет": "2",
                },
                {
                    "№": "2",
                    "ЕСЛИ": "напряжение на шинах выходит за пределы ±5% от номинала",
                    "ТО": "активировать регулирование РПН и компенсацию реактивной мощности",
                    "Приоритет": "2",
                },
                {
                    "№": "3",
                    "ЕСЛИ": "фиксируются перегрев, резкий скачок тока или срабатывание защит",
                    "ТО": "инициировать аварийное отключение и локализацию поврежденного участка",
                    "Приоритет": "1",
                },
                {
                    "№": "4",
                    "ЕСЛИ": "ожидаемый пик нагрузки в ближайшие 1–3 часа превышает порог",
                    "ТО": "заранее включить резерв и предупредить персонал о пике",
                    "Приоритет": "3",
                },
            ],
            column_weights=[0.55, 3.7, 3.4, 0.95],
            wrap_widths=[4, 42, 42, 10],
            header_wrap_widths=[4, 12, 8, 10],
            alignments=["center", "left", "left", "center"],
            figsize=(13.4, 5.6),
            font_size=8.7,
            title_size=13.5,
            caption_size=8.4,
        ),
        TableSpec(
            filename="report_table_5.png",
            title="Таблица 5. Структура базы знаний экспертной системы",
            caption="Подпись: уровни фактов, правил и метазнаний, используемые в базе знаний.",
            columns=[
                "Уровень",
                "Содержимое",
                "Назначение",
            ],
            rows=[
                {
                    "Уровень": "Факты",
                    "Содержимое": "измерения нагрузки, напряжения, состояния линий, календарные и погодные признаки",
                    "Назначение": "хранят текущую картину сети и контекста",
                },
                {
                    "Уровень": "Правила",
                    "Содержимое": "продукционные правила вида ЕСЛИ ... ТО ..., пороги и сценарии действий",
                    "Назначение": "формируют рекомендации и предупреждения",
                },
                {
                    "Уровень": "Метазнания",
                    "Содержимое": "приоритеты правил, разрешение конфликтов, режимы активации и пороговые уровни",
                    "Назначение": "управляют выводом и адаптацией базы знаний",
                },
            ],
            column_weights=[1.2, 3.5, 2.9],
            wrap_widths=[14, 46, 36],
            header_wrap_widths=[12, 14, 14],
            alignments=["left", "left", "left"],
            figsize=(11.6, 4.8),
            font_size=9.2,
            title_size=13.5,
            caption_size=8.4,
        ),
        TableSpec(
            filename="report_table_6.png",
            title="Таблица 6. Сравнение гибридной архитектуры с моно-подходами",
            caption="Подпись: сопоставление гибридной архитектуры, только машинного обучения и только экспертной системы.",
            columns=[
                "Критерий",
                "Гибридная архитектура",
                "Только машинное обучение",
                "Только экспертная система",
            ],
            rows=[
                {
                    "Критерий": "Точность прогноза",
                    "Гибридная архитектура": "Высокая: совмещает данные и правила",
                    "Только машинное обучение": "Высокая на обучающих данных, но хуже вне выборки",
                    "Только экспертная система": "Средняя: зависит от полноты правил",
                },
                {
                    "Критерий": "Объяснимость решений",
                    "Гибридная архитектура": "Высокая: есть и правила, и интерпретация модели",
                    "Только машинное обучение": "Низкая или средняя",
                    "Только экспертная система": "Высокая",
                },
                {
                    "Критерий": "Адаптивность к новым ситуациям",
                    "Гибридная архитектура": "Высокая: можно дообучать модель и обновлять правила",
                    "Только машинное обучение": "Высокая при наличии новых данных",
                    "Только экспертная система": "Низкая без ручного обновления",
                },
                {
                    "Критерий": "Требования к данным",
                    "Гибридная архитектура": "Средние: возможна работа при неполной разметке",
                    "Только машинное обучение": "Высокие: нужны большие наборы данных",
                    "Только экспертная система": "Низкие: достаточно экспертных знаний",
                },
                {
                    "Критерий": "Доверие диспетчерского персонала",
                    "Гибридная архитектура": "Высокое: решения можно проверить по правилам",
                    "Только машинное обучение": "Среднее",
                    "Только экспертная система": "Высокое, но ограничено масштабируемостью",
                },
            ],
            column_weights=[1.65, 2.6, 2.4, 2.4],
            wrap_widths=[24, 34, 32, 32],
            header_wrap_widths=[16, 22, 24, 24],
            alignments=["left", "left", "left", "left"],
            figsize=(14.0, 5.6),
            font_size=8.8,
            title_size=13.5,
            caption_size=8.4,
        ),
        TableSpec(
            filename="report_table_7.png",
            title="Таблица 7. Ожидаемые показатели алгоритма выявления и локализации повреждений",
            caption="Подпись: целевые значения скорости обнаружения, точности локализации и доли ложных срабатываний по зонам.",
            columns=[
                "Тип зоны",
                "Время обнаружения",
                "Точность локализации",
                "Ложные срабатывания",
            ],
            rows=[
                {
                    "Тип зоны": "Городская застройка\n(кабельная сеть)",
                    "Время обнаружения": "< 10 с",
                    "Точность локализации": "≤ 50 м",
                    "Ложные срабатывания": "< 1 %",
                },
                {
                    "Тип зоны": "Промышленная зона\n(смешанная сеть)",
                    "Время обнаружения": "< 15 с",
                    "Точность локализации": "≤ 100 м",
                    "Ложные срабатывания": "< 2 %",
                },
                {
                    "Тип зоны": "Сельская местность\n(воздушная сеть)",
                    "Время обнаружения": "< 30 с",
                    "Точность локализации": "≤ 500 м",
                    "Ложные срабатывания": "< 3 %",
                },
            ],
            column_weights=[1.9, 1.35, 1.45, 1.25],
            wrap_widths=[24, 16, 18, 16],
            header_wrap_widths=[14, 16, 18, 16],
            alignments=["left", "center", "center", "center"],
            figsize=(12.4, 4.9),
            font_size=9.0,
            title_size=13.5,
            caption_size=8.4,
        ),
        TableSpec(
            filename="report_table_8.png",
            title="Таблица 8. Ожидаемые показатели алгоритма изоляции и реконфигурации сети",
            caption="Подпись: целевые значения времени изоляции и восстановления питания по типам зон.",
            columns=[
                "Тип зоны",
                "Время изоляции",
                "Время восстановления",
                "Доля восстановленных потребителей",
            ],
            rows=[
                {
                    "Тип зоны": "Городская застройка",
                    "Время изоляции": "< 30 с",
                    "Время восстановления": "< 5 мин",
                    "Доля восстановленных потребителей": "> 90 %",
                },
                {
                    "Тип зоны": "Промышленная зона",
                    "Время изоляции": "< 60 с",
                    "Время восстановления": "< 10 мин",
                    "Доля восстановленных потребителей": "> 85 %",
                },
                {
                    "Тип зоны": "Сельская местность",
                    "Время изоляции": "< 120 с",
                    "Время восстановления": "< 30 мин",
                    "Доля восстановленных потребителей": "> 75 %",
                },
            ],
            column_weights=[1.7, 1.15, 1.25, 1.8],
            wrap_widths=[22, 14, 14, 20],
            header_wrap_widths=[14, 14, 18, 22],
            alignments=["left", "center", "center", "center"],
            figsize=(12.6, 4.9),
            font_size=8.9,
            title_size=13.5,
            caption_size=8.4,
        ),
        TableSpec(
            filename="report_table_9.png",
            title="Таблица 9. Ожидаемое снижение потерь электроэнергии по типам зон",
            caption="Подпись: целевые диапазоны снижения технических потерь после внедрения алгоритма.",
            columns=[
                "Тип зоны",
                "Текущие потери, %",
                "Ожидаемые потери, %",
                "Снижение, %",
            ],
            rows=[
                {
                    "Тип зоны": "Городская застройка",
                    "Текущие потери, %": "6–8 %",
                    "Ожидаемые потери, %": "4–6 %",
                    "Снижение, %": "25–30 %",
                },
                {
                    "Тип зоны": "Промышленная зона",
                    "Текущие потери, %": "8–12 %",
                    "Ожидаемые потери, %": "6–9 %",
                    "Снижение, %": "20–25 %",
                },
                {
                    "Тип зоны": "Сельская местность",
                    "Текущие потери, %": "12–18 %",
                    "Ожидаемые потери, %": "9–14 %",
                    "Снижение, %": "25–30 %",
                },
            ],
            column_weights=[1.7, 1.25, 1.3, 1.0],
            wrap_widths=[22, 16, 16, 12],
            header_wrap_widths=[14, 16, 16, 12],
            alignments=["left", "center", "center", "center"],
            figsize=(12.0, 4.8),
            font_size=9.1,
            title_size=13.5,
            caption_size=8.4,
        ),
    ]


def main() -> int:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    created_paths: list[Path] = []
    for spec in build_specs():
        created_paths.append(render_table(spec))

    print("Созданы файлы:")
    for path in created_paths:
        print(f"- {path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
