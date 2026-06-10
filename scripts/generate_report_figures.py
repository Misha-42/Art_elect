from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "plots"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 150,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
    }
)


COLORS = {
    "bg": "#F8FAFC",
    "card": "#FFFFFF",
    "card2": "#EEF6FF",
    "card3": "#E9F7F0",
    "card4": "#FFF4E8",
    "blue": "#2563EB",
    "blue_dark": "#1E3A8A",
    "cyan": "#0EA5E9",
    "teal": "#14B8A6",
    "green": "#16A34A",
    "lime": "#65A30D",
    "orange": "#F59E0B",
    "amber": "#D97706",
    "red": "#DC2626",
    "rose": "#E11D48",
    "slate": "#475569",
    "muted": "#94A3B8",
    "ink": "#0F172A",
    "grid": "#D9E2F2",
}


def wrap(text: str, width: int = 20) -> str:
    parts = []
    for line in str(text).split("\n"):
        parts.append(
            "\n".join(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=False,
                    replace_whitespace=False,
                )
            )
            if line.strip()
            else ""
        )
    return "\n".join(parts)


def setup_canvas(figsize=(12, 7), title: str | None = None, subtitle: str | None = None):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    if title:
        ax.text(
            0.5,
            0.965,
            title,
            ha="center",
            va="top",
            fontsize=18,
            color=COLORS["ink"],
            fontweight="bold",
        )
    if subtitle:
        ax.text(
            0.5,
            0.925,
            subtitle,
            ha="center",
            va="top",
            fontsize=10.5,
            color=COLORS["slate"],
        )

    return fig, ax


def save_figure(fig, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def add_shadow_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    *,
    facecolor=COLORS["card"],
    edgecolor=COLORS["blue"],
    textcolor=COLORS["ink"],
    fontsize=12,
    weight="normal",
    radius=0.02,
    lw=1.8,
    align="center",
    va="center",
    zorder=2,
):
    shadow = FancyBboxPatch(
        (x + 0.008, y - 0.008),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=0,
        facecolor="#CBD5E1",
        alpha=0.25,
        zorder=zorder - 1,
    )
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(shadow)
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        wrap(text, 24),
        ha=align,
        va=va,
        fontsize=fontsize,
        color=textcolor,
        fontweight=weight,
        zorder=zorder + 1,
    )
    return box


def add_arrow(ax, start, end, color=COLORS["blue"], lw=1.8, rad=0.0, arrowstyle="-|>"):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=arrowstyle,
        mutation_scale=16,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)
    return arrow


def add_caption(ax, text: str):
    ax.text(
        0.5,
        0.03,
        text,
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=COLORS["slate"],
    )


def add_tag(ax, x, y, text, color=COLORS["blue"]):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color=color,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25,rounding_size=0.08",
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
        ),
    )


def draw_grid_background(ax):
    for x in np.linspace(0.08, 0.92, 10):
        ax.plot([x, x], [0.1, 0.88], color="#E2E8F0", lw=0.5, zorder=0)
    for y in np.linspace(0.12, 0.86, 8):
        ax.plot([0.08, 0.92], [y, y], color="#E2E8F0", lw=0.5, zorder=0)


def fig_021_architecture():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.1 — Трёхуровневая архитектура интеллектуальной системы управления",
        "Полевой уровень → сетевой уровень → корпоративный уровень",
    )
    draw_grid_background(ax)

    add_shadow_box(
        ax,
        0.08,
        0.18,
        0.24,
        0.56,
        "Полевой уровень\n\nТерминальные устройства (RTU)\nРЗА\nинтеллектуальные датчики\n\nДискретность:\n1 с – 1 мин",
        facecolor=COLORS["card2"],
        edgecolor=COLORS["blue"],
        fontsize=11,
        weight="bold",
    )
    add_shadow_box(
        ax,
        0.38,
        0.18,
        0.24,
        0.56,
        "Сетевой уровень\n\nПромышленные шлюзы\nModbus → IEC 60870-5-104\nOPC UA\nБуферизация до 72 ч",
        facecolor=COLORS["card4"],
        edgecolor=COLORS["orange"],
        fontsize=11,
        weight="bold",
    )
    add_shadow_box(
        ax,
        0.68,
        0.18,
        0.24,
        0.56,
        "Корпоративный уровень\n\nХранилище временных рядов (TimescaleDB) на PostgreSQL\nМодель CatBoost (CatBoost)\nЭкспертная система\nдиспетчерский контур (SCADA)\nчеловеко-машинный интерфейс (HMI)",
        facecolor=COLORS["card3"],
        edgecolor=COLORS["green"],
        fontsize=11,
        weight="bold",
    )

    add_arrow(ax, (0.32, 0.46), (0.38, 0.46))
    add_arrow(ax, (0.62, 0.46), (0.68, 0.46))
    add_tag(ax, 0.2, 0.8, "Сбор")
    add_tag(ax, 0.5, 0.8, "Маршрутизация")
    add_tag(ax, 0.8, 0.8, "Аналитика")
    add_caption(ax, "Трёхуровневая логика обеспечивает масштабируемость, отказоустойчивость и бесшовную интеграцию с диспетчерским контуром (SCADA).")
    return fig


def fig_022_modules():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.2 — Схема взаимодействия функциональных модулей интеллектуальной системы управления",
        "Непрерывный цикл обработки данных и формирования рекомендаций",
    )
    draw_grid_background(ax)

    centers = {
        "Мониторинг": (0.5, 0.74),
        "Диагностика": (0.76, 0.48),
        "Прогнозирование": (0.5, 0.22),
        "Управление": (0.24, 0.48),
    }
    box_style = {
        "Мониторинг": (COLORS["card2"], COLORS["blue"]),
        "Диагностика": (COLORS["card4"], COLORS["orange"]),
        "Прогнозирование": (COLORS["card3"], COLORS["green"]),
        "Управление": ("#F4F0FF", "#7C3AED"),
    }
    notes = {
        "Мониторинг": "Сбор, валидация,\nагрегация",
        "Диагностика": "Норма / предупреждение / авария",
        "Прогнозирование": "1–168 ч\nМодель CatBoost (CatBoost)",
        "Управление": "Рекомендации\nв режиме советчика",
    }
    sizes = (0.22, 0.14)
    for name, (cx, cy) in centers.items():
        fc, ec = box_style[name]
        add_shadow_box(
            ax,
            cx - sizes[0] / 2,
            cy - sizes[1] / 2,
            sizes[0],
            sizes[1],
            f"{name}\n\n{notes[name]}",
            facecolor=fc,
            edgecolor=ec,
            fontsize=11,
            weight="bold",
        )

    add_arrow(ax, (0.61, 0.74), (0.7, 0.58), color=COLORS["orange"], rad=-0.15)
    add_arrow(ax, (0.76, 0.42), (0.61, 0.26), color=COLORS["green"], rad=-0.15)
    add_arrow(ax, (0.39, 0.22), (0.28, 0.42), color="#7C3AED", rad=-0.15)
    add_arrow(ax, (0.24, 0.54), (0.39, 0.7), color=COLORS["blue"], rad=-0.15)
    ax.text(
        0.5,
        0.5,
        "Контур принятия решений",
        ha="center",
        va="center",
        fontsize=13,
        color=COLORS["ink"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["muted"], linewidth=1.2),
    )
    add_caption(ax, "Каждый модуль работает независимо, но данные и решения проходят через общий замкнутый контур.")
    return fig


def fig_023_zone_map():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.3 — Карта зон распределительной сети г. Ишимбая и прилегающих районов",
        "Схематичное деление территории на три типа зон",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    outline = np.array(
        [
            [0.10, 0.44],
            [0.16, 0.76],
            [0.28, 0.86],
            [0.46, 0.90],
            [0.62, 0.84],
            [0.76, 0.76],
            [0.88, 0.60],
            [0.90, 0.42],
            [0.84, 0.24],
            [0.72, 0.14],
            [0.52, 0.10],
            [0.34, 0.14],
            [0.18, 0.24],
        ]
    )
    ax.add_patch(
        Polygon(
            outline,
            closed=True,
            facecolor="#F8FAFC",
            edgecolor=COLORS["slate"],
            linewidth=2.0,
            zorder=1,
        )
    )
    ax.add_patch(
        Ellipse(
            (0.48, 0.52),
            0.34,
            0.24,
            facecolor="#DBEAFE",
            edgecolor=COLORS["blue"],
            linewidth=1.8,
            alpha=0.85,
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle(
            (0.24, 0.45),
            0.48,
            0.12,
            facecolor="#FDE68A",
            edgecolor=COLORS["orange"],
            linewidth=1.8,
            alpha=0.75,
            zorder=3,
        )
    )
    ax.add_patch(
        Polygon(
            np.array([[0.18, 0.26], [0.76, 0.26], [0.80, 0.40], [0.22, 0.40]]),
            closed=True,
            facecolor="#DCFCE7",
            edgecolor=COLORS["green"],
            linewidth=1.8,
            alpha=0.55,
            zorder=2,
        )
    )
    ax.text(0.48, 0.52, "Городская\nзастройка", ha="center", va="center", fontsize=13, color=COLORS["blue_dark"], fontweight="bold", zorder=4)
    ax.text(0.48, 0.52, "\n\n\nПлотная кабельная сеть", ha="center", va="center", fontsize=9.5, color=COLORS["blue_dark"], zorder=4)
    ax.text(0.50, 0.57, "Промышленная зона", ha="center", va="center", fontsize=12, color=COLORS["amber"], fontweight="bold", zorder=4)
    ax.text(0.48, 0.30, "Сельская местность", ha="center", va="center", fontsize=12, color=COLORS["green"], fontweight="bold", zorder=4)

    legend_x = 0.94
    add_shadow_box(
        ax,
        0.83,
        0.72,
        0.14,
        0.16,
        "Легенда\n\n■ город\n■ промзона\n■ село",
        facecolor="white",
        edgecolor=COLORS["muted"],
        fontsize=9.5,
        weight="bold",
    )
    add_tag(ax, 0.16, 0.82, "Центр")
    add_tag(ax, 0.78, 0.18, "Периферия")
    add_tag(ax, 0.74, 0.58, "Промышленный пояс", color=COLORS["amber"])
    add_caption(ax, "Зоны используются для дифференцированного выбора параметров прогнозирования и управления.")
    return fig


def fig_024_information_flow():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.4 — Схема информационных потоков интеллектуальной системы управления",
        "От источников телеметрии к моделям и управляющим рекомендациям",
    )
    draw_grid_background(ax)

    source_y = 0.74
    source_specs = [
        (0.10, "Телеметрия (SCADA)", COLORS["card2"], COLORS["blue"], "U, I, P, Q, f"),
        (0.10, "АИИС КУЭ", COLORS["card4"], COLORS["orange"], "Почасовой учёт"),
        (0.10, "Паспорт оборудования", COLORS["card3"], COLORS["green"], "Надёжность"),
        (0.10, "Метеоданные / ГИС / Календарь", "#F4F0FF", "#7C3AED", "Контекст"),
    ]
    for i, (x, title, fc, ec, sub) in enumerate(source_specs):
        add_shadow_box(
            ax,
            x,
            source_y - i * 0.13,
            0.24,
            0.10,
            f"{title}\n{sub}",
            facecolor=fc,
            edgecolor=ec,
            fontsize=10.5,
            weight="bold",
        )

    add_shadow_box(
        ax,
        0.40,
        0.56,
        0.20,
        0.14,
        "Предобработка\n\nочистка → синхронизация → восстановление",
        facecolor="white",
        edgecolor=COLORS["blue"],
        fontsize=10.5,
        weight="bold",
    )
    add_shadow_box(
        ax,
        0.40,
        0.33,
        0.20,
        0.14,
        "Формирование признаков\n\nлаги • окна • календарь",
        facecolor="white",
        edgecolor=COLORS["green"],
        fontsize=10.5,
        weight="bold",
    )
    add_shadow_box(
        ax,
        0.70,
        0.56,
        0.20,
        0.14,
        "Модели машинного обучения (ML)\n\nCatBoost (CatBoost) / градиентный бустинг (LightGBM) / многослойный перцептрон (MLP)",
        facecolor=COLORS["card2"],
        edgecolor=COLORS["blue_dark"],
        fontsize=10.5,
        weight="bold",
    )
    add_shadow_box(
        ax,
        0.70,
        0.33,
        0.20,
        0.14,
        "Управляющие рекомендации\n\nдиспетчерский контур (SCADA) / АРМ диспетчера",
        facecolor=COLORS["card3"],
        edgecolor=COLORS["green"],
        fontsize=10.5,
        weight="bold",
    )

    for y in [0.79, 0.66, 0.53, 0.40]:
        add_arrow(ax, (0.34, y), (0.40, 0.63), color=COLORS["blue"])
    add_arrow(ax, (0.60, 0.63), (0.70, 0.63), color=COLORS["orange"])
    add_arrow(ax, (0.80, 0.56), (0.80, 0.47), color=COLORS["green"])
    add_arrow(ax, (0.50, 0.33), (0.80, 0.40), color=COLORS["teal"], rad=0.12)
    ax.text(
        0.5,
        0.84,
        "Сквозной информационный конвейер",
        ha="center",
        va="center",
        fontsize=13,
        color=COLORS["ink"],
        fontweight="bold",
    )
    add_caption(ax, "Все каналы сведены в единое пространство данных с обратной связью в диспетчерский контур (SCADA).")
    return fig


def fig_025_sources():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.5 — Источники данных интеллектуальной системы управления",
        "Шесть каналов данных для мониторинга, прогнозирования и диагностики",
    )
    draw_grid_background(ax)

    cards = [
        (0.08, 0.58, "Диспетчерская телеметрия (SCADA) / РЗА", "Оперативная телеметрия", COLORS["card2"], COLORS["blue"], "1 с – 1 мин"),
        (0.38, 0.58, "АИИС КУЭ", "Коммерческий учёт", COLORS["card4"], COLORS["orange"], "1 ч / 30 мин"),
        (0.68, 0.58, "Паспорт оборудования", "Техническое состояние активов", COLORS["card3"], COLORS["green"], "по событию"),
        (0.08, 0.26, "Метеоданные", "Температура, ветер, влажность", "#F4F0FF", "#7C3AED", "форматы JSON / CSV"),
        (0.38, 0.26, "ГИС", "Топология сети и координаты", "#FFF7ED", COLORS["amber"], "GeoJSON"),
        (0.68, 0.26, "Календарь / сменность", "Сезонность и режимы нагрузки", "#ECFEFF", COLORS["cyan"], "CSV"),
    ]
    for x, y, title, subtitle, fc, ec, note in cards:
        add_shadow_box(
            ax,
            x,
            y,
            0.24,
            0.18,
            f"{title}\n\n{subtitle}\n\n{note}",
            facecolor=fc,
            edgecolor=ec,
            fontsize=10.5,
            weight="bold",
        )

    ax.text(
        0.5,
        0.81,
        "Источники данных",
        ha="center",
        va="center",
        fontsize=14,
        color=COLORS["ink"],
        fontweight="bold",
    )
    add_caption(ax, "Входные потоки используются для валидации, обогащения признаков и кросс-проверки телеметрии.")
    return fig


def fig_026_preprocessing():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.6 — Конвейер предобработки данных интеллектуальной системы управления",
        "Каскад от сырых данных к обучающим выборкам",
    )
    draw_grid_background(ax)
    steps = [
        ("Входные данные", "телеметрия (SCADA) / АИИС / внешние источники", COLORS["card2"], COLORS["blue"]),
        ("Фильтрация шума", "Физические пределы и выбросы", COLORS["card4"], COLORS["orange"]),
        ("Восстановление пропусков", "Интерполяция / буферизация / восстановление пропусков (Imputation)", COLORS["card3"], COLORS["green"]),
        ("Нормализация и валидация", "Единый формат и контроль качества", "#F4F0FF", "#7C3AED"),
    ]
    xs = [0.08, 0.31, 0.54, 0.77]
    for (title, subtitle, fc, ec), x in zip(steps, xs):
        add_shadow_box(
            ax,
            x,
            0.44,
            0.17,
            0.20,
            f"{title}\n\n{subtitle}",
            facecolor=fc,
            edgecolor=ec,
            fontsize=10.5,
            weight="bold",
        )
    for start_x in [0.25, 0.48, 0.71]:
        add_arrow(ax, (start_x, 0.54), (start_x + 0.06, 0.54))
    ax.add_patch(
        FancyBboxPatch(
            (0.23, 0.20),
            0.54,
            0.12,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="white",
            edgecolor=COLORS["muted"],
            linewidth=1.3,
        )
    )
    ax.text(0.50, 0.26, "Выход: обучающая выборка и наборы для инференса", ha="center", va="center", fontsize=11, color=COLORS["ink"], fontweight="bold")
    add_caption(ax, "Каскадная предобработка изолирует аппаратные погрешности и разрывы связи.")
    return fig


def fig_027_features():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.7 — Процесс формирования признаков интеллектуальной системы управления",
        "Временные, пространственные и внешние признаки объединяются в единый вектор",
    )
    draw_grid_background(ax)

    add_shadow_box(
        ax,
        0.06,
        0.56,
        0.22,
        0.24,
        "Временные признаки\n\nлаги t-1, t-24, t-168\nскользящие окна\nкалендарные индикаторы",
        facecolor=COLORS["card2"],
        edgecolor=COLORS["blue"],
        fontsize=10.2,
        weight="bold",
    )
    add_shadow_box(
        ax,
        0.06,
        0.22,
        0.22,
        0.24,
        "Пространственные признаки\n\nфидер / подстанция\nтип зоны\nтопологическое расстояние",
        facecolor=COLORS["card3"],
        edgecolor=COLORS["green"],
        fontsize=10.2,
        weight="bold",
    )
    add_shadow_box(
        ax,
        0.72,
        0.39,
        0.22,
        0.22,
        "Внешние признаки\n\nметео\nсменность\nистория аварийности",
        facecolor=COLORS["card4"],
        edgecolor=COLORS["orange"],
        fontsize=10.2,
        weight="bold",
    )
    add_shadow_box(
        ax,
        0.38,
        0.36,
        0.24,
        0.22,
        "Хранилище признаков (Feature Store)\n\nкодирование • взаимодействия • отбор",
        facecolor="white",
        edgecolor=COLORS["blue_dark"],
        fontsize=11,
        weight="bold",
    )
    add_shadow_box(
        ax,
        0.38,
        0.12,
        0.24,
        0.12,
        "Матрица признаков\n\nготова к обучению моделей",
        facecolor="#EEFDF8",
        edgecolor=COLORS["teal"],
        fontsize=10.5,
        weight="bold",
    )
    add_arrow(ax, (0.28, 0.68), (0.38, 0.47), color=COLORS["blue"])
    add_arrow(ax, (0.28, 0.34), (0.38, 0.47), color=COLORS["green"])
    add_arrow(ax, (0.72, 0.50), (0.62, 0.47), color=COLORS["orange"])
    add_arrow(ax, (0.50, 0.36), (0.50, 0.24), color=COLORS["teal"])
    add_caption(ax, "Инженерия признаков учитывает режимы работы сети, сезонность и контекст внешней среды.")
    return fig


def fig_028_training():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.8 — Процедура обучения и настройки модели прогнозирования",
        "Пошаговый цикл подготовки модели CatBoost (CatBoost)",
    )
    draw_grid_background(ax)

    stages = [
        ("1", "Разделение данных", "Обучение 70%\nВалидация 15%\nТест 15%\nбез перемешивания (shuffle=False)", COLORS["card2"], COLORS["blue"]),
        ("2", "Подбор гиперпараметров", "Полный поиск / случайный поиск (Grid / Random Search)\nглубина • скорость обучения (learning_rate)\nколичество итераций", COLORS["card4"], COLORS["orange"]),
        ("3", "Регуляризация", "L2\nмаксимальная глубина ≤ 8 (max_depth)\nподвыборка ≤ 0,8 (subsample)", COLORS["card3"], COLORS["green"]),
        ("4", "Ранняя остановка", "ранняя остановка: 50 раундов (early_stopping_rounds)\nсохранение лучшей версии", "#F4F0FF", "#7C3AED"),
    ]
    for i, (n, title, subtitle, fc, ec) in enumerate(stages):
        x = 0.08 + i * 0.23
        add_shadow_box(
            ax,
            x,
            0.48,
            0.18,
            0.22,
            f"{n}\n{title}\n\n{subtitle}",
            facecolor=fc,
            edgecolor=ec,
            fontsize=10.3,
            weight="bold",
        )
        if i < len(stages) - 1:
            add_arrow(ax, (x + 0.18, 0.59), (x + 0.23, 0.59), color=ec)

    add_shadow_box(
        ax,
        0.18,
        0.18,
        0.64,
        0.16,
        "Зональная настройка: городская застройка • промышленная зона • сельская местность",
        facecolor="white",
        edgecolor=COLORS["muted"],
        fontsize=11,
        weight="bold",
    )
    add_tag(ax, 0.30, 0.14, "глубина 6–8")
    add_tag(ax, 0.50, 0.14, "скорость обучения 0,03–0,10")
    add_tag(ax, 0.70, 0.14, "фиксированное зерно 42")
    add_caption(ax, "Воспроизводимость обеспечивается фиксированием параметров, временем и конфигурацией моделей.")
    return fig


def fig_029_metrics():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor(COLORS["bg"])
    zones = ["Городская", "Промышленная", "Сельская"]
    data = {
        "Средняя абсолютная ошибка (MAE), %": ([4.0, 5.5, 7.5], [1.0, 1.5, 2.5], COLORS["blue"]),
        "Среднеквадратичная ошибка (RMSE), %": ([6.5, 8.5, 11.5], [1.5, 1.5, 3.5], COLORS["orange"]),
        "Коэффициент детерминации (R²)": ([0.925, 0.885, 0.850], [0.025, 0.035, 0.050], COLORS["green"]),
    }
    for ax, (title, (means, errs, color)) in zip(axes, data.items()):
        ax.set_facecolor("white")
        x = np.arange(len(zones))
        ax.bar(x, means, yerr=errs, color=color, alpha=0.85, capsize=6, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(zones, fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold", color=COLORS["ink"])
        ax.grid(axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.8)
        ax.set_axisbelow(True)
        for xi, val in zip(x, means):
            ax.text(xi, val + max(errs) * 0.12, f"{val:.3f}" if "R²" in title else f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        if "R²" in title:
            ax.set_ylim(0.78, 0.98)
        else:
            ax.set_ylim(0, 15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(COLORS["muted"])
        ax.spines["bottom"].set_color(COLORS["muted"])

    fig.suptitle(
        "Рисунок 2.9 — Метрики оценки точности модели прогнозирования",
        fontsize=17,
        fontweight="bold",
        y=0.98,
        color=COLORS["ink"],
    )
    fig.text(
        0.5,
        0.02,
        "Диапазоны взяты из текста отчёта: городская, промышленная и сельская зоны.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=COLORS["slate"],
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.92])
    return fig


def fig_0210_sources_for_rules():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.10 — Источники знаний для формирования базы правил экспертной системы",
        "Нормативная документация, опыт диспетчеров и архив решений",
    )
    draw_grid_background(ax)
    source_cards = [
        (0.08, 0.28, "Нормативно-техническая\nдокументация", "ПТЭЭП\nГОСТ 32144-2021\nинструкции по переключениям", COLORS["card2"], COLORS["blue"]),
        (0.39, 0.28, "Опыт диспетчерского\nперсонала", "Эвристики\nтиповые и нештатные сценарии", COLORS["card4"], COLORS["orange"]),
        (0.70, 0.28, "История принятия\nрешений", "Архив 12+ месяцев\nпроверка и уточнение правил", COLORS["card3"], COLORS["green"]),
    ]
    for x, y, title, subtitle, fc, ec in source_cards:
        add_shadow_box(
            ax,
            x,
            y,
            0.22,
            0.26,
            f"{title}\n\n{subtitle}",
            facecolor=fc,
            edgecolor=ec,
            fontsize=10.5,
            weight="bold",
        )
    add_shadow_box(
        ax,
        0.34,
        0.66,
        0.32,
        0.14,
        "Формирование базы правил\n\nЕСЛИ → ТО → ПРИОРИТЕТ",
        facecolor="white",
        edgecolor=COLORS["blue_dark"],
        fontsize=11,
        weight="bold",
    )
    add_arrow(ax, (0.19, 0.54), (0.45, 0.66), color=COLORS["blue"], rad=0.08)
    add_arrow(ax, (0.50, 0.54), (0.50, 0.66), color=COLORS["orange"])
    add_arrow(ax, (0.81, 0.54), (0.55, 0.66), color=COLORS["green"], rad=-0.08)
    add_caption(ax, "Знания проходят верификацию и превращаются в формализованные продукционные правила.")
    return fig


def fig_0211_knowledge_base():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.11 — Структура базы знаний экспертной системы",
        "Факты, правила и метазнания формируют единый контур вывода",
    )
    draw_grid_background(ax)
    layers = [
        (0.10, 0.62, 0.34, 0.12, "Уровень фактов\nПараметры сети в реальном времени", COLORS["card2"], COLORS["blue"]),
        (0.10, 0.42, 0.34, 0.12, "Уровень правил\nПродукционные правила «ЕСЛИ–ТО»", COLORS["card4"], COLORS["orange"]),
        (0.10, 0.22, 0.34, 0.12, "Уровень метазнаний\nИстория, статистика, связи", COLORS["card3"], COLORS["green"]),
    ]
    for x, y, w, h, text, fc, ec in layers:
        add_shadow_box(ax, x, y, w, h, text, facecolor=fc, edgecolor=ec, fontsize=10.5, weight="bold")

    add_shadow_box(
        ax,
        0.56,
        0.40,
        0.34,
        0.28,
        "Экспертный движок\n\n1. Сопоставление (Matching)\n2. Разрешение конфликтов (Conflict Resolution)\n3. Исполнение (Acting)\n4. Журналирование (Logging)",
        facecolor="white",
        edgecolor=COLORS["blue_dark"],
        fontsize=11,
        weight="bold",
    )
    add_arrow(ax, (0.44, 0.68), (0.56, 0.60), color=COLORS["blue"])
    add_arrow(ax, (0.44, 0.48), (0.56, 0.54), color=COLORS["orange"])
    add_arrow(ax, (0.44, 0.28), (0.56, 0.48), color=COLORS["green"])
    add_tag(ax, 0.73, 0.73, "Прямой вывод")
    add_caption(ax, "Структура базы знаний поддерживает быстрый логический вывод и последующий аудит правил.")
    return fig


def fig_0212_hybrid_architecture():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.12 — Схема взаимодействия модулей в гибридной архитектуре системы",
        "Модель CatBoost (CatBoost) + экспертная система + проверка нормативных ограничений",
    )
    draw_grid_background(ax)
    nodes = [
        (0.10, 0.52, 0.18, 0.16, "Прогнозирование\nмодель CatBoost (CatBoost)", COLORS["card2"], COLORS["blue"]),
        (0.36, 0.72, 0.20, 0.16, "Генерация рекомендаций\nЭкспертная система", COLORS["card4"], COLORS["orange"]),
        (0.62, 0.52, 0.18, 0.16, "Верификация\nНормативы и топология", COLORS["card3"], COLORS["green"]),
        (0.36, 0.22, 0.20, 0.16, "Исполнение\nдиспетчерский контур (SCADA) / АРМ", "#F4F0FF", "#7C3AED"),
    ]
    for x, y, w, h, text, fc, ec in nodes:
        add_shadow_box(ax, x, y, w, h, text, facecolor=fc, edgecolor=ec, fontsize=10.4, weight="bold")

    add_arrow(ax, (0.28, 0.60), (0.36, 0.80), color=COLORS["blue"], rad=0.08)
    add_arrow(ax, (0.56, 0.80), (0.62, 0.60), color=COLORS["orange"], rad=0.08)
    add_arrow(ax, (0.71, 0.52), (0.56, 0.30), color=COLORS["green"], rad=0.08)
    add_arrow(ax, (0.36, 0.30), (0.28, 0.60), color="#7C3AED", rad=0.08)
    ax.text(
        0.50,
        0.52,
        "Приоритет нормативных ограничений",
        ha="center",
        va="center",
        fontsize=12,
        color=COLORS["ink"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLORS["muted"], linewidth=1.2),
    )
    add_caption(ax, "Гибридная схема сочетает вероятностный прогноз и детерминированные правила.")
    return fig


def fig_0213_fault_localization():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.13 — Алгоритм выявления и локализации повреждений в распределительной сети",
        "Диагностика аварии по телеметрии, сигнатурам и РЗА",
    )
    draw_grid_background(ax)
    steps = [
        ("1", "Мониторинг параметров", "Ток, напряжение, мощность,\nстатусы коммутационных аппаратов", COLORS["card2"], COLORS["blue"]),
        ("2", "Анализ сигнатур аварий", "КЗ, обрыв фазы,\nзамыкание на землю, перегрузка", COLORS["card4"], COLORS["orange"]),
        ("3", "Локализация", "Импедансный метод /\nрефлектометрия", COLORS["card3"], COLORS["green"]),
        ("4", "Карта аварии", "Координаты, тип повреждения,\nрекомендации бригаде", "#F4F0FF", "#7C3AED"),
    ]
    xs = [0.08, 0.31, 0.54, 0.77]
    for i, ((n, title, subtitle, fc, ec), x) in enumerate(zip(steps, xs)):
        add_shadow_box(ax, x, 0.44, 0.17, 0.22, f"{n}\n{title}\n\n{subtitle}", facecolor=fc, edgecolor=ec, fontsize=9.9, weight="bold")
        if i < len(steps) - 1:
            add_arrow(ax, (x + 0.17, 0.55), (x + 0.23, 0.55), color=ec)

    ax.add_patch(Circle((0.50, 0.25), 0.08, facecolor="#FEE2E2", edgecolor=COLORS["red"], linewidth=1.8))
    ax.text(0.50, 0.25, "АВАРИЯ", ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["red"])
    add_caption(ax, "Порог срабатывания связан с отклонениями тока и напряжения от допустимых границ.")
    return fig


def fig_0214_isolation_reconfiguration():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.14 — Алгоритм изоляции аварийного участка и реконфигурации сети",
        "Изоляция повреждения, перевод нагрузки на резерв и восстановление питания",
    )
    draw_grid_background(ax)
    panel_positions = [0.08, 0.30, 0.52, 0.74]
    panel_titles = ["Обнаружение", "Изоляция", "Переключение", "Восстановление"]
    panel_subtitles = [
        "Аварийный участок\nна линии",
        "Открытие\nкоммутационного аппарата",
        "Питание от резерва\nи перераспределение потоков",
        "Нормализация\nи контроль параметров",
    ]
    panel_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["teal"]]
    for x, title, subtitle, color in zip(panel_positions, panel_titles, panel_subtitles, panel_colors):
        add_shadow_box(ax, x, 0.40, 0.18, 0.24, f"{title}\n\n{subtitle}", facecolor="white", edgecolor=color, fontsize=10.2, weight="bold")

    # Simplified feeder line across the panels
    for i, x in enumerate([0.15, 0.37, 0.59, 0.81]):
        ax.plot([x - 0.04, x + 0.04], [0.30, 0.30], color=COLORS["slate"], lw=3)
        ax.add_patch(Rectangle((x - 0.02, 0.275), 0.04, 0.05, facecolor="white", edgecolor=COLORS["slate"], linewidth=1.2))
    ax.plot([0.10, 0.88], [0.30, 0.30], color=COLORS["slate"], lw=2.2, alpha=0.8)
    # Fault marker
    ax.plot([0.33, 0.36], [0.30, 0.34], color=COLORS["red"], lw=3)
    ax.plot([0.33, 0.36], [0.34, 0.30], color=COLORS["red"], lw=3)
    # Reserve path
    ax.plot([0.60, 0.60], [0.30, 0.18], color=COLORS["green"], lw=2.5)
    ax.plot([0.60, 0.82], [0.18, 0.18], color=COLORS["green"], lw=2.5)
    ax.add_patch(Rectangle((0.78, 0.16), 0.04, 0.04, facecolor="white", edgecolor=COLORS["green"], linewidth=1.3))
    add_arrow(ax, (0.18, 0.22), (0.28, 0.22), color=COLORS["blue"])
    add_arrow(ax, (0.40, 0.22), (0.50, 0.22), color=COLORS["orange"])
    add_arrow(ax, (0.62, 0.22), (0.72, 0.22), color=COLORS["green"])
    add_arrow(ax, (0.84, 0.22), (0.90, 0.22), color=COLORS["teal"])
    add_caption(ax, "Алгоритм обеспечивает быструю локализацию и минимальное время простоя потребителей.")
    return fig


def fig_0215_power_flow():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.15 — Алгоритм оптимизации потоков мощности в распределительной сети",
        "Минимизация технических потерь и выравнивание загрузки линий",
    )
    draw_grid_background(ax)
    steps = [
        ("1", "Сбор параметров", "P, Q, U, I\nзагрузка линий", COLORS["card2"], COLORS["blue"]),
        ("2", "Оптимизационная модель", "ограничения по топологии\nи допустимым режимам", COLORS["card4"], COLORS["orange"]),
        ("3", "Расчёт сценариев", "перераспределение потоков\nпо альтернативным путям", COLORS["card3"], COLORS["green"]),
        ("4", "Результат", "снижение потерь\nи повышение устойчивости", "#F4F0FF", "#7C3AED"),
    ]
    xs = [0.08, 0.31, 0.54, 0.77]
    for i, ((n, title, subtitle, fc, ec), x) in enumerate(zip(steps, xs)):
        add_shadow_box(ax, x, 0.44, 0.17, 0.22, f"{n}\n{title}\n\n{subtitle}", facecolor=fc, edgecolor=ec, fontsize=10.0, weight="bold")
        if i < len(steps) - 1:
            add_arrow(ax, (x + 0.17, 0.55), (x + 0.23, 0.55), color=ec)
    # Small objective curve
    xx = np.linspace(0.18, 0.82, 150)
    yy = 0.18 + 0.05 * np.sin((xx - 0.18) * np.pi * 4) + 0.02 * np.cos((xx - 0.18) * np.pi * 8)
    ax.plot(xx, yy, color=COLORS["blue_dark"], lw=2)
    ax.text(0.50, 0.11, "Критерий: минимум потерь электроэнергии", ha="center", va="center", fontsize=11, color=COLORS["ink"], fontweight="bold")
    add_caption(ax, "Оптимизация осуществляется с учётом нормативных ограничений и баланса нагрузки по зонам.")
    return fig


def fig_0216_auth_audit():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.16 — Механизмы аутентификации, авторизации и аудита интеллектуальной системы управления",
        "Контур управления доступом с разделением ролей и журналированием",
    )
    draw_grid_background(ax)
    cards = [
        (0.08, "Аутентификация", "многофакторная аутентификация (MFA)\nсертификаты\nединый вход (SSO)", COLORS["card2"], COLORS["blue"]),
        (0.39, "Авторизация", "управление доступом по ролям (RBAC)\nроли и права\nограничение функций", COLORS["card4"], COLORS["orange"]),
        (0.70, "Аудит", "журналы событий\nсистема корреляции событий безопасности (SIEM)\nкриптографическая фиксация", COLORS["card3"], COLORS["green"]),
    ]
    for x, title, subtitle, fc, ec in cards:
        add_shadow_box(ax, x, 0.30, 0.22, 0.32, f"{title}\n\n{subtitle}", facecolor=fc, edgecolor=ec, fontsize=10.6, weight="bold")
    add_arrow(ax, (0.30, 0.46), (0.39, 0.46), color=COLORS["blue"])
    add_arrow(ax, (0.61, 0.46), (0.70, 0.46), color=COLORS["orange"])
    add_tag(ax, 0.19, 0.72, "Пользователь")
    add_tag(ax, 0.50, 0.72, "Права доступа")
    add_tag(ax, 0.81, 0.72, "Журнал")
    add_caption(ax, "Механизмы предотвращают несанкционированные действия и сохраняют подотчётность операций.")
    return fig


def fig_0217_crypto_resilience():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.17 — Механизмы криптографической защиты и обеспечения отказоустойчивости",
        "Защита каналов, контроль целостности и аварийное резервирование",
    )
    draw_grid_background(ax)
    cards = [
        (0.08, "Шифрование каналов", "канальное шифрование (TLS 1.3)\nГОСТ 34.12-2018\nбезопасный трафик", COLORS["card2"], COLORS["blue"]),
        (0.39, "Контроль целостности", "хеш-суммы\nпроверка конфигураций\nверификация ПО", COLORS["card4"], COLORS["orange"]),
        (0.70, "Аварийное резервирование", "резервный режим (fallback)\nфиксация состояния\nрежим защиты", COLORS["card3"], COLORS["green"]),
    ]
    for x, title, subtitle, fc, ec in cards:
        add_shadow_box(ax, x, 0.30, 0.22, 0.32, f"{title}\n\n{subtitle}", facecolor=fc, edgecolor=ec, fontsize=10.2, weight="bold")
    add_arrow(ax, (0.30, 0.46), (0.39, 0.46), color=COLORS["blue"])
    add_arrow(ax, (0.61, 0.46), (0.70, 0.46), color=COLORS["orange"])
    ax.text(
        0.50,
        0.20,
        "время восстановления (RTO) ≤ 15 мин",
        ha="center",
        va="center",
        fontsize=12,
        color=COLORS["ink"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=COLORS["muted"], linewidth=1.2),
    )
    add_caption(ax, "Защита включает криптографию, проверку целостности и переключение на безопасный режим.")
    return fig


def fig_0218_testing():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.18 — Схема процедур тестирования компонентов интеллектуальной системы управления",
        "От модульных тестов к цифровому двойнику распределительной сети",
    )
    draw_grid_background(ax)
    cards = [
        (0.08, "Модульное тестирование", "модульные тесты\nсинтетические выборки\nпокрытие кода ≥ 85%", COLORS["card2"], COLORS["blue"]),
        (0.39, "Интеграционное тестирование", "промышленные шлюзы\nприкладной интерфейс (API)\nOPC UA / IEC 60870-5-104", COLORS["card4"], COLORS["orange"]),
        (0.70, "Системное тестирование", "цикл управления\nцифровой двойник\nштатные и аварийные режимы", COLORS["card3"], COLORS["green"]),
    ]
    for x, title, subtitle, fc, ec in cards:
        add_shadow_box(ax, x, 0.30, 0.22, 0.32, f"{title}\n\n{subtitle}", facecolor=fc, edgecolor=ec, fontsize=10.1, weight="bold")
    add_arrow(ax, (0.30, 0.46), (0.39, 0.46), color=COLORS["blue"])
    add_arrow(ax, (0.61, 0.46), (0.70, 0.46), color=COLORS["orange"])
    add_tag(ax, 0.50, 0.18, "Цифровой двойник")
    add_caption(ax, "Тестирование выполняется последовательно, чтобы исключить риск передачи некорректных команд в сеть.")
    return fig


def fig_0219_validation():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.19 — Схема валидации модулей машинного обучения и верификации экспертной системы",
        "Ретроспективная проверка, правила и стресс-тесты",
    )
    draw_grid_background(ax)
    cards = [
        (0.08, "Ретроспективная проверка машинного обучения (ML)", "скользящая ретроспективная проверка (Walk-Forward)\nR² > 0,85\nMAE / MAPE", COLORS["card2"], COLORS["blue"]),
        (0.39, "Верификация экспертной системы", "архив аварийных сценариев\nсравнение с действиями диспетчера", COLORS["card4"], COLORS["orange"]),
        (0.70, "Стресс-тестирование", "зашумлённые данные\nобрыв связи\nложные срабатывания", COLORS["card3"], COLORS["green"]),
    ]
    for x, title, subtitle, fc, ec in cards:
        add_shadow_box(ax, x, 0.30, 0.22, 0.32, f"{title}\n\n{subtitle}", facecolor=fc, edgecolor=ec, fontsize=9.9, weight="bold")
    add_arrow(ax, (0.30, 0.46), (0.39, 0.46), color=COLORS["blue"])
    add_arrow(ax, (0.61, 0.46), (0.70, 0.46), color=COLORS["orange"])
    add_caption(ax, "Валидация фиксирует корректность прогнозов, правил и устойчивость к аномальным данным.")
    return fig


def fig_0220_acceptance():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Рисунок 2.20 — Критерии приёмки и дорожная карта пилотного внедрения интеллектуальной системы управления",
        "Критерии готовности и этапы опытной эксплуатации",
    )
    draw_grid_background(ax)

    # Left: criteria cards
    criteria = [
        ("Функциональная готовность", "Полное соответствие ТЗ\nинтеграционные тесты пройдены\nнет критических ошибок", COLORS["card2"], COLORS["blue"]),
        ("Эксплуатационная надёжность", "Время отклика < 1 с\nвосстановление сессии ≤ 5 с\nбезопасный отказ", COLORS["card4"], COLORS["orange"]),
        ("Киберустойчивость", "ФЗ-187 и ФСТЭК № 239\nаутентификация, шифрование, аудит", COLORS["card3"], COLORS["green"]),
        ("Экономическая эффективность", "Снижение потерь ≥ 3%\nсокращение перерывов в питании\n6 месяцев пилота", "#F4F0FF", "#7C3AED"),
    ]
    for i, (title, subtitle, fc, ec) in enumerate(criteria):
        y = 0.72 - i * 0.15
        add_shadow_box(ax, 0.05, y, 0.34, 0.11, f"{title}\n{subtitle}", facecolor=fc, edgecolor=ec, fontsize=8.9, weight="bold")

    # Timeline right side
    ax.plot([0.46, 0.94], [0.38, 0.38], color=COLORS["slate"], lw=2.5)
    phases = [
        (0.48, "1–2 мес.", "Подготовительный этап", COLORS["blue"]),
        (0.62, "3–4 мес.", "Теневая эксплуатация", COLORS["orange"]),
        (0.76, "5–8 мес.", "Опытно-промышленная эксплуатация", COLORS["green"]),
        (0.90, "с 9 мес.", "Промышленная эксплуатация", "#7C3AED"),
    ]
    for x, period, title, color in phases:
        ax.add_patch(Circle((x, 0.38), 0.018, facecolor=color, edgecolor="white", linewidth=1.0, zorder=3))
        add_shadow_box(ax, x - 0.06, 0.48, 0.12, 0.12, f"{period}\n{title}", facecolor="white", edgecolor=color, fontsize=8.8, weight="bold")
        add_arrow(ax, (x, 0.40), (x, 0.48), color=color, rad=0.0)

    ax.text(
        0.70,
        0.18,
        "Переход: советчик → ограниченное автоматическое управление → промышленная эксплуатация",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["ink"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=COLORS["muted"], linewidth=1.2),
    )
    add_caption(ax, "Дорожная карта завершает цикл проектирования и задаёт формат безопасного пилота.")
    return fig


def fig_perimeter_architecture():
    fig, ax = setup_canvas(
        (13, 7.5),
        "Архитектура защищённого периметра интеллектуальной системы управления",
        "МЭК 62443: зоны, каналы (conduits) и демилитаризованная зона",
    )
    draw_grid_background(ax)

    perimeter = FancyBboxPatch(
        (0.06, 0.18),
        0.88,
        0.58,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        linewidth=2.0,
        edgecolor=COLORS["slate"],
        facecolor="white",
        zorder=1,
    )
    ax.add_patch(perimeter)
    ax.text(
        0.5,
        0.76,
        "Зоны и каналы (Zone & Conduit)",
        ha="center",
        va="center",
        fontsize=13,
        color=COLORS["blue_dark"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.08", facecolor="white", edgecolor=COLORS["blue"], linewidth=1.3),
    )

    add_shadow_box(
        ax,
        0.09,
        0.30,
        0.23,
        0.28,
        "Полевой уровень\n\nТерминальные устройства (RTU), РЗА, датчики\nлокальные шкафы\nфизический доступ",
        facecolor=COLORS["card2"],
        edgecolor=COLORS["blue"],
        fontsize=10.2,
        weight="bold",
    )
    add_shadow_box(
        ax,
        0.40,
        0.30,
        0.18,
        0.28,
        "Демилитаризованная зона (DMZ)\n\nмежсетевой экран нового поколения (NGFW) / глубокий анализ пакетов (DPI)\nпромышленные шлюзы\nконтроль трафика",
        facecolor=COLORS["card4"],
        edgecolor=COLORS["orange"],
        fontsize=10.1,
        weight="bold",
    )
    add_shadow_box(
        ax,
        0.67,
        0.30,
        0.24,
        0.28,
        "Корпоративный уровень\n\nдиспетчерский контур (SCADA), БД, система корреляции событий безопасности (SIEM)\nАРМ диспетчера\nаналитические сервера",
        facecolor=COLORS["card3"],
        edgecolor=COLORS["green"],
        fontsize=10.1,
        weight="bold",
    )

    add_arrow(ax, (0.32, 0.49), (0.40, 0.49), color=COLORS["blue"])
    add_arrow(ax, (0.58, 0.49), (0.67, 0.49), color=COLORS["green"])
    add_arrow(ax, (0.58, 0.40), (0.40, 0.40), color=COLORS["orange"], rad=-0.12)
    add_arrow(ax, (0.40, 0.57), (0.32, 0.57), color=COLORS["cyan"], rad=0.08)

    pill_specs = [
        (0.22, 0.20, "Шифрование и МФА"),
        (0.50, 0.20, "Роли и журнал"),
        (0.78, 0.20, "Контроль DMZ"),
    ]
    for x, y, text in pill_specs:
        add_tag(ax, x, y, text, color=COLORS["slate"])

    ax.text(
        0.50,
        0.12,
        "Прямая маршрутизация между зонами исключена: обмен проходит через демилитаризованную зону (DMZ) и контролируемые каналы связи.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["slate"],
        fontweight="bold",
    )
    add_caption(ax, "Защищённый периметр реализует эшелонированную защиту, разделение зон и контроль доверенных проводников данных.")
    return fig


FIGURE_BUILDERS = [
    ("fig_02_01_architecture.png", "Рисунок 2.1 — Трёхуровневая архитектура интеллектуальной системы управления", fig_021_architecture),
    ("fig_02_02_modules.png", "Рисунок 2.2 — Схема взаимодействия функциональных модулей интеллектуальной системы управления", fig_022_modules),
    ("fig_02_03_zone_map.png", "Рисунок 2.3 — Карта зон распределительной сети г. Ишимбая и прилегающих районов", fig_023_zone_map),
    ("fig_02_04_information_flow.png", "Рисунок 2.4 — Схема информационных потоков интеллектуальной системы управления", fig_024_information_flow),
    ("fig_02_05_sources.png", "Рисунок 2.5 — Источники данных интеллектуальной системы управления", fig_025_sources),
    ("fig_02_06_preprocessing.png", "Рисунок 2.6 — Конвейер предобработки данных интеллектуальной системы управления", fig_026_preprocessing),
    ("fig_02_07_features.png", "Рисунок 2.7 — Процесс формирования признаков интеллектуальной системы управления", fig_027_features),
    ("fig_02_08_training.png", "Рисунок 2.8 — Процедура обучения и настройки модели прогнозирования", fig_028_training),
    ("fig_02_09_metrics.png", "Рисунок 2.9 — Метрики оценки точности модели прогнозирования", fig_029_metrics),
    ("fig_02_10_sources_for_rules.png", "Рисунок 2.10 — Источники знаний для формирования базы правил экспертной системы", fig_0210_sources_for_rules),
    ("fig_02_11_knowledge_base.png", "Рисунок 2.11 — Структура базы знаний экспертной системы", fig_0211_knowledge_base),
    ("fig_02_12_hybrid_architecture.png", "Рисунок 2.12 — Схема взаимодействия модулей в гибридной архитектуре системы", fig_0212_hybrid_architecture),
    ("fig_02_13_fault_localization.png", "Рисунок 2.13 — Алгоритм выявления и локализации повреждений в распределительной сети", fig_0213_fault_localization),
    ("fig_02_14_isolation_reconfiguration.png", "Рисунок 2.14 — Алгоритм изоляции аварийного участка и реконфигурации сети", fig_0214_isolation_reconfiguration),
    ("fig_02_15_power_flow.png", "Рисунок 2.15 — Алгоритм оптимизации потоков мощности в распределительной сети", fig_0215_power_flow),
    ("fig_02_16_auth_audit.png", "Рисунок 2.16 — Механизмы аутентификации, авторизации и аудита интеллектуальной системы управления", fig_0216_auth_audit),
    ("fig_02_17_crypto_resilience.png", "Рисунок 2.17 — Механизмы криптографической защиты и обеспечения отказоустойчивости", fig_0217_crypto_resilience),
    ("fig_02_18_testing.png", "Рисунок 2.18 — Схема процедур тестирования компонентов интеллектуальной системы управления", fig_0218_testing),
    ("fig_02_19_validation.png", "Рисунок 2.19 — Схема валидации модулей машинного обучения и верификации экспертной системы", fig_0219_validation),
    ("fig_02_20_acceptance.png", "Рисунок 2.20 — Критерии приёмки и дорожная карта пилотного внедрения интеллектуальной системы управления", fig_0220_acceptance),
]


SUPPLEMENTAL_FIGURE_BUILDERS = [
    ("fig_02_perimeter_architecture.png", "Рисунок – Архитектура защищённого периметра интеллектуальной системы управления", fig_perimeter_architecture),
]


def main():
    parser = argparse.ArgumentParser(description="Генератор русскоязычных рисунков для НИР")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Каталог для сохранения PNG")
    parser.add_argument("--manifest", type=str, default=None, help="Путь к CSV-манифесту; по умолчанию внутри каталога output")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "figures_manifest.csv"

    manifest_rows = []
    for filename, caption, builder in FIGURE_BUILDERS:
        fig = builder()
        save_figure(fig, output_dir / filename)
        manifest_rows.append({"file": filename, "caption": caption, "type": "figure"})
        print(f"Сохранено: {output_dir / filename}")

    for filename, caption, builder in SUPPLEMENTAL_FIGURE_BUILDERS:
        fig = builder()
        save_figure(fig, output_dir / filename)
        manifest_rows.append({"file": filename, "caption": caption, "type": "supplemental"})
        print(f"Сохранено: {output_dir / filename}")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "caption", "type"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Манифест: {manifest_path}")
    print(f"Готово: {len(FIGURE_BUILDERS)} основных рисунков и {len(SUPPLEMENTAL_FIGURE_BUILDERS)} дополнительных")


if __name__ == "__main__":
    main()
