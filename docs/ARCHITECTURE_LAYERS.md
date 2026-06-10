# Архитектура репозитория — 3 слоя

> Логическое разделение без физического перемещения файлов.

---

## Слой 1: core-ml (ML-ядро)

**Назначение:** эксперименты, модели, датасеты, обучение.

| Компонент | Путь |
|-----------|------|
| Baseline v1 (классификация) | `experiments/energy-ml-v1_baseline/` |
| Прогнозирование нагрузки v2 | `experiments/energy-ml-v2_forecast/` |
| Детекция аномалий v3 | `experiments/energy-ml-v3_anomaly/` |
| Экспертная система v4 | `experiments/energy-ml-v4_expert/` |
| Управление v5 | `experiments/energy-ml-v5_control/` |
| Датасеты | `experiments/.../datasets/` |
| CatBoost | `run_catboost_*.py` |
| SHAP-интерпретация | `run_shap_*.py` |
| Инженерия признаков | `build_features_*.py` |

## Слой 2: ops-control (управление)

**Назначение:** эксплуатация, safety, мониторинг, режим советчика.

| Компонент | Путь |
|-----------|------|
| Мониторинг телеметрии | `scripts/online_monitor.py` |
| Shadow control | `scripts/shadow_control.py` |
| Экспертная система (inference) | `scripts/expert_inference.py` |
| Streamlit dashboard | `dashboard/` |
| Safety архитектура | `SAFETY.md` |
| Benchmark моделей | `scripts/benchmark_models.py` |
| Конвейер данных | `scripts/build_dataset.py` |

## Слой 3: docs-showcase (документация и демо)

**Назначение:** НИР, отчёты, графики, навигация.

| Компонент | Путь |
|-----------|------|
| НИР-2 (отчёт) | `nir/Musin_NIR2_.docx` |
| Архитектура | `docs/ARCHITECTURE_LAYERS.md` |
| Roadmap | `docs/ROADMAP_10_STEPS.md` |
| Реестр датасетов | `experiments/DATASETS.md` |
| Словарь терминов | `docs/ML_GLOSSARY.md` |
| Графики для НИР | `plots/` |
| Навигация | `AI-NAVIGATOR.md`, `AI-INDEX.json` |

---

**Принцип:** все новые файлы создавать в соответствующем слое.
