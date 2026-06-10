# BACKLOG.md · Art_elect — Сводная таблица метрик и результатов

> Актуальные результаты экспериментов и моделей.
> Обновляется по мере выполнения шагов ROADMAP.

---

## Сводная таблица метрик

| Эксперимент | Модель | Метрика | Значение | Статус |
|-------------|--------|---------|:--------:|:------:|
| baseline-v1 | Logistic Regression | BA | — | ⏳ |
| baseline-v1 | Random Forest | BA | — | ⏳ |
| forecast-v2 | CatBoost (1ч) | R² | — | ⏳ |
| forecast-v2 | CatBoost (24ч) | R² | — | ⏳ |
| forecast-v2 | CatBoost (168ч) | R² | — | ⏳ |
| anomaly-v3 | CatBoost | PR-AUC | — | ⏳ |
| expert-v4 | Expert System | precision | — | ⏳ |
| control-v5 | Hybrid | accuracy | — | ⏳ |

---

## Ближайшие задачи

- [ ] Сбор телеметрии Ишимбайских РЭС
- [ ] Baseline v1: классификация режимов
- [ ] Инвентаризация SCADA-тегов
