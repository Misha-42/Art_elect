"""
Экспертная система управления Ишимбайскими РЭС.
Раздел 2.4 НИР-2: продукционные правила, база знаний, прямой вывод.

Компоненты:
- KnowledgeBase — загрузка и хранение правил
- InferenceEngine — прямой логический вывод (Forward Chaining)
- ExpertSystem — интеграция с CatBoost

Использование:
    python code/expert_system.py --data ../../energy-ml-v1_baseline/datasets/telemetry_v3_synthetic.csv
"""

import sys; sys.stdout.reconfigure(encoding='utf-8')
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

RANDOM_STATE = 42
RULE_PATH = Path(__file__).parent.parent / 'knowledge_base' / 'rules.json'


class KnowledgeBase:
    """База знаний экспертной системы (раздел 2.4.2 НИР-2)."""

    def __init__(self, path: Path = RULE_PATH):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.version = data['metadata']['version']
        self.priority_levels = data['priority_levels']
        self.rules = data['rules']
        print(f'\n📚 База знаний v{self.version}')
        print(f'   Правил: {len(self.rules)}')
        for level, desc in self.priority_levels.items():
            count = sum(1 for r in self.rules if r['priority'] == level)
            print(f'   {level}: {count} правил — {desc}')

    def get_rules_by_priority(self, priority: str) -> list:
        return [r for r in self.rules if r['priority'] == priority]

    def get_rules_by_mode(self, mode: int) -> list:
        return [r for r in self.rules if r['mode'] == mode]

    def get_critical_rules(self) -> list:
        return self.get_rules_by_priority('critical')


class TelemetryFacts:
    """Факты о текущем состоянии сети (из телеметрии)."""

    def __init__(self, row: pd.Series, df_history: pd.DataFrame = None):
        self.row = row
        self.timestamp = row.get('timestamp', datetime.now())
        self.features = {}

        # Базовые телеметрические параметры
        for col in ['U_10kV', 'I_feeder_1', 'P_feeder_1', 'Q_feeder_1',
                     'f_grid', 'cos_fi']:
            if col in row:
                self.features[col] = float(row[col])

        # Производные признаки (если есть история)
        if df_history is not None:
            self._compute_derived(df_history)

    def _compute_derived(self, df: pd.DataFrame):
        """Вычисление производных признаков (тренды, диффы)."""
        if len(df) < 2: return

        # Diff только числовых колонок (исключая timestamp)
        numeric_cols = df.select_dtypes(include=['number']).columns
        diff = df[numeric_cols].diff().iloc[-1]
        self.features['U_diff'] = diff.get('U_10kV', 0)
        self.features['I_diff'] = diff.get('I_feeder_1', 0)
        self.features['P_diff'] = diff.get('P_feeder_1', 0)

        # Флаги аварийных состояний
        self.features['I_alarm'] = 1 if self.features.get('I_feeder_1', 0) > 120 else 0
        self.features['U_alarm'] = 1 if self.features.get('U_10kV', 0) < 6.0 else 0
        self.features['f_alarm'] = 1 if self.features.get('f_grid', 50) <= 49.0 else 0

        # Тренды (изменение за последние 10 шагов = 1 час)
        if len(df) >= 10:
            recent = df.tail(10)
            self.features['P_trend'] = (recent['P_feeder_1'].iloc[-1] -
                                         recent['P_feeder_1'].iloc[0]) / 10
            self.features['cos_fi_trend'] = (recent['cos_fi'].iloc[-1] -
                                              recent['cos_fi'].iloc[0]) / 10
            self.features['P_feeder_1_trend'] = self.features['P_trend']
            self.features['I_diff_duration'] = 0  # упрощённо

        # Потери данных (упрощённо — 0%)
        self.features['data_loss'] = 0
        self.features['time_since_alarm'] = 60  # нет аварии

        # Календарь
        ts = pd.to_datetime(self.timestamp) if not isinstance(self.timestamp, pd.Timestamp) else self.timestamp
        self.features['hour'] = ts.hour
        self.features['is_weekend'] = 1 if ts.dayofweek >= 5 else 0

    def satisfies(self, condition: dict) -> bool:
        """Проверка условия правила."""
        # Извлекаем имя поля из ключа (может быть field_op, ищем в features)
        cond_key = list(condition.keys())[0]
        cond = condition[cond_key]
        value = self.features.get(cond_key)

        if value is None:
            value = self.row.get(cond_key)
        if value is None:
            return False

        op = cond['op']
        target = cond['value']

        try:
            if op == 'gt': return float(value) > float(target)
            elif op == 'ge': return float(value) >= float(target)
            elif op == 'lt': return float(value) < float(target)
            elif op == 'le': return float(value) <= float(target)
            elif op == 'eq': return float(value) == float(target)
            elif op == 'in':
                if isinstance(target, list):
                    return int(value) in [int(x) for x in target]
                return False
            else: return False
        except (TypeError, ValueError):
            return False


class InferenceEngine:
    """Механизм прямого логического вывода (Forward Chaining).
    Раздел 2.4.2 НИР-2: 4 такта: Matching → Conflict Resolution → Acting → Logging.
    """

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.history_log = []
        self.fired_rules = []

    def _matching(self, facts) -> List[Dict]:
        """Сопоставление: поиск правил, чьи условия истинны."""
        matched = []
        for rule in self.kb.rules:
            conditions = rule['conditions']
            all_true = True
            for cond_key, cond_val in conditions.items():
                single_cond = {cond_key: cond_val}
                if not facts.satisfies(single_cond):
                    all_true = False
                    break
            if all_true:
                matched.append(rule)
        return matched

    def infer(self, facts) -> List[Dict]:
        """Полный цикл вывода."""
        # === Такт 1: Matching (сопоставление) ===
        matched = self._matching(facts)

        # === Такт 2: Conflict Resolution (выбор) ===
        selected = self._conflict_resolution(matched)

        # === Такт 3: Acting (исполнение) ===
        actions = self._acting(selected, facts)

        # === Такт 4: Logging (фиксация) ===
        self._logging(selected)

        return actions

    def _matching(self, facts: TelemetryFacts) -> List[Dict]:
        """Сопоставление: поиск правил, чьи условия истинны."""
        matched = []
        for rule in self.kb.rules:
            conditions = rule['conditions']
            # Все условия ИЛИ? Используем И (AND)
            all_true = all(facts.satisfies({k: v}) for k, v in conditions.items())
            if all_true:
                matched.append(rule)
        return matched

    def _conflict_resolution(self, matched: List[Dict]) -> List[Dict]:
        """Разрешение конфликтов: возвращаем все сработавшие правила,
        отсортированные по приоритету."""
        if not matched:
            return []

        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        sorted_rules = sorted(matched,
            key=lambda r: priority_order.get(r['priority'], 99))
        return sorted_rules

    def _acting(self, rules: List[Dict], facts: TelemetryFacts) -> List[Dict]:
        """Формирование управляющих воздействий."""
        actions = []
        for rule in rules:
            actions.append({
                'rule_id': rule['id'],
                'rule_name': rule['name'],
                'priority': rule['priority'],
                'mode': rule['mode'],
                'actions': rule['actions'],
                'description': rule['description'],
                'timestamp': datetime.now().isoformat(),
            })
            self.fired_rules.append(rule)
        return actions

    def _logging(self, rules: List[Dict]):
        """Фиксация сработавших правил в журнал."""
        for rule in rules:
            entry = {
                'rule_id': rule['id'],
                'rule_name': rule['name'],
                'priority': rule['priority'],
                'timestamp': datetime.now().isoformat(),
            }
            self.history_log.append(entry)

    def get_stats(self) -> Dict:
        """Статистика работы ЭС."""
        return {
            'total_fired': len(self.fired_rules),
            'total_history': len(self.history_log),
            'critical_fired': sum(1 for r in self.fired_rules
                                  if r['priority'] == 'critical'),
            'by_mode': {
                0: sum(1 for r in self.fired_rules if r['mode'] == 0),
                1: sum(1 for r in self.fired_rules if r['mode'] == 1),
                2: sum(1 for r in self.fired_rules if r['mode'] == 2),
                3: sum(1 for r in self.fired_rules if r['mode'] == 3),
            }
        }


class CatBoostExpertHybrid:
    """Гибридная архитектура: CatBoost + Экспертная система.
    Раздел 2.4.3 НИР-2.
    """

    def __init__(self, expert_system: InferenceEngine, catboost_preds: np.ndarray = None):
        self.expert = expert_system
        self.catboost_preds = catboost_preds

    def decide(self, facts: TelemetryFacts,
               ml_prediction: Optional[int] = None) -> Dict:
        """Гибридное принятие решений.

        1. CatBoost даёт прогноз режима (ML)
        2. Экспертная система проверяет по правилам
        3. Если правила critical — они имеют приоритет
        4. Иначе — решение ML
        """
        expert_actions = self.expert.infer(facts)

        # Находим самое приоритетное правило
        critical_rules = [a for a in expert_actions
                          if a['priority'] == 'critical']
        high_rules = [a for a in expert_actions
                      if a['priority'] == 'high']
        medium_rules = [a for a in expert_actions
                        if a['priority'] == 'medium']

        if critical_rules:
            # Приоритет безопасности: критические правила блокируют ML
            decision = 'ЭС (critical)'
            final_actions = critical_rules
            final_mode = 2  # аварийный
        elif high_rules:
            decision = 'ЭС (high)'
            final_actions = high_rules
            final_mode = 1  # предупредительный
        elif medium_rules:
            decision = 'ЭС (medium)'
            final_actions = medium_rules
            final_mode = 3  # пост-аварийный (по умолчанию для medium)
            # Корректировка mode по первому правилу
            if medium_rules:
                final_mode = medium_rules[0]['mode']
        elif expert_actions:
            # Остальные (low)
            decision = 'ЭС (low)'
            final_actions = expert_actions
            final_mode = expert_actions[0]['mode']
        elif ml_prediction is not None:
            # Нет правил — решение CatBoost (или норма)
            decision = 'CatBoost'
            final_actions = [{
                'rule_id': 'ML_PRED',
                'rule_name': 'Прогноз градиентного бустинга',
                'priority': 'low',
                'mode': int(ml_prediction),
                'actions': [f'Прогнозируемый режим: {int(ml_prediction)}'],
                'description': 'Решение на основе ML',
            }]
            final_mode = int(ml_prediction)
        else:
            # Fallback: если ничего не сработало — нормальный режим
            decision = 'норма (fallback)'
            final_actions = []
            final_mode = 0

        return {
            'decision': decision,
            'mode': final_mode,
            'actions': final_actions,
            'n_fired': len(expert_actions),
            'timestamp': datetime.now().isoformat(),
            'stats': self.expert.get_stats(),
        }

    def validate(self, true_modes: np.ndarray,
                 expert_decisions: List[Dict]) -> Dict:
        """Валидация гибридной системы."""
        correct = 0
        total = len(true_modes)

        for i, dec in enumerate(expert_decisions):
            if i < total and dec['mode'] == true_modes[i]:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
        }
