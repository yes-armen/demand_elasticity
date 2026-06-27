"""Надёжность оценки эластичности по срезу.

Модель глобальная → надёжность = устойчивость ОЦЕНКИ на выбранном срезе, не «обучение сегмента».
Критично: при малом q_base (ожидаемая база спроса в м²) эластичность мусорная (деление на ~ноль).
"""
from __future__ import annotations

TIERS = {
    "high":   dict(active_lots=200, q_base=200.0),
    "medium": dict(active_lots=50,  q_base=50.0),
}


def reliability_tier(active_lots: int, q_base: float) -> str:
    for name, thr in TIERS.items():
        if active_lots >= thr["active_lots"] and q_base >= thr["q_base"]:
            return name
    return "low"


def tier_message(tier: str, active_lots: int, q_base: float) -> str:
    if tier == "high":
        return f"Высокая надёжность (лотов {active_lots:,}, q_base {q_base:,.0f} м²)."
    if tier == "medium":
        return f"Средняя надёжность (лотов {active_lots:,}, q_base {q_base:,.0f} м²)."
    return (f"⚠️ Низкая надёжность: лотов {active_lots:,}, q_base {q_base:,.0f} м². "
            "Эластичность ненадёжна — расширьте срез (укрупните сегмент).")
