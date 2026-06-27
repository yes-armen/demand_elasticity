"""Расчётный движок дашборда эластичности (поверх калиброванной модели 05).

Семантика сценария — V4 «пропорциональная реценка» + per-lot deviation: цена лота (price И
price_no_discount) двигается на Δ, доля скидки (discount_pct) сохраняется, медиана сегмента
фиксирована (конкуренты не реагируют) → двигается rel_price. Это чистая ценовая эластичность
без эндогенного «скидка=дистресс». Метрики:
  N = Σ p·calib            — ожидаемое число продаж (с калибр-поправкой),
  Q = Σ p·area·calib       — ожидаемый спрос, м²,
  R = Σ p·area·price·(1+Δ)·calib — ожидаемая выручка,
  E = ΔQ/(Q·Δ)             — area-взвешенная эластичность.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import MODEL_DATASET_PARQUET, CALIBRATED_MODEL_PATH, TARGET_COL, PRICE_COL
from src.modeling import CalibratedModel  # noqa: F401 — нужен для joblib.load
from src.elasticity import apply_price_shock as _apply_price_shock

PCT_GRID = (-0.15, -0.125, -0.10, -0.075, -0.05, -0.025, 0.0,
            0.025, 0.05, 0.075, 0.10, 0.125, 0.15)
# Поправка абсолютных величин (N, Q, R) считается ДИНАМИЧЕСКИ под загруженную модель —
# см. compute_calibration(). На эластичность и проценты НЕ влияет (константа сокращается в отношении).
HORIZON_CUT = pd.Timestamp("2025-04-30")  # последний labelable срез (для actual-кривой)
TEST_FROM = pd.Timestamp("2024-12-01")    # начало тестового окна модели (для селектора срезов)

# колонки-селекторы дашборда
FILTER_COLS = [
    "region", "macro_district", "district", "project_class",
    "project_name", "developer", "room_count", "premises_type",
    "stage_k", "finish_tier",
]


def load_model() -> CalibratedModel:
    return joblib.load(CALIBRATED_MODEL_PATH)


def load_data() -> pd.DataFrame:
    return pd.read_parquet(MODEL_DATASET_PARQUET)


def active_slice(df: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    """Активные (не пост-продажные) лоты выбранного среза-месяца."""
    return df[(df["is_sold_past"] == 0) & (df["file_date"] == month)].copy()


def compute_calibration(model, df: pd.DataFrame) -> tuple[dict, float]:
    """Динамическая поправка абсолютных величин под ЗАГРУЖЕННУЮ модель.

    Для каждого labelable среза (есть полное 30-дневное окно) считается
        factor = Σ(факт продаж) / Σ(p_cal по всем активным лотам среза).
    Это привязывает базовый прогноз продаж к фактическому уровню на срезе:
    factor < 1 — модель переоценивает продажи (типичный случай: часть лотов
    получает повышенную P, но уходит «обрывом» без сделки), factor > 1 —
    недооценивает. Возвращает (factor по месяцам, глобальный fallback для
    нелейблабельных срезов). На эластичность/проценты не влияет (сокращается).
    """
    lab = df[(df["is_sold_past"] == 0) & (df["file_date"] <= HORIZON_CUT)].copy()
    lab["_p"] = predict_p(model, lab[model.features])
    g = lab.groupby(lab["file_date"].dt.to_period("M")).agg(
        sy=(TARGET_COL, "sum"), sp=("_p", "sum"))
    by_month = {k: float(sy / sp) for k, sy, sp in zip(g.index, g["sy"], g["sp"]) if sp > 0}
    global_factor = float(g["sy"].sum() / g["sp"].sum())
    return by_month, global_factor


def apply_price_shock(X: pd.DataFrame, pct: float) -> pd.DataFrame:
    """Шок цены по УРОВНЮ: пропорциональная реценка (price и price_no_discount × (1+Δ)),
    доля скидки сохраняется, медиана сегмента фикс → двигается rel_price. Признаки ИЗМЕНЕНИЯ
    цены (price_change_*) и коллинеарные относительные (rel_price_district, price_gap_to_p10)
    заморожены, иначе эластичность завышается. Делегирует src.elasticity (один источник истины)."""
    return _apply_price_shock(X, pct, price_col=PRICE_COL)


def predict_p(model: CalibratedModel, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[model.features])[:, 1]


def scenario_curve(model, data, pct_grid=PCT_GRID, calib=1.0) -> pd.DataFrame:
    """Кривая агрегатов по сетке Δ для одного среза/сегмента."""
    X = data[model.features]
    area = data["area"].to_numpy()
    price = data[PRICE_COL].to_numpy()
    rows = []
    for pct in pct_grid:
        p = predict_p(model, apply_price_shock(X, pct))
        rows.append(dict(
            pct=pct,
            n_sales=p.sum() * calib,
            q_sqm=(p * area).sum() * calib,
            revenue=(p * area * price * (1 + pct)).sum() * calib,
        ))
    m = pd.DataFrame(rows)
    q0 = m.loc[m["pct"] == 0.0, "q_sqm"].iloc[0]
    r0 = m.loc[m["pct"] == 0.0, "revenue"].iloc[0]
    m["q_chg_pct"] = 100 * (m["q_sqm"] / q0 - 1)
    m["rev_chg_pct"] = 100 * (m["revenue"] / r0 - 1)
    return m


def elasticity_pm5(curve: pd.DataFrame) -> float:
    """Area-взвешенная эластичность центральной разностью ±5%."""
    q0 = curve.loc[curve["pct"] == 0.0, "q_sqm"].iloc[0]
    qm = curve.loc[curve["pct"] == -0.05, "q_sqm"].iloc[0]
    qp = curve.loc[curve["pct"] == 0.05, "q_sqm"].iloc[0]
    return (qp - qm) / (q0 * 0.10) if q0 else np.nan


def revenue_gate(curve: pd.DataFrame, min_uplift_pct: float = 0.5) -> dict:
    """Health-gate на рекомендацию цены по выручке.

    Рекомендация выдаётся, если у выручки есть ИНТЕРЬЕРНЫЙ максимум (не на краю диапазона)
    с нетривиальным приростом (>= min_uplift_pct). Немонотонность спроса НЕ вето: плоский
    неэластичный хвост — норма (асимметрия спроса), это лишь информационный флаг.
    """
    c = curve.sort_values("pct").reset_index(drop=True)
    rev = c["revenue"].to_numpy()
    q = c["q_sqm"].to_numpy()
    i = int(rev.argmax())
    interior = 0 < i < len(c) - 1
    base_rev = c.loc[c["pct"] == 0.0, "revenue"].iloc[0]
    uplift = 100.0 * (rev[i] / base_rev - 1) if base_rev else 0.0
    demand_monotone = bool(np.all(np.diff(q) <= 1e-9))   # допускаем плато (Δq=0)
    return dict(ok=bool(interior and uplift >= min_uplift_pct),
                opt=float(c["pct"].iloc[i]), uplift=uplift,
                interior=interior, demand_monotone=demand_monotone)


def demand_shape(model, data, n_bins=10):
    """Факт. доля продаж vs модель p по децилю rel_price (асимметрия). None если срез не labelable."""
    d = data.dropna(subset=["rel_price"]).copy()
    if d.empty:
        return None
    d["p"] = predict_p(model, d[model.features])
    labelable = (d["file_date"] <= HORIZON_CUT).all()
    d["rp_bin"] = pd.qcut(d["rel_price"], n_bins, duplicates="drop")
    agg = {"rel_price": ("rel_price", "mean"), "model": ("p", "mean"), "n": ("p", "size")}
    if labelable:
        agg["actual"] = (TARGET_COL, "mean")
    return d.groupby("rp_bin", observed=True).agg(**agg).reset_index(drop=True), labelable
