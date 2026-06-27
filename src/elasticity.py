"""Локальная ценовая эластичность из модели вероятности продажи.

Шок цены пересчитывает только признаки УРОВНЯ цены (price, log_price,
price_no_discount, discount_abs, rel_price); сегментные/районные агрегаты и
история конкурентов фиксированы (частичное равновесие). Признаки изменения цены
не трогаем — иначе эластичность завышается.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Признаки уровня цены лота (двигаются пропорционально цене). rel_price_district и
# price_gap_to_p10 не двигаем — коллинеарны с rel_price, иначе двойной счёт.
_PROPORTIONAL = ("price", "price_no_discount", "rel_price")


def apply_price_shock(
    X: pd.DataFrame,
    pct: float,
    *,
    price_col: str = "price",
    seg_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Шок цены ×(1+pct): двигает уровень цены, замораживает изменения цены.

    Двигаются price, price_no_discount, log_price, discount_abs, rel_price.
    Заморожены признаки изменения цены (price_change_*), коллинеарные относительные
    (rel_price_district, price_gap_to_p10) и сегментные/макро-агрегаты.
    seg_cols (опц.) — пересчитать price_rank_seg на одном срезе.
    """
    if not pct:
        return X
    out = X.copy()
    f = 1.0 + pct
    base_price = out[price_col].to_numpy(copy=True)

    for c in _PROPORTIONAL:
        if c in out.columns:
            out[c] = out[c] * f
    if "log_price" in out.columns:
        out["log_price"] = np.log1p(out[price_col])
    if "discount_abs" in out.columns and "price_no_discount" in out.columns:
        out["discount_abs"] = out["price_no_discount"] - out[price_col]
    # discount_pct и признаки изменения цены не трогаем (см. docstring).

    if "price_rank_seg" in out.columns and seg_cols and all(c in out.columns for c in seg_cols):
        shocked = out[price_col].to_numpy()
        ranks = np.empty(len(out), dtype="float64")
        for pos in out.groupby(list(seg_cols), observed=True).indices.values():
            s = np.sort(base_price[pos])
            ranks[pos] = np.searchsorted(s, shocked[pos], side="right") / len(s)
        out["price_rank_seg"] = ranks
    return out


def point_elasticity(
    model,
    X: pd.DataFrame,
    *,
    price_col: str = "price",
    delta: float = 0.01,
    seg_cols: list[str] | None = None,
) -> pd.Series:
    """ε = d log P(sale) / d log цена (численная производная при шаге delta)."""
    X_up = apply_price_shock(X, delta, price_col=price_col, seg_cols=seg_cols)

    p0 = np.asarray(model.predict_proba(X))[:, 1]
    p1 = np.asarray(model.predict_proba(X_up))[:, 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        elast = np.where(p0 > 1e-8, (p1 - p0) / (p0 * delta), np.nan)

    return pd.Series(elast, index=X.index, name="elasticity")
