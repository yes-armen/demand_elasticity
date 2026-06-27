"""Обучение и инференс модели вероятности продажи."""
from __future__ import annotations

import numpy as np
import pandas as pd


class CalibratedModel:
    """CatBoost + изотоническая калибровка. predict_proba → [P0, P1]."""

    def __init__(self, model, calibrator, features, cat_features):
        self.model = model
        self.calibrator = calibrator
        self.features = features
        self.cat_features = cat_features

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features].copy()
        for c in self.cat_features:
            X[c] = X[c].fillna("__NA__").astype(str)
        raw = self.model.predict_proba(X)[:, 1]
        cal = np.clip(self.calibrator.predict(raw), 0, 1)
        return np.column_stack([1 - cal, cal])
