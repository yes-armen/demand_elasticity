"""Пути проекта и глобальные константы."""
from pathlib import Path

REPO_ROOT     = Path(__file__).resolve().parents[1]
RAW_DIR       = REPO_ROOT / "data" / "raw"
INTERIM_DIR   = REPO_ROOT / "data" / "interim"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR    = REPO_ROOT / "models"
REPORTS_DIR   = REPO_ROOT / "reports"

FILTERED_PARQUET     = INTERIM_DIR / "price_filtered.parquet"
FILTERED_CSV         = INTERIM_DIR / "price_filtered.csv"   # legacy
JOINED_LEFT_PARQUET  = INTERIM_DIR / "price_deals_left_unit_match_key.parquet"
JOINED_INNER_PARQUET = INTERIM_DIR / "price_deals_inner_unit_match_key.parquet"
JOINED_LEFT_CSV      = INTERIM_DIR / "price_deals_left_unit_match_key.csv"   # legacy
JOINED_INNER_CSV     = INTERIM_DIR / "price_deals_inner_unit_match_key.csv"  # legacy
JOINED_CSV           = JOINED_LEFT_PARQUET
# EDA-датасеты после финального отбора колонок (02c) — на них крутится EDA
EDA_LEFT_PARQUET  = PROCESSED_DIR / "eda_left.parquet"
EDA_INNER_PARQUET = PROCESSED_DIR / "eda_inner.parquet"
# Модельный датасет после фиче-инжиниринга (04_features)
MODEL_DATASET_PARQUET = PROCESSED_DIR / "model_dataset.parquet"
# Макро-ряды (ключевая ставка ЦБ РФ + инфляция)
MACRO_XLSX = INTERIM_DIR / "interest_rate_and_inflation.xlsx"
# Геофичи (OSM) — считаются на уровне project_id в 04a_geo_features
OSM_CACHE_DIR        = INTERIM_DIR / "osm_cache"
GEO_FEATURES_PARQUET = PROCESSED_DIR / "geo_features.parquet"

OBS_PARQUET      = PROCESSED_DIR / "observations.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "observations_features.parquet"
OBS_CSV      = PROCESSED_DIR / "observations.csv"       # legacy
FEATURES_CSV = PROCESSED_DIR / "observations_features.csv"  # legacy
MODEL_PATH           = MODELS_DIR / "catboost_sale_prob.cbm"
CALIBRATED_MODEL_PATH = MODELS_DIR / "catboost_sale_prob_calibrated.pkl"
ELAST_CSV   = PROCESSED_DIR / "elasticity_by_segment.csv"

# Горизонты таргетов (дни после file_date до registration_date_raw)
SALE_WINDOW_DAYS = 30          # is_sold_next_month — основной горизонт модели (30 дней)
SALE_WINDOW_QUARTER_DAYS = 90  # is_sold_next_quarter — альтернативный квартальный горизонт

# Таргеты (строятся в notebooks/02c_column_selection.ipynb)
TARGET_COL = "is_sold_next_month"            # основной таргет модели (30 дней)
TARGET_COL_QUARTER = "is_sold_next_quarter"  # альтернативный квартальный таргет (90 дней)

# Ценовые колонки: joined (interim) vs model_dataset (processed)
PRICE_COL_JOINED = "price_price"
AREA_COL_JOINED  = "area_price"
PRICE_COL    = "price"        # ₽/м² в model_dataset
AREA_COL     = "area"
DISCOUNT_COL = "discount_pct"
