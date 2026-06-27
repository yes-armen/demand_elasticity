# Моделирование эластичности спроса на квартиры с использованием методов машинного обучения

Проект по моделированию ценовой эластичности спроса на квартиры первичного рынка (Москва, Новая Москва, Московская область).

Подход: CatBoost обучается предсказывать вероятность продажи лота в течение 30 дней (таргет `is_sold_next_month`, горизонт 30 дней), затем численная производная вероятности по цене даёт точечную эластичность. Короткий горизонт выбран потому, что на нём цена прайс-среза ближе к фактической цене сделки — это важнее для оценки ценовой чувствительности, чем больший охват сделок у квартального горизонта.

## Структура репозитория

```
demand_elasticity/
├── data/
│   ├── raw/                       # исходные CSV/XLSX (кладутся локально)
│   │   ├── price_dataflat_*.csv   # прайс-срезы объявлений
│   │   └── deals_dataflat_*.csv   # реестр сделок
│   ├── interim/                   # промежуточные parquet (фильтрация, JOIN price↔deals, кэш OSM)
│   └── processed/                 # eda_left/inner, geo_features, model_dataset, elasticity_*/scenarios_*
├── notebooks/                     # пайплайн 00–08
│   ├── 00_data_audit.ipynb
│   ├── 01_filtering.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 02b_representativeness.ipynb
│   ├── 02c_column_selection.ipynb
│   ├── 03_eda_macro.ipynb
│   ├── 03_eda_projects.ipynb
│   ├── 04a_geo_features.ipynb
│   ├── 04_features.ipynb
│   ├── 05_training.ipynb
│   ├── 06_elasticity.ipynb
│   ├── 07_scenarios.ipynb
│   └── 08_segments.ipynb
├── src/
│   ├── config.py                  # пути и константы
│   ├── unit_key.py                # сборка unit_match_key (price ↔ deals)
│   ├── modeling.py                # CalibratedModel (CatBoost + изотоническая калибровка)
│   ├── geo_features.py            # геофичи окружения ЖК из OSM (OSMnx)
│   └── elasticity.py              # единый движок шока цены + point/segment elasticity
├── models/
│   ├── catboost_sale_prob.cbm
│   ├── catboost_sale_prob_calibrated.pkl
│   └── fi_shap_summary.json       # важность по группам + SHAP финальной модели
├── dashboard/                     # Streamlit-дашборд (app / engine / reliability)
├── reports/
│   └── figures/                   # графики EDA, обучения, эластичности, сценариев, экраны дашборда
├── docs/                          # методология, формулы, model card, coursework.tex / report*.tex
├── DATASET_STRUCTURE.md
├── requirements.txt
└── README.md
```

Фичеинжиниринг, таргеты, обучение и калибровка реализованы в ноутбуках; `src/` содержит переиспользуемые утилиты.

## Пайплайн

```
raw CSVs
  → [01] фильтрация (month_span='begin', нормализация типологии, has_budget=1)
  → [02] rename RU→EN, unit_key.py, LEFT/INNER JOIN price ← deals
  → [02b] репрезентативность inner vs left
  → [02c] отбор колонок, таргеты is_sold* → eda_left/inner.parquet
  → [03] EDA по проектам (+ row_number, unit_status) и макро (ставка, инфляция)
  → [04a] геофичи окружения ЖК по координатам (OSM) → geo_features.parquet (по project_id)
  → [04] фичи (цена, сегмент, лаги, макро, гео) → model_dataset.parquet
  → [05] поблочно (base / +market / +geo); финал = base+market+geo (дефолт CatBoost, depth 6; CV/тюнинг
         отключены) + isotonic calibration; важность по группам + SHAP → P(продажа за 30 дней) [.cbm + CalibratedModel]
  → [06] эластичность (шок УРОВНЯ цены, единый src.elasticity) → d log P / d log цена ≈ −0.7 → elasticity_by_segment.csv
  → [07] сценарии ±15% цены (движок дашборда)
  → [08] динамика и сегментный разбор эластичности
```

## Таргеты

Строятся в `02c_column_selection` (`delta = registration_date_raw − file_date`; в left-join метка `1` ставится только на последнем срезе лота через `is_last`):

| Колонка | Условие | Роль |
|---|---|---|
| `is_sold_next_month` | 0 < delta ≤ 30 | **основной таргет модели** (горизонт 30 дней) |
| `is_sold_next_quarter` | 0 < delta ≤ 90 | альтернативный квартальный горизонт |
| `is_sold` | delta ≥ 0 на последнем срезе | продажа когда-либо после среза |
| `is_sold_past` | delta < 0 | продан до среза; исключается из риск-сета |

## Модули `src/`

| Модуль | Назначение |
|---|---|
| `config.py` | Пути (`RAW` / `INTERIM` / `PROCESSED` / `MODELS`), `TARGET_COL` (=`is_sold_next_month`), `TARGET_COL_QUARTER`, горизонты `SALE_WINDOW_DAYS=30` / `SALE_WINDOW_QUARTER_DAYS=90` |
| `unit_key.py` | Составной ключ `project_id@building_id@section@conditional_number@floor@area@room_count`; нормализация «ст»→1, площади до 1 знака, «пустой корпус»→'' |
| `modeling.py` | `CalibratedModel` — CatBoost + изотоническая калибровка для инференса |
| `geo_features.py` | геофичи окружения ЖК из OSM (OSMnx): `compute_geo_features`, `feature_columns`; кэш слоёв в `data/interim/osm_cache/` |
| `elasticity.py` | `apply_price_shock` — **единый движок шока цены** (двигает уровень: `price`, `log_price`, `price_no_discount`, `discount_abs`, `rel_price`; замораживает `price_change_*` и коллинеарные `rel_price_district`/`price_gap_to_p10`). Один источник истины для `06`/`07`/`08` и дашборда. Плюс `point_elasticity`, `segment_elasticity` |

`PRICE_RENAME_MAP` / `DEALS_RENAME_MAP` — в `notebooks/02_preprocessing.ipynb`.

## Фичи модели

Обучение (`05_training`): rolling-snapshot — в риск-сет входят все месячные срезы лота (вес лота `1/√(n снапшотов)`). Right-censoring: `file_date ≤ 2025-04-30` (max registration 2025-05-30 − 30 дней). Temporal split с purge-gap = горизонт (30д, зазоры по 1 мес): train < 2024-05 · val 2024-06…08 · test 2024-10…2025-04-30. Итого **89 числовых + 13 категориальных = 102 признака** (лота, рыночные, гео).

**Числовые лота/цены/времени (26):** `area`, `floor`, `price`, `log_price`, `price_no_discount`, `discount_pct`, `discount_abs`, `rel_price`, `seg_median_price`, `price_rank_seg`, `seg_count`, `price_change_pct_1`, `price_change_from_first`, `discount_change_1`, `days_since_prev`, `n_price_changes`, `exposure`, `row_number`, `days_since_sales_start`, `months_to_completion`, `file_month`, `file_year`, `history_truncated`, `key_rate`, `inflation`, `real_rate`

**Рыночные (28, состояние и динамика сегмента/конкуренции — `04` секция 6b):** `seg_price_iqr`, `seg_price_cv`, `price_gap_to_p10`, `n_active_project`, `n_active_building`, `n_active_district`, `competitor_density`, `room_share_project`, `seg_count_chg_1m`, `seg_count_chg_3m`, `seg_median_price_chg_1m`, `seg_median_price_chg_3m`, `seg_median_price_chg_6m`, `share_new_supply`, `seg_sold_prev_3m`, `seg_absorption_3m`, `months_of_inventory`, `seg_median_exposure`, `rel_exposure`, `seg_median_age`, `share_stale_seg`, `key_rate_chg_3m`, `key_rate_chg_6m`, `real_rate_chg_3m`, `inflation_chg_3m`, `rel_price_district`, `seg_premium`, `dev_price_premium`

**Категориальные (13):** `project_class`, `project_name`, `region`, `macro_district`, `district`, `room_count`, `premises_type`, `finish_tier`, `finish_in_report`, `unit_typology`, `stage_k`, `contract_type_k`, `developer`

**Геофичи окружения ЖК (35, OSM — `04a`, мердж по `project_id`):** расстояния до ближайшего объекта (`dist_to_metro_m`, `dist_to_school_m`, `dist_to_water_m`, `dist_to_industrial_m`, `dist_to_cemetery_m`, `dist_to_power_line_m`, `dist_to_major_road_m`, `dist_to_railway_m`); по кольцам 500/1000/2000 м — число школ (`schools_count_*`), доли площади зелени/воды/промзон/кладбищ (`green_share_*`, `water_share_*`, `industrial_share_*`, `cemetery_share_*`), длины дорог/ж.д./ЛЭП (`major_roads_length_*`, `railway_length_*`, `power_lines_length_*`), плотность соседних ЖК (`nearby_complexes_count_*`). Считаются один раз на проект; кэш OSM-слоёв в `data/interim/osm_cache/`.

Подробнее о колонках и правилах join — в `DATASET_STRUCTURE.md`.

## Установка

Требуется Python 3.11+.

```bash
pip install -r requirements.txt
```

Сырые данные (`data/raw/`) кладутся локально в `data/raw/` перед запуском ноутбуков.
</content>
</invoke>
