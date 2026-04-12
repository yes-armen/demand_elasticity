"""One-off: Jaccard overlap of unit_match_key between price and deals + example rows."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.unit_key import add_unit_match_key, fix_price_room_count_from_pool

# Same as notebooks/preprocessing.ipynb (needed so renamed=True sees all key columns)
PRICE_RENAME_MAP: dict[str, str] = {
    "pool": "pool",
    "ID Проекта": "project_id",
    "ID Проекта для окружения": "env_project_id",
    "ID Корпус": "building_id",
    "Проект": "project_name",
    "Девелопер": "developer",
    "Класс проекта": "project_class",
    "Регион": "region",
    "Округ": "macro_district",
    "Район": "district",
    "Старт продаж": "sales_start",
    "Сдача К": "completion_k",
    "Ключи": "keys_status",
    "Тип помещения": "premises_type",
    "Секция": "section",
    "Этаж": "floor",
    "Отделка в отчет": "finish_in_report",
    "Тип кв/ап": "unit_typology",
    "Комнатность": "room_count",
    "Площадь": "area",
    "Цена": "price",
    "Цена без скидки": "price_list",
    "Старый бюджет": "budget_old",
    "Бюджет без скидки": "budget_list",
    "Бюджет": "budget",
    "Скидка, %": "discount_pct",
    "Наличие бюджета": "has_budget",
    "Изменение цены последнее": "last_price_change",
    "Дата файла": "file_date",
    "Начало/конец месяца": "month_span",
    "Источник": "source",
    "Период": "period",
    "lat": "lat",
    "lng": "lng",
    "Условный номер": "conditional_number",
    "Отделка К": "finish_tier",
    "Отделка текст": "finish_text",
    "Договор К": "contract_type_k",
    "Стадия К": "stage_k",
    "Экспозиция": "exposure",
}
DEALS_RENAME_MAP: dict[str, str] = {
    "key": "deal_row_key",
    "lat": "lat",
    "lng": "lng",
    "ID Проекта": "project_id",
    "ID Проекта для окруженияSales": "env_project_id_sales",
    "ID Корпус": "building_id",
    "Проект": "project_name",
    "Девелопер": "developer",
    "Класс проекта": "project_class",
    "Регион": "region",
    "Округ": "macro_district",
    "Район": "district",
    "Округ Направление": "macro_district_direction",
    "Комнатность": "room_count",
    "Этаж": "floor",
    "Условный номер": "conditional_number",
    "Площадь": "area",
    "Тип помещения": "premises_type",
    "Наличие бюджета": "has_budget",
    "Цена": "price",
    "Цена без скидки": "price_list",
    "Бюджет": "budget",
    "Бюджет без скидки": "budget_list",
    "Ключи": "keys_status",
    "Корпус": "building_name",
    "Покупатель ФЛ": "buyer_person",
    "Покупатель ЮЛ": "buyer_company",
    "Номер регистрации": "registration_number",
    "Дата регистрации": "registration_date",
    "Дата регистрации_": "registration_date_raw",
    "Залогодержатель/Банк": "pledge_holder_bank",
    "Длительность обременения": "pledge_duration_months",
    "Тип обременения": "pledge_type",
    "Отделка": "finish",
    "Дата подписания": "signing_date",
    "Дата ДДУ": "ddu_date",
    "Срок сдачи": "completion_due",
    "Стадия строительства": "construction_stage",
    "Ипотека": "mortgage",
    "Секция": "section",
    "Старт продаж": "sales_start",
    "Продавец ЮЛ": "seller_company",
    "Продавец ФЛ ID": "seller_person_id",
    "Описание помещения": "premises_description",
    "Опт": "is_wholesale",
    "Стадия строительства в дату ДДУ": "construction_stage_at_ddu",
    "Цена ДДУ": "price_ddu",
    "Тип сделки": "deal_type",
    "Дата регистрации модель": "registration_date_model",
    "Цена кв. м": "price_per_sqm",
    "ID дом.рф": "dom_rf_id",
    "Бюджет по ПД": "budget_pd",
    "Цена по ПД": "price_pd",
}


def rename_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    m = {c: PRICE_RENAME_MAP[c] for c in df.columns if c in PRICE_RENAME_MAP}
    return df.rename(columns=m)


def rename_deals_columns(df: pd.DataFrame) -> pd.DataFrame:
    m = {c: DEALS_RENAME_MAP[c] for c in df.columns if c in DEALS_RENAME_MAP}
    return df.rename(columns=m)


def preprocess_price(df: pd.DataFrame) -> pd.DataFrame:
    x = rename_price_columns(df)
    x = fix_price_room_count_from_pool(x)
    return add_unit_match_key(
        x,
        renamed=True,
        studio_typology_col="unit_typology",
        studio_typology_value="Студия",
        studio_key_room_count=1,
    )


def preprocess_deals(df: pd.DataFrame) -> pd.DataFrame:
    return add_unit_match_key(rename_deals_columns(df), renamed=True)


def main() -> None:
    data_dir = REPO_ROOT / "data"
    N = 400_000  # per price file; reduce if OOM

    price_slices = {
        "2025_01_05": "price_dataflat_2025_01_05.csv",
        "2024_01_06": "price_dataflat_2024_01_06.csv",
    }
    parts = []
    for label, name in price_slices.items():
        p = data_dir / name
        if not p.is_file():
            continue
        part = pd.read_csv(p, nrows=N, low_memory=False)
        part = part.assign(price_slice=label)
        parts.append(part)
    if not parts:
        raise SystemExit("No price CSVs found")
    price_raw = pd.concat(parts, ignore_index=True)

    deals_path = data_dir / "deals_dataflat_2020_2025.csv"
    deals_raw = pd.read_csv(deals_path, nrows=N * 2, low_memory=False)

    price_pp = preprocess_price(price_raw)
    deals_pp = preprocess_deals(deals_raw)

    kp = set(price_pp["unit_match_key"].dropna().astype(str).unique())
    kd = set(deals_pp["unit_match_key"].dropna().astype(str).unique())
    inter = kp & kd
    union = kp | kd
    ratio = len(inter) / len(union) if union else float("nan")

    print("=== Сэмпл ===")
    print(f"price rows: {len(price_pp)}, unique keys: {len(kp)}")
    print(f"deals rows: {len(deals_pp)}, unique keys: {len(kd)}")
    print(f"|intersection|: {len(inter)}, |union|: {len(union)}")
    print(f"Jaccard |∩|/|∪| = {ratio:.6f}")
    print()

    cols_p = ["pool", "unit_match_key", "project_id", "building_id", "section", "conditional_number", "floor", "area", "room_count", "price_slice"]
    cols_p = [c for c in cols_p if c in price_pp.columns]
    cols_d = ["unit_match_key", "project_id", "building_id", "section", "conditional_number", "floor", "area", "room_count"]
    cols_d = [c for c in cols_d if c in deals_pp.columns]

    print("--- Примеры: ключ есть в ОБОИХ (pool из прайса, та же unit_match_key в сделках) ---")
    for k in list(sorted(inter))[:5]:
        rp = price_pp.loc[price_pp["unit_match_key"].astype(str) == k, cols_p].iloc[0]
        rd = deals_pp.loc[deals_pp["unit_match_key"].astype(str) == k, cols_d].iloc[0]
        print(f"\nunit_match_key = {k!r}")
        print("  price:", rp.to_dict())
        print("  deals:", rd.to_dict())

    only_p = kp - kd
    only_d = kd - kp
    print("\n--- Примеры: ключ только в ПРАЙСЕ (нет такой строки в сделках в этом сэмпле) ---")
    for k in list(sorted(only_p))[:5]:
        rp = price_pp.loc[price_pp["unit_match_key"].astype(str) == k, cols_p].iloc[0]
        print(f"\nunit_match_key = {k!r}")
        print("  price:", rp.to_dict())

    print("\n--- Примеры: ключ только в СДЕЛКАХ (нет в прайсе в этом сэмпле) ---")
    for k in list(sorted(only_d))[:5]:
        rd = deals_pp.loc[deals_pp["unit_match_key"].astype(str) == k, cols_d].iloc[0]
        print(f"\nunit_match_key = {k!r}")
        print("  deals:", rd.to_dict())

    # Near-miss: same project, building, conditional_number but different full key (parameter drift)
    print("\n--- Пары (project_id, building_id, conditional_number) с разными unit_match_key в price vs deals ---")
    gcols = ["project_id", "building_id", "conditional_number"]
    if all(c in price_pp.columns for c in gcols) and all(c in deals_pp.columns for c in gcols):
        pp = price_pp.dropna(subset=gcols).copy()
        dd = deals_pp.dropna(subset=gcols).copy()
        pk = pp.groupby(gcols, dropna=False)["unit_match_key"].agg(lambda s: set(s.astype(str).unique()))
        dk = dd.groupby(gcols, dropna=False)["unit_match_key"].agg(lambda s: set(s.astype(str).unique()))
        common_idx = pk.index.intersection(dk.index)
        drift = []
        for idx in common_idx:
            sp, sd = pk.loc[idx], dk.loc[idx]
            if sp == sd:
                continue
            inter_keys = sp & sd
            if inter_keys:
                continue
            drift.append((idx, sp, sd))
            if len(drift) >= 8:
                break
        for (proj, bid, cn), sp, sd in drift[:5]:
            print(f"\nproject={proj}, building={bid}, conditional_number={cn!r}")
            print("  price keys (sample):", list(sp)[:3])
            print("  deals keys (sample):", list(sd)[:3])


if __name__ == "__main__":
    main()
