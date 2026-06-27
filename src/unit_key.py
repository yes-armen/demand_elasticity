"""Сборка составного ключа unit_match_key для сопоставления строк price / deals."""

from __future__ import annotations

import math

import pandas as pd

UNIT_KEY_PARTS_EN: tuple[str, ...] = (
    "project_id",
    "building_id",
    "section",
    "conditional_number",
    "floor",
    "area",
    "room_count",
)

# Оставлен для обратной совместимости импортов; внутри модуля не используется.
UNIT_KEY_PARTS_RU: tuple[str, ...] = (
    "ID Проекта",
    "ID Корпус",
    "Секция",
    "Условный номер",
    "Этаж",
    "Площадь",
    "Комнатность",
)

# Метка студии в колонке комнатности (deals): в ключе и в таблице приводим к 1.
STUDIO_ROOM_COUNT_TOKEN = "ст"

# Подстрока «нет корпуса» в building_id (сделки): сегмент в ключе пустой.
BUILDING_PLACEHOLDER_SUBSTR = "пустой корпус"

_DASH_CHARS: frozenset[str] = frozenset({"-", "—", "–"})  # -, —, –

# Текстовые значения раздела «Секция» в сделках, означающие отсутствие секции.
_SECTION_EMPTY_TOKENS: frozenset[str] = frozenset({"без секции"})


def _is_empty(v: object) -> bool:
    """True для None/NaN/NA, пустых строк и dash-символов."""
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except TypeError:
        pass
    t = str(v).strip()
    return not t or t in _DASH_CHARS


def _to_whole_int(v: object) -> int | None:
    """Число (int/float/str) → целое, если оно целое с точностью 1e-9. Иначе None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return None
        r = round(v)
        return r if abs(v - r) < 1e-9 else None
    s = str(v).strip().replace(",", ".")
    try:
        x = float(s)
    except ValueError:
        return None
    if math.isnan(x):
        return None
    r = round(x)
    return r if abs(x - r) < 1e-9 else None


# ---------- сегментные функции ----------

def _seg_project_id(v: object) -> str:
    # К моменту вызова колонка уже приведена к Int64, поэтому v — int или pd.NA.
    if _is_empty(v):
        return ""
    return str(int(v))


def _seg_building_id(v: object) -> str:
    """Числовой корпус → целое; тире/пусто/«пустой корпус» → ''."""
    if _is_empty(v):
        return ""
    if BUILDING_PLACEHOLDER_SUBSTR in str(v).casefold():
        return ""
    n = _to_whole_int(v)
    return str(n) if n is not None else ""


def _seg_section(v: object) -> str:
    """Секция — идентификатор: strip + запятая→точка; тире/пусто/«без секции» → ''."""
    if _is_empty(v):
        return ""
    if isinstance(v, bool):
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        r = round(v)
        return str(r) if abs(v - r) < 1e-9 else str(v)
    s = str(v).strip()
    if s.casefold() in _SECTION_EMPTY_TOKENS:
        return ""
    return s.replace(",", ".")


def _seg_generic(v: object) -> str:
    """conditional_number, floor: strip; тире/пусто → ''."""
    if _is_empty(v):
        return ""
    return str(v).strip()


def _seg_area(v: object) -> str:
    """Площадь: округление до 1 знака, запятая как разделитель, целые без «,0»."""
    if _is_empty(v):
        return ""
    raw = str(v).strip().replace(",", ".") if isinstance(v, str) else v
    try:
        x = float(raw)
    except (TypeError, ValueError):
        return ""
    if math.isnan(x):
        return ""
    rounded = round(x, 1)
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.1f}".replace(".", ",")


def _seg_room_count(v: object) -> str:
    """Комнатность: только целое; 'ст' → '1'; нечисловое/дробное/пусто → ''."""
    if _is_empty(v) or isinstance(v, bool):
        return ""
    if str(v).strip() == STUDIO_ROOM_COUNT_TOKEN:
        return "1"
    n = _to_whole_int(v)
    return str(n) if n is not None else ""


# Индекс поля в UNIT_KEY_PARTS_EN → функция сегмента.
_SEG_FN = (
    _seg_project_id,   # 0: project_id
    _seg_building_id,  # 1: building_id
    _seg_section,      # 2: section
    _seg_generic,      # 3: conditional_number
    _seg_generic,      # 4: floor
    _seg_area,         # 5: area
    _seg_room_count,   # 6: room_count
)


def add_unit_match_key(
    df: pd.DataFrame,
    *,
    renamed: bool = True,  # noqa: ARG001 — параметр сохранён для обратной совместимости
    out_col: str = "unit_match_key",
    studio_typology_col: str | None = None,
    studio_typology_value: str = "Студия",
    studio_key_room_count: int = 1,
) -> pd.DataFrame:
    """Составной ключ project_id@building_id@section@conditional_number@floor@area@room_count
    для сопоставления квартиры в price и deals.

    Пустые/прочерки → пустой сегмент; площадь округляется до 0.1, room_count — к целому
    ('ст'→1). Детали нормализации сегментов — в _seg_* функциях.
    """
    cols = UNIT_KEY_PARTS_EN
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Нет колонок для ключа: {missing}")

    out = df.copy()
    room_col = cols[-1]

    # Нормализация 'ст' → 1 в таблице.
    is_studio_token = out[room_col].map(
        lambda v: not _is_empty(v) and str(v).strip() == STUDIO_ROOM_COUNT_TOKEN
    )
    if is_studio_token.any():
        out[room_col] = out[room_col].astype(object)
        out.loc[is_studio_token, room_col] = 1

    # Приведение project_id к nullable Int64.
    proj_col = cols[0]
    out[proj_col] = out[proj_col].map(
        lambda v: pd.NA if _is_empty(v) else _to_whole_int(v)
    ).astype("Int64")

    # Сборка ключа. Явный astype(object) защищает от Int64/float64 dtype
    # при пустых DataFrame, когда pandas не вызывает fn ни разу и не выводит тип.
    parts = [out[c].map(fn).astype(object) for c, fn in zip(cols, _SEG_FN)]

    if studio_typology_col and studio_typology_col in out.columns:
        is_studio = out[studio_typology_col].map(
            lambda v: not _is_empty(v) and str(v).strip() == studio_typology_value
        )
        parts[-1] = parts[-1].where(~is_studio, str(int(studio_key_room_count)))

    key = parts[0].astype(str)
    for seg in parts[1:]:
        key = key + "@" + seg.astype(str)
    out[out_col] = key

    return out


# ---------- вспомогательные функции для fix_price_room_count_from_pool ----------

def _is_numeric_room(v: object) -> bool:
    """True если значение можно трактовать как число для комнатности."""
    if _is_empty(v) or isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return not (isinstance(v, float) and math.isnan(v))
    s = str(v).strip()
    try:
        float(s.replace(",", "."))
        return True
    except ValueError:
        return False


def _parse_pool_last_segment(pool_val: object) -> int | float | None:
    """Последний сегмент строки pool (после последнего @) как число."""
    if _is_empty(pool_val):
        return None
    s = str(pool_val).strip()
    if "@" not in s:
        return None
    last = s.split("@")[-1].strip()
    if not last or last in _DASH_CHARS:
        return None
    try:
        x = float(last.replace(",", "."))
    except ValueError:
        return None
    if math.isnan(x):
        return None
    r = round(x)
    return r if abs(x - r) < 1e-9 else round(x, 1)


def fix_price_room_count_from_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Только price: если room_count не число, подставляет последний сегмент pool."""
    if "pool" not in df.columns or "room_count" not in df.columns:
        return df
    out = df.copy()
    bad = ~out["room_count"].map(_is_numeric_room)
    if not bad.any():
        return out
    out["room_count"] = out["room_count"].astype(object)
    from_pool = out.loc[bad, "pool"].map(_parse_pool_last_segment)
    ok = from_pool.notna()
    if ok.any():
        out.loc[from_pool[ok].index, "room_count"] = from_pool[ok]
    return out
