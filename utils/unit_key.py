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

UNIT_KEY_PARTS_RU: tuple[str, ...] = (
    "ID Проекта",
    "ID Корпус",
    "Секция",
    "Условный номер",
    "Этаж",
    "Площадь",
    "Комнатность",
)

# Текстовая метка студии в колонке комнатности (как в deals): в ключе и в таблице приводим к 1.
STUDIO_ROOM_COUNT_TOKEN = "ст"

# Заглушка «нет корпуса» в ``ID Корпус`` / ``building_id`` (сделки): сегмент в ключе пустой.
BUILDING_PLACEHOLDER_SUBSTR = "пустой корпус"


def _is_empty_placeholder_token(s: str) -> bool:
    t = s.strip()
    if not t:
        return True
    if t == "-":
        return True
    if t in ("\u2014", "\u2013"):  # — –
        return True
    return False


def _float_is_whole(x: float) -> bool:
    if math.isnan(x):
        return False
    r = round(x)
    return abs(x - r) < 1e-9


def _coerce_project_id_cell(v: object) -> object:
    """Целое для колонки project_id; неприводимое → pd.NA."""
    if pd.isna(v):
        return pd.NA
    if isinstance(v, bool):
        return pd.NA
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        if math.isnan(v):
            return pd.NA
        if _float_is_whole(float(v)):
            return int(round(v))
        return pd.NA
    s = str(v).strip()
    if _is_empty_placeholder_token(s):
        return pd.NA
    try:
        x = float(s.replace(",", "."))
    except ValueError:
        return pd.NA
    if math.isnan(x):
        return pd.NA
    if _float_is_whole(x):
        return int(round(x))
    return pd.NA


def _seg_project_id_key(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(int(v))


def _seg_room_count_key(v: object) -> str:
    """Сегмент комнатности в ключе: только целое (строка без дробной части); иначе пусто."""
    if pd.isna(v):
        return ""
    if isinstance(v, bool):
        return ""
    if isinstance(v, int):
        return str(int(v))
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if _float_is_whole(float(v)):
            return str(int(round(v)))
        return ""
    s = str(v).strip()
    if _is_empty_placeholder_token(s):
        return ""
    if s == STUDIO_ROOM_COUNT_TOKEN:
        return "1"
    try:
        x = float(s.replace(",", "."))
    except ValueError:
        return ""
    if math.isnan(x):
        return ""
    if _float_is_whole(x):
        return str(int(round(x)))
    return ""


def _seg_building_id(v: object) -> str:
    """Числовой корпус → строка целого; ``-`` / тире / пусто / «пустой корпус» → пусто; дробь ≈ целого → int."""
    if pd.isna(v):
        return ""
    if isinstance(v, bool):
        return ""
    if isinstance(v, int):
        return str(int(v))
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if _float_is_whole(float(v)):
            return str(int(round(v)))
        return ""
    s = str(v).strip()
    if _is_empty_placeholder_token(s):
        return ""
    if BUILDING_PLACEHOLDER_SUBSTR in s.casefold():
        return ""
    try:
        x = float(s.replace(",", "."))
    except ValueError:
        return ""
    if math.isnan(x):
        return ""
    if _float_is_whole(x):
        return str(int(round(x)))
    return ""


def add_unit_match_key(
    df: pd.DataFrame,
    *,
    renamed: bool = True,
    out_col: str = "unit_match_key",
    studio_typology_col: str | None = None,
    studio_typology_value: str = "Студия",
    studio_key_room_count: int = 1,
) -> pd.DataFrame:
    """Добавляет составной ключ для сопоставления одной и той же квартиры в price и deals.

    Общие сегменты (кроме площади и секции): NaN, пустая строка, пробелы, `-` и Unicode-тире → пусто (`@@` между соседними пустыми).
    Секция: без округления; дробные значения в ключе с точкой как разделителем; строки — strip и `,` → `.`.
    Площадь: после округления до 1 знака целые пишутся без дроби (`248`); дробная часть — одна цифра, разделитель запятая (`203,1`). Невалидное → пусто.

    ``project_id``: в датафрейме приводится к целому (nullable ``Int64``); в ключе — десятичная строка целого; нечисловое / нецелое → пустой сегмент.

    ``building_id``: ``-`` / тире / пусто и подстрока ``\"пустой корпус\"`` (``BUILDING_PLACEHOLDER_SUBSTR``) → пустой сегмент;
    значение с плавающей точкой, целое с точностью до округления, → целое в ключе; иначе нецелое дробное → пусто.

    Если ``room_count`` после strip равен ``\"ст\"`` (см. ``STUDIO_ROOM_COUNT_TOKEN``), в датафрейме
    значение заменяется на ``1``, в ключе сегмент комнатности — ``1``.

    Сегмент комнатности в ключе — **целое** (например ``4``, не ``4.0``); нечисловое / нецелое → пустой сегмент.

    Если задан ``studio_typology_col`` и значение совпадает с ``studio_typology_value`` (после strip),
    в ключе для сегмента комнатности подставляется ``studio_key_room_count`` (по умолчанию 1 для студий),
    **без изменения** колонки ``room_count`` в датафрейме (кроме случая ``\"ст\"`` выше).

    Parameters
    ----------
    df
        Таблица с исходными русскими именами колонок или уже переименованная.
    renamed
        True — ожидаются английские имена из пайплайна (`UNIT_KEY_PARTS_EN`).
        False — русские имена (`UNIT_KEY_PARTS_RU`).
    out_col
        Имя новой колонки.
    studio_typology_col
        Колонка типа квартиры (напр. ``unit_typology`` / ``Тип кв/ап``); None — правило студий не применяется.
    studio_typology_value
        Текст студии для сравнения.
    studio_key_room_count
        Значение сегмента комнатности в ключе для студий.
    """
    cols = UNIT_KEY_PARTS_EN if renamed else UNIT_KEY_PARTS_RU
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Нет колонок для ключа: {missing}")

    out = df.copy()
    room_col = cols[-1]
    st_room = out[room_col].map(
        lambda v: (not pd.isna(v)) and str(v).strip() == STUDIO_ROOM_COUNT_TOKEN
    )
    if st_room.any():
        out[room_col] = out[room_col].astype(object)
        out.loc[st_room, room_col] = 1

    proj_col = cols[0]
    out[proj_col] = out[proj_col].map(_coerce_project_id_cell).astype("Int64")

    def _seg_generic(v: object) -> str:
        if pd.isna(v):
            return ""
        s = str(v).strip()
        if _is_empty_placeholder_token(s):
            return ""
        return s

    def _seg_area(v: object) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, str):
            s = v.strip().replace(",", ".")
            if _is_empty_placeholder_token(s):
                return ""
            try:
                x = float(s)
            except ValueError:
                return ""
        else:
            try:
                x = float(v)
            except (TypeError, ValueError):
                return ""
        rounded = round(float(x), 1)
        if rounded == int(rounded):
            return str(int(rounded))
        return f"{rounded:.1f}".replace(".", ",")

    def _seg_section(v: object) -> str:
        """Секция: не округляем; дробное — в ключ с точкой (идентификатор, не число для расчётов)."""
        if pd.isna(v):
            return ""
        if isinstance(v, bool):
            return ""
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if pd.isna(v):
                return ""
            if v == int(v):
                return str(int(v))
            s = str(v)
            if "e" in s.lower():
                s = ("%f" % v).rstrip("0").rstrip(".")
            return s.replace(",", ".")
        s = str(v).strip()
        if _is_empty_placeholder_token(s):
            return ""
        return s.replace(",", ".")

    acc = out[proj_col].map(_seg_project_id_key)
    middle = cols[1:-1]
    for j, c in enumerate(middle):
        i = j + 1
        if i == 1:
            seg_fn = _seg_building_id
        elif i == 5:
            seg_fn = _seg_area
        elif i == 2:
            seg_fn = _seg_section
        else:
            seg_fn = _seg_generic
        acc = acc + "@" + out[c].map(seg_fn)

    room_seg = out[room_col].map(_seg_room_count_key)
    if studio_typology_col and studio_typology_col in out.columns:
        is_studio = out[studio_typology_col].map(
            lambda v: (not pd.isna(v)) and str(v).strip() == studio_typology_value
        )
        room_seg = room_seg.where(~is_studio, str(int(studio_key_room_count)))
    acc = acc + "@" + room_seg

    out[out_col] = acc
    return out


def _is_numeric_room(v: object) -> bool:
    """True, если значение можно трактовать как число для комнатности."""
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return not math.isnan(v)
    s = str(v).strip()
    if not s:
        return False
    try:
        float(s.replace(",", "."))
        return True
    except ValueError:
        return False


def _parse_pool_last_segment(pool_val: object) -> int | float | None:
    """Последний сегмент строки ``pool`` (после последнего ``@``), как число."""
    if pd.isna(pool_val):
        return None
    s = str(pool_val).strip()
    if "@" not in s:
        return None
    last = s.split("@")[-1].strip()
    if not last or last in ("-", "\u2014", "\u2013"):
        return None
    try:
        x = float(last.replace(",", "."))
    except ValueError:
        return None
    if math.isnan(x):
        return None
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return round(x, 1)


def fix_price_room_count_from_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Только price: если ``room_count`` не число, берём последний сегмент ``pool`` и пишем в ``room_count``."""
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
