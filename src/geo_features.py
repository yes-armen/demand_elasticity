"""Геофичи окружения ЖК из OpenStreetMap (OSMnx).

Считаются на уровне проекта (один центроид на project_id, ~750 точек — дешевле,
чем на 4.6M строк); геометрия в метрической проекции EPSG:32637. Слои OSM качаются
потайлово и кэшируются (при пустом слое — мягкая деградация). Точка входа —
compute_geo_features(eda_df) → DataFrame по project_id.
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- CRS -------------------------------------------------------------------
WGS84 = "EPSG:4326"
METRIC_CRS = "EPSG:32637"  # UTM zone 37N — метры, подходит для Москвы и МО

# --- параметры фич ----------------------------------------------------------
RADIUS_M = 5000          # буфер вокруг общего bbox проектов для скачивания OSM
RINGS = [500, 1000, 2000]  # радиусы (м) для count / share / length фич

# Сторона тайла (м): крупный регион одним запросом Overpass рвёт связь — режем на тайлы.
TILE_SIZE_M = 10_000
# Тяжёлые площадные/линейные слои режем мельче; точечные (metro/schools) — крупно.
LAYER_TILE_M = {
    "green": 5_000,
    "water": 5_000,
    "industrial": 5_000,
    "major_roads": 5_000,
    "railway": 5_000,
    "power_lines": 5_000,
}


def _layer_tile_m(layer_name: str) -> int:
    """Размер тайла для слоя: override для тяжёлых, иначе общий TILE_SIZE_M."""
    return LAYER_TILE_M.get(layer_name, TILE_SIZE_M)

# Таймаут одного Overpass-запроса (с).
REQUESTS_TIMEOUT = 300
# Зеркала Overpass — ротируем между ретраями (берётся первое ответившее).
OVERPASS_ENDPOINTS = [
    "https://maps.mail.ru/osm/tools/overpass/api",
    "https://overpass.osm.ch/api",
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
    "https://overpass.kumi.systems/api",
]
# Ретраи при обрыве соединения (с нарастающей паузой).
OVERPASS_MAX_RETRIES = 8

# Границы правдоподобия координат (Москва + МО + НМ); вне — мусор/нули.
LAT_BOUNDS = (53.5, 58.5)
LNG_BOUNDS = (34.0, 41.5)

# --- OSM-теги по слоям -------------------------------------------------------
OSM_TAGS = {
    "metro": {
        "railway": ["station", "halt", "subway_entrance"],
        "station": ["subway"],
        "subway": True,
        "public_transport": ["station", "stop_position"],
    },
    "schools": {
        "amenity": ["school", "kindergarten", "college", "university"],
    },
    "green": {
        "leisure": ["park", "garden", "nature_reserve"],
        "landuse": ["forest", "grass", "meadow", "recreation_ground"],
        "natural": ["wood", "grassland", "scrub"],
    },
    "water": {
        "natural": ["water"],
        "waterway": True,
        "landuse": ["reservoir", "basin"],
    },
    "industrial": {
        "landuse": ["industrial", "brownfield", "garages"],
        "man_made": ["works", "wastewater_plant"],
        "power": ["plant", "substation"],
    },
    "cemetery": {
        "landuse": ["cemetery"],
        "amenity": ["grave_yard", "crematorium"],
    },
    "power_lines": {
        "power": ["line", "minor_line", "tower", "pole", "substation"],
    },
    "major_roads": {
        "highway": [
            "motorway", "trunk", "primary", "secondary",
            "motorway_link", "trunk_link", "primary_link", "secondary_link",
        ],
    },
    "railway": {
        "railway": ["rail", "light_rail", "subway", "tram"],
    },
}

# Какую метрику считать по каждому слою:
#   nearest — расстояние до ближайшего объекта (расстояния всегда информативны)
#   count   — число точечных POI в кольцах
#   share   — доля площади кольца под полигонами
#   length  — суммарная длина линейных объектов в кольце
NEAREST_LAYERS = {
    "metro": "dist_to_metro_m",
    "schools": "dist_to_school_m",
    "water": "dist_to_water_m",
    "industrial": "dist_to_industrial_m",
    "cemetery": "dist_to_cemetery_m",
    "power_lines": "dist_to_power_line_m",
    "major_roads": "dist_to_major_road_m",
    "railway": "dist_to_railway_m",
}
COUNT_LAYERS = {"schools": "schools_count"}
SHARE_LAYERS = {
    "green": "green_share",
    "water": "water_share",
    "industrial": "industrial_share",
    "cemetery": "cemetery_share",
}
LENGTH_LAYERS = {
    "major_roads": "major_roads_length",
    "railway": "railway_length",
    "power_lines": "power_lines_length",
}


# Прокси для OSM: overpass доступен через локальный прокси — ниже находим живой
# порт и выставляем его в окружение процесса.
_PROXY_READY = False


# Нейтральные хосты для проверки прокси (не overpass — он может быть забанен).
_PROXY_PROBE_URLS = ("https://www.openstreetmap.org/copyright", "https://example.com")


def _proxy_works(url: str) -> bool:
    """Умеет ли прокси вообще выйти в интернет (по нейтральному хосту)."""
    import requests
    for probe in _PROXY_PROBE_URLS:
        try:
            r = requests.get(
                probe, proxies={"http": url, "https": url},
                timeout=8, headers={"Accept": "*/*"},
            )
            if r.status_code < 500:
                return True
        except Exception:
            continue
    return False


def _local_proxy_ports() -> list:
    """LISTEN-порты локальных процессов avito-ai (через lsof, без sudo)."""
    import re, subprocess
    try:
        out = subprocess.run(
            ["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return []
    ports = []
    for ln in out.splitlines():
        if "avito" in ln.lower():
            m = re.search(r"127\.0\.0\.1:(\d+)", ln)
            if m and int(m.group(1)) not in ports:
                ports.append(int(m.group(1)))
    return ports


def ensure_proxy(force: bool = False):
    """Гарантировать рабочий прокси для OSM в окружении процесса.

    Если в env уже задан рабочий прокси — оставляем его. Иначе ищем живой порт
    avito-ai (lsof) и выставляем `HTTP(S)_PROXY`. Идемпотентно (проверка один
    раз; `force=True` — перепроверить). Возвращает URL прокси или None.
    """
    import os
    global _PROXY_READY
    if _PROXY_READY and not force:
        return os.environ.get("HTTPS_PROXY")

    cur = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if cur and _proxy_works(cur):
        _PROXY_READY = True
        return cur

    for port in _local_proxy_ports():
        url = f"http://127.0.0.1:{port}"
        if _proxy_works(url):
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                os.environ[k] = url
            print(f"[geo] OSM через прокси {url}")
            _PROXY_READY = True
            return url

    print("[geo] ⚠ рабочий прокси не найден — overpass, скорее всего, недоступен "
          "(прямой выход закрыт). Проверь, что сессия avito-ai активна.")
    _PROXY_READY = True
    return None


def _osmnx():
    """Ленивый импорт osmnx + гарантия рабочего прокси для OSM-запросов."""
    ensure_proxy()
    import osmnx as ox

    from src.config import INTERIM_DIR

    ox.settings.use_cache = True
    # держим http-кэш osmnx под data/ (gitignore), а не в cwd
    ox.settings.cache_folder = str(INTERIM_DIR / "osm_http_cache")
    ox.settings.log_console = False
    ox.settings.requests_timeout = REQUESTS_TIMEOUT
    # Пропускаем pre-flight статус Overpass: в части окружений эндпоинт /api/status
    # отвечает 406 и осmnx зависает в ожидании паузы. Между слоями всё равно
    # делаем sleep(1), а на 429 у осmnx есть встроенный retry с Retry-After.
    ox.settings.overpass_rate_limit = False
    return ox


# ---------------------------------------------------------------------------
# 1. Точки проектов
# ---------------------------------------------------------------------------
def project_points(eda_df: pd.DataFrame):
    """Уникальные проекты → один репрезентативный центроид (median lat/lng).

    Возвращает GeoDataFrame в METRIC_CRS с колонками
    `project_id`, `project_name`, `lat`, `lng`, `geometry`.
    Координаты вне границ правдоподобия и нули отбрасываются до агрегации.
    """
    import geopandas as gpd

    cols = ["project_id", "lat", "lng"]
    if "project_name" in eda_df.columns:
        cols.append("project_name")
    df = eda_df[cols].copy()

    ok = (
        df["lat"].between(*LAT_BOUNDS)
        & df["lng"].between(*LNG_BOUNDS)
    )
    df = df[ok]

    agg = {"lat": "median", "lng": "median"}
    if "project_name" in df.columns:
        agg["project_name"] = "first"
    pts = df.groupby("project_id", as_index=False).agg(agg)

    gdf = gpd.GeoDataFrame(
        pts,
        geometry=gpd.points_from_xy(pts["lng"], pts["lat"]),
        crs=WGS84,
    )
    return gdf.to_crs(METRIC_CRS)


def tile_polygons(points_metric, tile_m: int = TILE_SIZE_M, buffer_m: int = RADIUS_M):
    """Режет область проектов на сетку тайлов по `tile_m` метров.

    Область — **объединение буферов** вокруг точек проектов (`buffer_m`), а не
    convex hull: это исключает пустые промежутки между удалёнными ЖК в МО,
    которые в фичи всё равно не входят (метрики считаются в радиусе ≤ `buffer_m`
    от проекта). Накрываем область регулярной сеткой и оставляем только тайлы,
    реально пересекающие её. Каждый тайл — отдельный маленький запрос Overpass.
    Возвращает список полигонов-тайлов в WGS84.
    """
    import geopandas as gpd
    from shapely.geometry import box

    region = points_metric.geometry.buffer(buffer_m).union_all()
    minx, miny, maxx, maxy = region.bounds

    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cell = box(x, y, min(x + tile_m, maxx), min(y + tile_m, maxy))
            if cell.intersects(region):  # пропускаем тайлы вне области проектов
                cells.append(cell)
            y += tile_m
        x += tile_m

    return list(gpd.GeoSeries(cells, crs=METRIC_CRS).to_crs(WGS84))


# ---------------------------------------------------------------------------
# 2. Загрузка слоёв OSM (потайлово, с кэшем в GeoPackage)
# ---------------------------------------------------------------------------
class _TileFailed(Exception):
    """Тайл не скачался за все ретраи — слой соберётся без него, докачается потом."""


def _is_no_data_error(err: Exception) -> bool:
    """Отличает «в тайле нет объектов» (это норма) от сетевого обрыва."""
    try:
        from osmnx._errors import InsufficientResponseError
        if isinstance(err, InsufficientResponseError):
            return True
    except Exception:
        pass
    msg = str(err).lower()
    return "no data" in msg or "did not return" in msg or "no matching" in msg


def _fetch_tile(layer_name: str, tags: dict, tile_wgs, idx: int, tile_dir: Path):
    """Скачивает один тайл слоя с резюм-кэшем.

    Возвращает GeoDataFrame в METRIC_CRS (может быть пустым). Кэширует результат
    тайла: `<idx>.gpkg` для непустых, `<idx>.empty` — маркер пустого тайла.
    Повторный запуск пропускает уже скачанные тайлы. При исчерпании ретраев на
    сетевой ошибке — поднимает `_TileFailed` (тайл НЕ кэшируется → докачается при
    следующем прогоне); вызывающий код пропускает такой тайл, не роняя слой.
    """
    import geopandas as gpd

    gpkg = tile_dir / f"{idx:04d}.gpkg"
    empty = tile_dir / f"{idx:04d}.empty"
    if gpkg.exists():
        try:
            return gpd.read_file(gpkg).to_crs(METRIC_CRS)
        except Exception:
            gpkg.unlink(missing_ok=True)  # битый кэш тайла — перекачаем
    if empty.exists():
        return gpd.GeoDataFrame(geometry=[], crs=METRIC_CRS)

    ox = _osmnx()
    last_err = None
    neps = len(OVERPASS_ENDPOINTS)
    for attempt in range(1, OVERPASS_MAX_RETRIES + 1):
        ep = OVERPASS_ENDPOINTS[(attempt - 1) % neps]  # ротация зеркал по ретраям
        ox.settings.overpass_url = ep
        host = ep.split("//")[-1].split("/")[0]
        try:
            gdf = ox.features_from_polygon(tile_wgs, tags)
            break
        except Exception as e:
            if _is_no_data_error(e):           # пустой тайл — это не ошибка
                empty.touch()
                return gpd.GeoDataFrame(geometry=[], crs=METRIC_CRS)
            last_err = e
            print(f"    тайл {idx}: попытка {attempt}/{OVERPASS_MAX_RETRIES} "
                  f"[{host}] не удалась: {str(e)[:70]}")
            time.sleep(min(30, 3 * attempt))  # нарастающая пауза перед сменой зеркала
    else:
        # тайл не кэшируем (ни gpkg, ни empty) → докачается при следующем прогоне
        raise _TileFailed(f"тайл {idx}: не скачался за {OVERPASS_MAX_RETRIES} попыток ({last_err})")

    gdf = gdf.reset_index()
    gdf = gdf[~gdf.geometry.isna()].copy()
    if gdf.empty:
        empty.touch()
        return gpd.GeoDataFrame(geometry=[], crs=METRIC_CRS)
    gdf = gdf.to_crs(METRIC_CRS)
    gdf["geometry"] = gdf.geometry.make_valid()
    out = gdf[["geometry"]].copy()  # OSM-атрибуты не нужны и ломают запись gpkg
    try:
        out.to_file(gpkg, driver="GPKG")
    except Exception as e:
        print(f"    тайл {idx}: не удалось записать кэш: {e}")
    time.sleep(1)
    return out


def fetch_osm_layer(layer_name: str, tags: dict, tiles_wgs, cache_dir: Path):
    """Тянет слой OSM потайлово, склеивает и дедуплицирует. Кэширует слой целиком.

    Объекты на границах тайлов попадают в несколько запросов → дедуп по WKB
    геометрии. Готовый слой пишется в `<layer>.gpkg`, после чего резюм-кэш
    тайлов удаляется. При пустом слое возвращает пустой GeoDataFrame (метрики
    деградируют мягко).
    """
    import geopandas as gpd
    import pandas as pd

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{layer_name}.gpkg"

    if cache_path.exists():
        try:
            return gpd.read_file(cache_path).to_crs(METRIC_CRS)
        except Exception as e:  # битый кэш — перекачаем
            print(f"  [{layer_name}] кэш повреждён ({e}); перекачиваю")

    tile_dir = cache_dir / "tiles" / layer_name
    tile_dir.mkdir(parents=True, exist_ok=True)

    n = len(tiles_wgs)
    print(f"  [{layer_name}] скачиваю {n} тайлов из OSM…")
    parts = []
    failed = []
    total_obj = 0
    t_start = time.time()
    for idx, tile in enumerate(tiles_wgs):
        t0 = time.time()
        try:
            part = _fetch_tile(layer_name, tags, tile, idx, tile_dir)
            k = len(part)
            if k:
                parts.append(part)
                total_obj += k
        except _TileFailed as e:
            failed.append(idx)
            k = -1  # пометка пропуска в логе
            print(f"    {e} — ПРОПУСК (докачается при перезапуске)", flush=True)
        dt = time.time() - t0
        done = idx + 1
        elapsed = time.time() - t_start
        eta = elapsed / done * (n - done)
        kstr = "skip" if k < 0 else f"+{k}"
        print(
            f"  [{layer_name}] тайл {done}/{n} | {kstr} (Σ{total_obj:,}) | "
            f"{dt:4.0f}c | прошло {elapsed/60:4.1f}м · ост.~{eta/60:4.1f}м"
            f"{' | пропущено '+str(len(failed)) if failed else ''}",
            flush=True,
        )

    if not parts:
        print(f"  [{layer_name}] пусто")
        return gpd.GeoDataFrame(geometry=[], crs=METRIC_CRS)

    out = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), crs=METRIC_CRS
    )[["geometry"]]
    # Дедуп объектов с границ тайлов по точной геометрии (WKB).
    out = out[~out.geometry.to_wkb().duplicated()].reset_index(drop=True)

    if failed:
        # слой неполный — НЕ финализируем кэш и НЕ чистим тайлы, чтобы перезапуск
        # докачал только пропущенные тайлы, а готовые взял из резюм-кэша.
        head = failed[:20]
        print(f"  [{layer_name}] ⚠ {len(out):,} об., но НЕ скачано тайлов: "
              f"{len(failed)} {head}{'…' if len(failed) > 20 else ''}")
        print(f"  [{layer_name}] ⚠ слой НЕПОЛНЫЙ — перезапусти ячейку, чтобы докачать пропущенные")
        return out

    try:
        out.to_file(cache_path, driver="GPKG")
        # слой собран полностью — резюм-кэш тайлов больше не нужен
        for f in tile_dir.glob("*"):
            f.unlink(missing_ok=True)
        tile_dir.rmdir()
    except Exception as e:
        print(f"  [{layer_name}] не удалось записать кэш слоя: {e}")

    print(f"  [{layer_name}] ГОТОВО: {len(out):,} объектов")
    return out


def fetch_all_layers(
    points_metric, cache_dir: Path, tags: dict | None = None
) -> dict:
    """Скачивает (или читает из кэша) все слои OSM потайлово.

    Сетка строится **под размер тайла каждого слоя** (`_layer_tile_m`): тяжёлые
    площадные/линейные слои режутся мельче, точечные — крупнее.
    """
    tags = tags or OSM_TAGS
    grids: dict[int, list] = {}  # кэшируем сетку по размеру тайла
    out = {}
    for name, t in tags.items():
        tm = _layer_tile_m(name)
        if tm not in grids:
            grids[tm] = tile_polygons(points_metric, tm)
            print(f"Сетка ~{tm / 1000:.0f} км: {len(grids[tm])} тайлов")
        out[name] = fetch_osm_layer(name, t, grids[tm], cache_dir)
    return out


def fetch_layer(points_metric, cache_dir: Path, layer_name: str):
    """Скачать ОДИН слой OSM потайлово — для пер-слойного запуска из ноутбука.

    Строит сетку под размер тайла этого слоя, печатает прогресс по тайлам с ETA
    и возвращает GeoDataFrame слоя (в METRIC_CRS). Готовый слой берётся из кэша
    мгновенно; прерванное скачивание продолжается с места обрыва.
    """
    if layer_name not in OSM_TAGS:
        raise KeyError(f"неизвестный слой {layer_name!r}; доступны: {list(OSM_TAGS)}")
    tm = _layer_tile_m(layer_name)
    tiles = tile_polygons(points_metric, tm)
    print(f"[{layer_name}] сетка ~{tm / 1000:.0f} км: {len(tiles)} тайлов")
    return fetch_osm_layer(layer_name, OSM_TAGS[layer_name], tiles, cache_dir)


# ---------------------------------------------------------------------------
# 3. Геометрические метрики (всё в METRIC_CRS)
# ---------------------------------------------------------------------------
def nearest_distance_m(points, objects, col_name: str) -> pd.Series:
    """Расстояние (м) до ближайшего объекта слоя."""
    import geopandas as gpd

    if objects.empty:
        return pd.Series(np.nan, index=points.index, name=col_name)
    joined = gpd.sjoin_nearest(
        points[["project_id", "geometry"]], objects[["geometry"]],
        how="left", distance_col=col_name,
    )
    out = joined.groupby(joined.index)[col_name].min()
    return out.reindex(points.index).rename(col_name)


def count_within_radius(points, objects, radius_m: int, col_name: str) -> pd.Series:
    """Число объектов слоя в радиусе (по пересечению с буфером точки)."""
    import geopandas as gpd

    if objects.empty:
        return pd.Series(0, index=points.index, name=col_name)
    buffers = points[["project_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(radius_m)
    joined = gpd.sjoin(objects[["geometry"]], buffers, how="inner", predicate="intersects")
    counts = joined.groupby("project_id").size()
    return (
        points.set_index("project_id").index.to_series()
        .map(counts).fillna(0).astype(int).set_axis(points.index).rename(col_name)
    )


def area_share_within_radius(points, polygons, radius_m: int, col_name: str) -> pd.Series:
    """Доля площади буфера под полигонами слоя (для линий/точек — 0)."""
    if polygons.empty:
        return pd.Series(0.0, index=points.index, name=col_name)
    polys = polygons[polygons.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if polys.empty:
        return pd.Series(0.0, index=points.index, name=col_name)
    sindex = polys.sindex
    out = []
    for geom in points.geometry:
        buf = geom.buffer(radius_m)
        cand = polys.iloc[list(sindex.query(buf, predicate="intersects"))]
        if cand.empty:
            out.append(0.0)
            continue
        inter = cand.geometry.intersection(buf).area.sum()
        out.append(float(inter / buf.area))
    return pd.Series(out, index=points.index, name=col_name)


def line_length_within_radius(points, lines, radius_m: int, col_name: str) -> pd.Series:
    """Суммарная длина (м) линейных объектов слоя в буфере."""
    if lines.empty:
        return pd.Series(0.0, index=points.index, name=col_name)
    lns = lines[lines.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    if lns.empty:
        return pd.Series(0.0, index=points.index, name=col_name)
    sindex = lns.sindex
    out = []
    for geom in points.geometry:
        buf = geom.buffer(radius_m)
        cand = lns.iloc[list(sindex.query(buf, predicate="intersects"))]
        out.append(float(cand.geometry.intersection(buf).length.sum()) if not cand.empty else 0.0)
    return pd.Series(out, index=points.index, name=col_name)


def nearby_complexes_count(points, radius_m: int, col_name: str) -> pd.Series:
    """Число других проектов в радиусе (плотность застройки)."""
    import geopandas as gpd

    buffers = points[["project_id", "geometry"]].copy()
    buffers["geometry"] = buffers.geometry.buffer(radius_m)
    joined = gpd.sjoin(
        points[["project_id", "geometry"]], buffers,
        how="inner", predicate="intersects", lsuffix="a", rsuffix="b",
    )
    joined = joined[joined["project_id_a"] != joined["project_id_b"]]
    counts = joined.groupby("project_id_b").size()
    return (
        points.set_index("project_id").index.to_series()
        .map(counts).fillna(0).astype(int).set_axis(points.index).rename(col_name)
    )


# ---------------------------------------------------------------------------
# 4. Сборка таблицы фич
# ---------------------------------------------------------------------------
def build_features(points, layers: dict, rings=RINGS) -> pd.DataFrame:
    """Считает все геофичи для точек проектов → DataFrame по project_id."""
    feats = points[["project_id"]].copy()

    for layer, col in NEAREST_LAYERS.items():
        feats[col] = nearest_distance_m(points, layers[layer], col).values

    for layer, base in COUNT_LAYERS.items():
        for r in rings:
            feats[f"{base}_{r}m"] = count_within_radius(points, layers[layer], r, base).values

    for layer, base in SHARE_LAYERS.items():
        for r in rings:
            feats[f"{base}_{r}m"] = area_share_within_radius(points, layers[layer], r, base).values

    for layer, base in LENGTH_LAYERS.items():
        for r in rings:
            feats[f"{base}_{r}m"] = line_length_within_radius(points, layers[layer], r, base).values

    for r in rings:
        feats[f"nearby_complexes_count_{r}m"] = nearby_complexes_count(points, r, "x").values

    return feats.set_index("project_id")


def feature_columns(rings=RINGS) -> list[str]:
    """Имена всех геофич (для NUM_FEATURES в 04_features)."""
    cols = list(NEAREST_LAYERS.values())
    for base in COUNT_LAYERS.values():
        cols += [f"{base}_{r}m" for r in rings]
    for base in SHARE_LAYERS.values():
        cols += [f"{base}_{r}m" for r in rings]
    for base in LENGTH_LAYERS.values():
        cols += [f"{base}_{r}m" for r in rings]
    cols += [f"nearby_complexes_count_{r}m" for r in rings]
    return cols


def compute_geo_features(eda_df: pd.DataFrame, cache_dir: Path, rings=RINGS) -> pd.DataFrame:
    """End-to-end: eda_df (с lat/lng/project_id) → таблица геофич по project_id."""
    points = project_points(eda_df)
    print(f"Проектов с валидными координатами: {len(points)}")
    layers = fetch_all_layers(points, cache_dir)
    return build_features(points, layers, rings=rings)
