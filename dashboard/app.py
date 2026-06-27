"""Дашборд локальной эластичности спроса на квартиры.

Запуск:  streamlit run dashboard/app.py
Методология: docs/dashboard_spec.md (V4-реценка, горизонт 30д, area-взвеш. ε, калибр-поправка).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import engine as E
from reliability import reliability_tier, tier_message

st.set_page_config(page_title="Эластичность спроса", layout="wide")
C1, C2, C3 = "#2C7BB6", "#D7542B", "#2D9E5F"

FILTER_LABELS = [
    ("region", "Регион"), ("macro_district", "Округ"), ("district", "Район"),
    ("project_class", "Класс"), ("project_name", "ЖК"), ("developer", "Застройщик"),
    ("room_count", "Комнатность"), ("premises_type", "Тип помещения"),
    ("stage_k", "Стадия"), ("finish_tier", "Отделка"),
]


@st.cache_resource
def get_model():
    return E.load_model()


@st.cache_data
def get_data():
    return E.load_data()


def fmt_money(x):
    return f"{x/1e9:.2f} млрд ₽" if abs(x) >= 1e9 else f"{x/1e6:.1f} млн ₽"


model = get_model()
df = get_data()


# Кэш тяжёлых расчётов по (месяц, применённые фильтры[, план])
# Живое сужение опций фильтров вызывает рерраны, но применённый сегмент при этом
# не меняется → отчёт берётся из кэша и не пересчитывается.
def _seg_for(month_iso: str, sel_items: tuple) -> pd.DataFrame:
    s = E.active_slice(df, pd.Timestamp(month_iso))
    for col, vals in sel_items:
        s = s[s[col].astype(str).isin(list(vals))]
    return s.copy()


@st.cache_data(show_spinner=False)
def cached_qbase(month_iso, sel_items):
    seg = _seg_for(month_iso, sel_items)
    return E.predict_p(model, seg[model.features])


@st.cache_data(show_spinner=False)
def cached_curve(month_iso, sel_items, calib):
    return E.scenario_curve(model, _seg_for(month_iso, sel_items), calib=calib)


@st.cache_data(show_spinner=False)
def get_calibration():
    # динамическая поправка абсолютных величин под текущую модель (факт/Σp по срезам)
    return E.compute_calibration(model, df)


@st.cache_data(show_spinner=False)
def cached_shape(month_iso, sel_items):
    return E.demand_shape(model, _seg_for(month_iso, sel_items))


@st.cache_data(show_spinner=False)
def cached_planscn(month_iso, sel_items, pct):
    seg = _seg_for(month_iso, sel_items)
    p = E.predict_p(model, E.apply_price_shock(seg[model.features], pct))
    area, price = seg["area"].to_numpy(), seg[E.PRICE_COL].to_numpy()
    return float(p.sum()), float((p * area).sum()), float((p * area * price * (1 + pct)).sum())


st.title("📉 Дашборд эластичности спроса на квартиры")
st.caption("Показывает, как изменение цены влияет на спрос и выручку выбранного набора новостроек. "
           "Прогноз на 30 дней (месяц).")

with st.expander("ℹ️ Как читать дашборд (простыми словами)", expanded=False):
    st.markdown("""
**Что считаем.** Для каждой квартиры модель оценивает вероятность, что её купят в ближайшие
30 дней. Складывая эти вероятности по выбранным квартирам, получаем ожидания по спросу.

- **Ожидаемые продажи** — сколько квартир, как ожидается, продастся за 30 дней (сумма вероятностей).
- **Спрос, м²** — то же в квадратных метрах (вероятность × площадь, просуммировано). Главная метрика.
- **Выручка** — ожидаемые проданные метры × цена.
- **Эластичность** — насколько спрос чувствителен к цене. «−1» значит: цена +1% → спрос −1%.
  По модулю **больше 1** — спрос сильно реагирует на цену (эластичный), **меньше 1** — слабо (неэластичный).
- **Плановая цена, ₽/м²** (поле слева) — задайте желаемую среднюю цену за метр; инструмент
  пропорционально переоценит все выбранные квартиры так, чтобы их средневзвешенная цена стала
  равной заданной, и покажет реакцию спроса и выручки.

**Как меняем цену.** Двигаем цену квартиры целиком (и «прайсовую», и со скидкой), сохраняя размер
скидки; считаем, что конкуренты свою цену не меняют.

**Графики.**
- *Кривые спроса и выручки* — что будет при разных изменениях цены; пунктир — ваше плановое изменение.
- *Форма спроса* — ключевой график: пока квартира **дешевле** похожих рядом (левее «1») — снижение
  цены сильно поднимает продажи; когда она на уровне соседей и **дороже** (правее «1») — цена почти
  не влияет.

**Надёжность.** Если в срезе мало квартир или мало ожидаемого спроса — оценка помечается как
ненадёжная (расширьте фильтры).

**Рекомендация цены** появляется только если у выручки есть настоящий пик внутри диапазона. Для
большинства срезов спрос неэластичен (выручка просто растёт с ценой) — тогда честно пишем, что
рекомендации нет.

**Оговорки.** Абсолютные числа — модельная оценка (модель немного завышает, поэтому они уменьшены;
на проценты и эластичность это не влияет). «Вероятность продажи» включает и случай, когда квартиру
просто сняли с продажи, не только реальную сделку.
""")

with st.expander("📐 Формулы и расчёты", expanded=False):
    st.markdown("**Вероятность продажи лота** — выход калиброванной модели:")
    st.latex(r"p_i = P(\text{лот } i \text{ продастся за 30 дней})")
    st.markdown("**Ожидаемое число продаж** (сумма вероятностей = матожидание числа событий, "
                "поэтому вероятности можно складывать):")
    st.latex(r"N = \sum_i p_i")
    st.markdown("**Спрос в м²** (главная метрика) и **выручка**:")
    st.latex(r"Q = \sum_i p_i \cdot \mathrm{area}_i \qquad\qquad R = \sum_i p_i \cdot \mathrm{area}_i \cdot \mathrm{price}_i")
    st.markdown("**Чувствительность к цене (эластичность)** — на сколько % меняется спрос (м²) "
                "при изменении цены на 1%:")
    st.latex(r"E = \frac{Q\big(P(1+\Delta)\big) - Q(P)}{Q(P)\,\cdot\,\Delta}")
    st.markdown("Это в точности средневзвешенная эластичность отдельных квартир, где вес — вклад "
                "квартиры в спрос:")
    st.latex(r"E = \sum_i w_i\,\varepsilon_i,\qquad "
             r"w_i = \frac{p_i\,\mathrm{area}_i}{\sum_j p_j\,\mathrm{area}_j},\qquad "
             r"\varepsilon_i = \frac{p_i\big(P(1+\Delta)\big) - p_i(P)}{p_i(P)\,\cdot\,\Delta}")
    st.markdown(
        r"При изменении цены на $\Delta$: цена и прайсовая цена лота умножаются на $(1+\Delta)$, "
        r"доля скидки сохраняется, медиана цен соседей фиксирована (конкуренты не реагируют) → "
        r"меняются относительные ценовые признаки. "
        r"Абсолютные величины $N, Q, R$ домножаются на калибр-поправку, которая считается "
        r"**динамически** для каждого среза как отношение фактических продаж к $\sum p$. "
        r"Модель склонна завышать $\sum p$: повышенную вероятность получают и лоты, которые затем "
        r"уходят с продажи «обрывом» без сделки, поэтому суммарная $\sum p$ выше числа реальных "
        r"сделок. На эластичность $E$ и проценты поправка не влияет (сокращается в отношении).")

# Фильтры
st.sidebar.header("Фильтры")
# Селектор срезов: только тестовое окно модели; даты для ВИЗУАЛИЗАЦИИ округлены до 1-го
# числа месяца (исходный file_date в данных не меняется — округление только в подписи).
_active_dates = pd.to_datetime(df.loc[df["is_sold_past"] == 0, "file_date"].unique())
months = sorted(d for d in _active_dates if E.TEST_FROM <= d <= E.HORIZON_CUT)
if not months:
    st.error("В датасете нет срезов из тестового окна.")
    st.stop()
month = st.sidebar.selectbox(
    "Месяц среза (тест)", months, index=len(months) - 1,
    format_func=lambda d: pd.Timestamp(d).replace(day=1).date().isoformat(),
)
base = E.active_slice(df, pd.Timestamp(month))

# Динамическая калибр-поправка абсолютных величин под текущую модель и срез
_calib_by_month, _calib_global = get_calibration()
calib = float(_calib_by_month.get(pd.Timestamp(month).to_period("M"), _calib_global))
st.sidebar.caption(f"Калибр-поправка абсолютных величин (этот срез): ×{calib:.2f} "
                   f"(модель {'переоценивает' if calib < 1 else 'недооценивает'} продажи; "
                   "на эластичность и % не влияет).")

# Живые КАСКАДНЫЕ фильтры: опции каждого фильтра сужаются по выбору в остальных.
# Отчёт строится по ПРИМЕНЁННЫМ фильтрам (кнопка «Применить»), а не по живым, —
# поэтому сужение опций (рерраны) не пересобирает тяжёлый отчёт (он из кэша).
_month_iso = pd.Timestamp(month).isoformat()
if st.session_state.get("flt_month") != _month_iso:               # смена месяца → сброс
    for c, _ in FILTER_LABELS:
        st.session_state.pop(f"flt_{c}", None)
    st.session_state["applied_sel"] = {}
    st.session_state["flt_month"] = _month_iso

st.sidebar.caption("Опции каждого фильтра сужаются под уже выбранные. "
                   "Нажмите «Применить», чтобы пересчитать отчёт.")
for col, label in FILTER_LABELS:
    # сужаем по ЖИВОМУ выбору в остальных фильтрах (cross-filter)
    d = base
    for c2, _ in FILTER_LABELS:
        v2 = st.session_state.get(f"flt_{c2}", [])
        if c2 != col and v2:
            d = d[d[c2].astype(str).isin(v2)]
    opts = sorted(d[col].dropna().astype(str).unique())
    key = f"flt_{col}"
    if key in st.session_state:   # выпавшие из опций значения убираем (иначе Streamlit падает)
        kept = [v for v in st.session_state[key] if v in opts]
        if kept != st.session_state[key]:
            st.session_state[key] = kept
    st.sidebar.multiselect(label, opts, key=key)

pending = {c: st.session_state.get(f"flt_{c}", []) for c, _ in FILTER_LABELS}
pending = {c: v for c, v in pending.items() if v}

c_apply, c_reset = st.sidebar.columns(2)
if c_apply.button("Применить", type="primary", use_container_width=True):
    st.session_state["applied_sel"] = pending
if c_reset.button("Сбросить", use_container_width=True):
    for c, _ in FILTER_LABELS:
        st.session_state.pop(f"flt_{c}", None)
    st.session_state["applied_sel"] = {}
    st.rerun()

sel = st.session_state.get("applied_sel", {})
_sel_items = tuple(sorted((c, tuple(v)) for c, v in sel.items()))
if sel:
    st.sidebar.caption("Применено: " + " · ".join(f"{c} ({len(v)})" for c, v in sel.items()))
if pending != sel:
    st.sidebar.info("Фильтры изменены — нажмите «Применить», чтобы пересчитать отчёт.")

seg = _seg_for(_month_iso, _sel_items)
if len(seg) == 0:
    st.warning("Под выбранные фильтры нет активных лотов. Ослабьте фильтры.")
    st.stop()

# Средневзвешенная по площади цена сегмента — текущая база для планирования.
wavg_price = float(np.average(seg[E.PRICE_COL], weights=seg["area"]))

# Инструмент: задаём плановую цену за м² напрямую (а не относительный сдвиг)
_seg_key = f"{pd.Timestamp(month).date()}|" + "|".join(
    f"{c}={','.join(v)}" for c, v in sorted(sel.items()))
plan_price = st.sidebar.number_input(
    "Плановая цена, ₽/м²", min_value=1.0,
    value=float(round(wavg_price, -2)), step=1000.0, key=f"plan_price::{_seg_key}",
    help="Задайте плановую средневзвешенную цену за м² для выбранного среза. Инструмент "
         "пропорционально переоценит все лоты так, чтобы их средневзвешенная цена стала "
         "равна заданной, и пересчитает спрос, выручку и чувствительность.",
)
pct = plan_price / wavg_price - 1.0
st.sidebar.caption(f"Текущая средневзвешенная цена: {wavg_price:,.0f} ₽/м²  ·  "
                   f"плановая: {plan_price:,.0f} ₽/м²  ({pct:+.1%}).")

# ЖК под условиями (наверху)
seg["_q_base"] = cached_qbase(_month_iso, _sel_items) * seg["area"].to_numpy() * calib
proj = (seg.groupby("project_name", observed=True)
          .agg(**{"лотов": ("project_name", "size"),
                  "в продаже, м²": ("area", "sum"),
                  "ожид. спрос, м²": ("_q_base", "sum")})
          .reset_index().rename(columns={"project_name": "ЖК"}))
proj["доля лотов, %"] = (100 * proj["лотов"] / len(seg)).round(1)
proj = proj.sort_values("лотов", ascending=False)
proj["в продаже, м²"] = proj["в продаже, м²"].round(0)
proj["ожид. спрос, м²"] = proj["ожид. спрос, м²"].round(0)
with st.expander(f"📋 ЖК под условиями: {len(proj)}  ·  лотов: {len(seg):,}", expanded=True):
    st.dataframe(proj, hide_index=True, width="stretch", height=260)

# Расчёт
curve = cached_curve(_month_iso, _sel_items, calib)
q0 = curve.loc[curve["pct"] == 0.0, "q_sqm"].iloc[0]
n0 = curve.loc[curve["pct"] == 0.0, "n_sales"].iloc[0]
r0 = curve.loc[curve["pct"] == 0.0, "revenue"].iloc[0]
eps = E.elasticity_pm5(curve)
supply = float(seg["area"].sum())
tier = reliability_tier(len(seg), q0)

# сценарий слайдера (может быть вне сетки ±15 → считаем напрямую)
if pct != 0:
    _ns, _qs, _rs = cached_planscn(_month_iso, _sel_items, pct)
    n_sc = _ns * calib
    q_sc = _qs * calib
    r_sc = _rs * calib
else:
    n_sc, q_sc, r_sc = n0, q0, r0

# Надёжность
{"high": st.success, "medium": st.info, "low": st.warning}[tier](tier_message(tier, len(seg), q0))

# KPI
st.subheader(f"Срез: {len(seg):,} активных лотов · {pd.Timestamp(month).replace(day=1).date()}")
k = st.columns(4)
k[0].metric("Всего в продаже", f"{supply:,.0f} м²",
            help="Сколько всего квадратных метров выставлено на продажу в выбранном срезе.")
k[1].metric("Средневзвешенная цена (тек.)", f"{wavg_price:,.0f} ₽/м²",
            help="Текущая средняя цена за м², взвешенная по площади лотов "
                 "(вклад крупных лотов больше).")
k[2].metric("Плановая цена", f"{plan_price:,.0f} ₽/м²", f"{pct:+.1%}" if pct else None,
            help="Заданная вами средневзвешенная цена за м²; стрелка — отклонение от текущей.")
k[3].metric("Чувствительность к цене", f"{eps:.2f}",
            help="Эластичность: на сколько % меняется спрос при изменении цены на 1%. "
                 "По модулю >1 — спрос сильно реагирует на цену, <1 — слабо.")

k = st.columns(4)
k[0].metric("Спрос сейчас", f"{q0:,.0f} м²",
            help="Ожидаемый объём продаж в м² за 30 дней при текущей цене.")
k[1].metric("Спрос при план. цене", f"{q_sc:,.0f} м²", f"{100*(q_sc/q0-1):+.1f}%" if pct else None,
            help="Ожидаемый спрос в м² при заданной плановой цене.")
k[2].metric("Выручка сейчас", fmt_money(r0),
            help="Ожидаемые проданные м² × цена, при текущей цене.")
k[3].metric("Выручка при план. цене", fmt_money(r_sc), f"{100*(r_sc/r0-1):+.1f}%" if pct else None,
            help="Ожидаемая выручка при заданной плановой цене. "
                 f"Ожид. продаж: {n0:,.0f} → {n_sc:,.0f}.")

# Кривые спроса и выручки
st.subheader("Кривые спроса и выручки")
lbl = [f"{p*100:+g}%" for p in curve["pct"]]             # +2.5% / +5% / -15% без лишних нулей
gx = curve["pct"] * 100                                  # числовая ось Δ%
plan_line = -15 <= pct * 100 <= 15                       # план в пределах сетки графика
c1, c2 = st.columns(2)
with c1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gx, curve["q_sqm"], "o-", color=C1, lw=2)
    if plan_line:
        ax.axvline(pct * 100, color=C2, ls=":", label=f"план {pct:+.1%}"); ax.legend()
    ax.set_title("Ожидаемый спрос, м²"); ax.set_xlabel("Δ цены, %"); ax.grid(alpha=.3)
    st.pyplot(fig)
with c2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gx, curve["revenue"] / 1e9, "s-", color=C3, lw=2)
    if plan_line:
        ax.axvline(pct * 100, color=C2, ls=":")
    ax.set_title("Ожидаемая выручка, млрд ₽"); ax.set_xlabel("Δ цены, %"); ax.grid(alpha=.3)
    st.pyplot(fig)

# revenue-gate: рекомендация цены только при интерьерном пике выручки с нетривиальным приростом
g = E.revenue_gate(curve)
if g["ok"]:
    msg = (f"✅ Выгоднее всего изменить цену на **{g['opt']*100:+g}%** — при этом выручка максимальна "
           f"в диапазоне (прирост ≈ {g['uplift']:+.1f}% к текущей).")
    if not g["demand_monotone"]:
        msg += " _(спрос слегка немонотонен на неэластичном хвосте — эффект модели, на вывод не влияет)_"
    st.success(msg)
elif g["interior"]:
    st.info(f"Оптимум выручки около текущей цены (Δ={g['opt']*100:+g}%, прирост лишь "
            f"{g['uplift']:+.1f}%) — менять цену смысла практически нет.")
else:
    st.warning("⚠️ Выручка растёт до края диапазона — спрос неэластичен, надёжной рекомендации "
               "по цене нет (модель не ограничивает цену сверху).")

# Форма спроса (асимметрия)
shape = cached_shape(_month_iso, _sel_items)
if shape is not None:
    sh, labelable = shape
    st.subheader("Форма спроса: дешевле или дороже соседей")
    st.caption("По горизонтали — цена квартиры относительно типичной цены похожих квартир рядом "
               "(1.0 = как у соседей, левее = дешевле, правее = дороже). Видно: дешевле соседей — "
               "снижение цены сильно повышает продажи; дороже — цена почти не влияет.")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(sh["rel_price"], sh["model"], "s--", color=C1, label="модель P(sale)")
    if labelable and "actual" in sh:
        ax.plot(sh["rel_price"], sh["actual"], "o-", color=C2, label="факт. доля продаж")
    ax.axvline(1.0, color="0.5", ls=":", label="медиана сегмента")
    ax.set_xlabel("rel_price = цена / медиана сегмента"); ax.set_ylabel("доля продаж за 30д")
    ax.grid(alpha=.3); ax.legend()
    st.pyplot(fig)
    if not labelable:
        st.caption("Срез не labelable (нет полного 30д окна) — фактическая кривая недоступна, "
                   "показана только модель.")

# Позиция плановой цены
st.subheader("Позиция плановой цены в распределении предложения")
st.caption("Гистограмма — распределение цен за м² лотов выбранного среза. Линии: "
           "синяя пунктирная — медиана предложения; серая полоса — межквартильный диапазон "
           "P25–P75 (центральная половина лотов); оранжевая — текущая средневзвешенная цена; "
           "красная — ваша плановая цена. Чем левее красная линия, тем дешевле вы встаёте "
           "относительно конкурентов.")
pp = seg[E.PRICE_COL].to_numpy()
p25, p50, p75 = (float(x) for x in np.percentile(pp, [25, 50, 75]))
plan_pctile = float((pp < plan_price).mean() * 100)        # доля лотов дешевле плановой цены
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.hist(pp, bins=40, color="#9ecae1", edgecolor="white", alpha=0.85)
ax.axvspan(p25, p75, color="0.5", alpha=0.15, label="P25–P75 (межквартильный диапазон)")
ax.axvline(p50, color=C1, ls="--", lw=2, label=f"медиана: {p50:,.0f} ₽/м²")
ax.axvline(wavg_price, color=C2, ls="-", lw=2, label=f"средневзвеш. (тек.): {wavg_price:,.0f} ₽/м²")
ax.axvline(plan_price, color="#D7191C", ls="-", lw=3, label=f"плановая: {plan_price:,.0f} ₽/м²")
ax.set_xlabel("Цена, ₽/м²"); ax.set_ylabel("Число лотов"); ax.grid(alpha=.3)
ax.legend(loc="upper right", fontsize=8)
st.pyplot(fig)
_where = "ниже" if plan_price < p50 else "выше"
_rel = abs(plan_price / p50 - 1) * 100
st.caption(f"Плановая цена {plan_price:,.0f} ₽/м² {_where} медианы предложения на {_rel:.1f}% "
           f"и соответствует примерно {plan_pctile:.0f}-му перцентилю: дешевле неё "
           f"{plan_pctile:.0f}% лотов среза.")

# Сценарная таблица
st.subheader("Сценарная таблица")
tbl = curve.assign(
    **{"Δ цены": lbl,
       "цена ₽/м²": (wavg_price * (1 + curve["pct"])).round(0),
       "N продаж": curve["n_sales"].round(0),
       "спрос Q, м²": curve["q_sqm"].round(0),
       "Δспрос %": curve["q_chg_pct"].round(1),
       "выручка млрд": (curve["revenue"] / 1e9).round(2),
       "Δвыручка %": curve["rev_chg_pct"].round(1)})
st.dataframe(tbl[["Δ цены", "цена ₽/м²", "N продаж", "спрос Q, м²", "Δспрос %",
                  "выручка млрд", "Δвыручка %"]], hide_index=True, width="stretch")
