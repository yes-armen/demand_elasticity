# Справочник по сырым данным

Структура исходных файлов прайса и сделок в `data/raw/`. Производные признаки, таргеты и processed-датасеты — в `README.md` и ноутбуках (`02c`, `04`).

---

## 1. Источники

| Файл | Что |
|---|---|
| `price_dataflat_*.csv` (7 срезов) | прайс-срезы объявлений о продаже квартир, Москва / Новая Москва / МО |
| `deals_dataflat_2020_2025.csv` | реестр сделок 2020–2025 |

`*.xlsx` в `data/raw/` — исходные выгрузки; в пайплайне читаются `.csv`.

---

## 2. Поля прайса (40 колонок)

| raw (рус.) | canonical | тип | примечания |
|---|---|---|---|
| pool | pool | str | источник/пул данных; не фича |
| ID Проекта | project_id | int | |
| ID Проекта для окружения | env_project_id | int | |
| ID Корпус | building_id | str | |
| Проект | project_name | str | текст |
| Девелопер | developer | categorical | |
| Класс проекта | project_class | categorical | комфорт / бизнес (премиум и де-люкс отфильтрованы в 01) |
| Регион | region | categorical | 3 знач.: Москва / МО / НМ |
| Округ | macro_district | categorical | |
| Район | district | categorical | ~211 уник. |
| Старт продаж | sales_start | date | формат DD.MM.YYYY |
| Сдача К | completion_k | date | плановая дата сдачи проекта; формат DD.MM.YYYY |
| Ключи | keys_status | categorical | стадия готовности |
| Тип помещения | premises_type | categorical | |
| Секция | section | categorical | |
| Этаж | floor | int | редкие нечисловые → NaN |
| Отделка в отчет | finish_in_report | categorical | б/о / с/о / wb / Н/Д |
| Тип кв/ап | unit_typology | categorical | |
| Комнатность | room_count | categorical | |
| Площадь | area | float | нечисловые '-' → NaN |
| Цена | price | float | ₽/м² |
| Цена без скидки | price_list | float | прайсовая цена без скидки |
| Старый бюджет | budget_old | float | предыдущий бюджет |
| Бюджет без скидки | budget_list | float | |
| Бюджет | budget | float | |
| Скидка, % | discount_pct | float | нечисловые '-' → NaN |
| Наличие бюджета | has_budget | categorical | в 01 оставляем только `has_budget=1` |
| Изменение цены последнее | last_price_change | float | последнее изменение цены (% или абс.) |
| Дата файла | file_date | date | DD.MM.YYYY; основа temporal split |
| Начало/конец месяца | month_span | categorical | в 01 оставляем только 'begin' |
| Источник | source | categorical | источник данных |
| Период | period | categorical | временной сегмент среза |
| lat | lat | float | широта ЖК → геофичи `04a` |
| lng | lng | float | долгота ЖК → геофичи `04a` |
| Условный номер | conditional_number | str | часть unit_match_key |
| Отделка К | finish_tier | categorical | 0-Нет / 1-Есть / 2-Частично / 3-Неизвестно |
| Отделка текст | finish_text | categorical | текстовое описание отделки |
| Договор К | contract_type_k | categorical | |
| Стадия К | stage_k | categorical | |
| Экспозиция | exposure | int | дней на рынке; '-' → NaN |

---

## 3. Поля сделок (53 колонки)

Сделки используются **только для построения таргета** и join'а.
Фичи из сделок в модель не идут (значения фиксируются на дату сделки, а не на дату прайс-снэпшота).

| raw (рус.) | canonical | тип | примечания |
|---|---|---|---|
| key | deal_row_key | int | уникальный ID сделки |
| lat | lat | float | |
| lng | lng | float | |
| ID Проекта | project_id | int | |
| ID Проекта для окруженияSales | env_project_id_sales | int | |
| ID Корпус | building_id | str | |
| Проект | project_name | str | |
| Девелопер | developer | categorical | дублирует прайс |
| Класс проекта | project_class | categorical | дублирует прайс |
| Регион | region | categorical | дублирует прайс |
| Округ | macro_district | categorical | дублирует прайс |
| Район | district | categorical | дублирует прайс |
| Округ Направление | macro_district_direction | categorical | |
| Комнатность | room_count | categorical | дублирует прайс |
| Этаж | floor | int | дублирует прайс |
| Условный номер | conditional_number | str | часть unit_match_key |
| Площадь | area | float | дублирует прайс |
| Тип помещения | premises_type | categorical | дублирует прайс |
| Наличие бюджета | has_budget | categorical | Да / Нет |
| Цена | price | float | цена на дату сделки |
| Цена без скидки | price_list | float | |
| Бюджет | budget | float | |
| Бюджет без скидки | budget_list | float | |
| Ключи | keys_status | categorical | дублирует прайс |
| Корпус | building_name | categorical | текстовое название корпуса |
| Покупатель ФЛ | buyer_person | int | анонимизированный ID покупателя |
| Покупатель ЮЛ | buyer_company | categorical | |
| Номер регистрации | registration_number | str | уникальный ID регистрации |
| Дата регистрации | registration_date | date | DD.MM.YYYY (округлена до начала месяца, день 01; не использовать) |
| **Дата регистрации_** | **registration_date_raw** | **date** | **DD.MM.YYYY; каноническая дата сделки (реальный день)** |
| Залогодержатель/Банк | pledge_holder_bank | categorical | |
| Длительность обременения | pledge_duration_months | int | месяцев |
| Тип обременения | pledge_type | categorical | ипотека / залог / -; не фича (leak) |
| Отделка | finish | categorical | |
| Дата подписания | signing_date | date | |
| Дата ДДУ | ddu_date | date | |
| Срок сдачи | completion_due | date | |
| Стадия строительства | construction_stage | categorical | |
| Ипотека | mortgage | bool | 0 / 1; не фича (leak) |
| Секция | section | categorical | дублирует прайс |
| Старт продаж | sales_start | date | дублирует прайс |
| Продавец ЮЛ | seller_company | categorical | |
| Продавец ФЛ ID | seller_person_id | str | |
| Описание помещения | premises_description | text | свободный текст |
| Опт | is_wholesale | bool | 0 / 1 |
| Стадия строительства в дату ДДУ | construction_stage_at_ddu | categorical | |
| Цена ДДУ | price_ddu | float | преимущественно '-' |
| Тип сделки | deal_type | categorical | ЗФ / ЗЮ / ФФ / ЗП |
| Дата регистрации модель | registration_date_model | date | |
| Цена кв. м | price_per_sqm | float | производная от price / area |
| ID дом.рф | dom_rf_id | str | |
| Бюджет по ПД | budget_pd | float | |
| Цена по ПД | price_pd | float | |