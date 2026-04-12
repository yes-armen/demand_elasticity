# demand_elasticity

Репозиторий проекта по прогнозированию эластичности спроса на квартиры.

## Состав репозитория

- `utils/unit_key.py` — сборка ключа, нормализация `project_id` / `building_id` / `room_count`, правки из `pool` для прайса.
- `notebooks/preprocessing.ipynb` — загрузка CSV, переименование колонок RU→EN, препроцессинг и ключ.
- `scripts/key_overlap_examples.py` — вспомогательный разбор пересечений ключей (опционально).

Сырые CSV в `data/*.csv` по умолчанию **не коммитятся** (см. `.gitignore`).

## Запуск

Требуется Python 3.11+ и зависимости из `requirements.txt` (например `pip install -r requirements.txt`).
# demand_elasticity
# demand_elasticity
