# VKR Backend Starter

Минимальный backend-скелет для ВКР: FastAPI + SQLAlchemy + PostgreSQL + загрузка CSV/XLSX + превью данных.

## Требования
- Python 3.11+
- PostgreSQL 14+

## Подготовка PostgreSQL
Создай базу данных, например:

```sql
CREATE DATABASE vkr_timeseries;
```

При необходимости создай отдельного пользователя и выдай ему права.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\\Scripts\\activate   # Windows
pip install -r requirements.txt
cp .env .env
uvicorn app.main:app --reload
```

## Настройка строки подключения
По умолчанию в `.env.example` задано:

```env
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/vkr_timeseries
```

Если у тебя другой пользователь, пароль, хост или порт — измени значение в `.env`.

## Что уже реализовано
- health-check
- загрузка CSV/XLSX
- сохранение файла на сервере
- чтение файла через pandas
- возврат списка колонок, числа строк и preview
- сохранение Dataset в PostgreSQL

## Эндпоинты
- `GET /api/v1/health`
- `POST /api/v1/datasets/upload`
- `GET /api/v1/datasets/{dataset_id}`
- `GET /api/v1/datasets`

## Swagger
После запуска открой:
- `http://127.0.0.1:8000/docs`
