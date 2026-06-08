from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Volatile Time Series Backend"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/vkr_timeseries"
    upload_dir: str = "app/storage/uploads"
    max_preview_rows: int = 10

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def upload_path(self) -> Path:
        path = Path(self.upload_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
