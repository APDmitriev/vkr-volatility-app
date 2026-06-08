import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException, UploadFile, status

from app.core.config import settings

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class FileService:
    @staticmethod
    def _get_extension(filename: str) -> str:
        return Path(filename).suffix.lower()

    @classmethod
    def validate_extension(cls, filename: str) -> str:
        ext = cls._get_extension(filename)
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Поддерживаются только файлы CSV, XLSX и XLS.",
            )
        return ext

    @staticmethod
    def save_upload(file: UploadFile, extension: str) -> Path:
        target_name = f"{uuid.uuid4().hex}{extension}"
        target_path = settings.upload_path / target_name
        with target_path.open("wb") as buffer:
            buffer.write(file.file.read())
        return target_path

    @staticmethod
    def read_dataframe(file_path: Path, extension: str) -> pd.DataFrame:
        try:
            if extension == ".csv":
                return pd.read_csv(file_path)
            return pd.read_excel(file_path)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Не удалось прочитать файл: {exc}",
            ) from exc

    @staticmethod
    def build_preview(df: pd.DataFrame) -> list[dict[str, Any]]:
        preview_df = df.head(settings.max_preview_rows).copy()
        preview_df = preview_df.where(pd.notnull(preview_df), None)
        preview_df = preview_df.astype(object)
        return preview_df.to_dict(orient="records")

    @staticmethod
    def serialize_columns(columns: list[str]) -> str:
        return json.dumps(columns, ensure_ascii=False)

    @staticmethod
    def deserialize_columns(columns_json: str) -> list[str]:
        return json.loads(columns_json)
