from pathlib import Path
from uuid import uuid4
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.dataset import Dataset
from app.models.project import Project

from app.schemas.dataset import (
    DatasetResponse,
    DatasetUploadResponse,
    DatasetValidationRequest,
    DatasetValidationResponse,
    DatasetPreprocessRequest,
    DatasetPreprocessResponse,
)
from app.services.dataset_validation_service import validate_dataset_dataframe
from app.services.preprocessing_service import preprocess_dataframe

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
UPLOAD_DIR = BASE_DIR / "storage" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = BASE_DIR / "storage" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def read_dataframe(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        # В пользовательских CSV часто встречаются разделители ",", ";" и табуляция.
        # sep=None включает определение разделителя через Python-engine.
        try:
            return pd.read_csv(file_path, sep=None, engine="python")
        except Exception:
            return pd.read_csv(file_path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)

    raise ValueError("Поддерживаются только CSV, XLSX и XLS")


@router.post(
    "/upload",
    response_model=DatasetUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_dataset(
    project_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Проект не найден",
        )

    original_name = file.filename or ""
    extension = Path(original_name).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Допустимы только файлы CSV, XLSX и XLS",
        )

    saved_name = f"{uuid4().hex}{extension}"
    saved_path = UPLOAD_DIR / saved_name

    content = await file.read()
    saved_path.write_bytes(content)

    try:
        df = read_dataframe(saved_path)
    except Exception as exc:
        if saved_path.exists():
            saved_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось прочитать файл: {exc}",
        ) from exc

    dataset = Dataset(
        project_id=project_id,
        name=Path(original_name).stem or saved_name,
        file_path=str(saved_path),
        rows_count=int(len(df)),
    )

    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    preview_df = df.head(10).copy()
    preview_df = preview_df.where(pd.notnull(preview_df), None)

    return DatasetUploadResponse(
        id=dataset.id,
        project_id=dataset.project_id,
        name=dataset.name,
        file_path=dataset.file_path,
        rows_count=dataset.rows_count or 0,
        columns=[str(col) for col in df.columns.tolist()],
        preview=preview_df.to_dict(orient="records"),
    )


@router.get("/", response_model=list[DatasetResponse])
def get_datasets(project_id: int | None = None, db: Session = Depends(get_db)):
    query = db.query(Dataset)

    if project_id is not None:
        query = query.filter(Dataset.project_id == project_id)

    return query.order_by(Dataset.id.desc()).all()


@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Датасет не найден",
        )

    return dataset

@router.post("/{dataset_id}/validate", response_model=DatasetValidationResponse)
def validate_dataset(
    dataset_id: int,
    payload: DatasetValidationRequest,
    db: Session = Depends(get_db),
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Датасет не найден",
        )

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Файл датасета не найден на сервере",
        )

    try:
        df = read_dataframe(file_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось прочитать файл датасета: {exc}",
        ) from exc

    if payload.date_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Столбец даты '{payload.date_column}' не найден в файле",
        )

    if payload.value_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Столбец значений '{payload.value_column}' не найден в файле",
        )

    report = validate_dataset_dataframe(
        df=df,
        date_column=payload.date_column,
        value_column=payload.value_column,
    )

    dataset.date_column = payload.date_column
    dataset.value_column = payload.value_column
    dataset.rows_count = report["rows_count"]

    db.commit()
    db.refresh(dataset)

    return DatasetValidationResponse(
        dataset_id=dataset.id,
        valid=report["valid"],
        rows_count=report["rows_count"],
        date_column=report["date_column"],
        value_column=report["value_column"],
        summary=report["summary"],
        issues=report["issues"],
    )

@router.post("/{dataset_id}/preprocess", response_model=DatasetPreprocessResponse)
def preprocess_dataset(
    dataset_id: int,
    payload: DatasetPreprocessRequest,
    db: Session = Depends(get_db),
):
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Датасет не найден",
        )

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Файл датасета не найден на сервере",
        )

    try:
        df = read_dataframe(file_path)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось прочитать файл датасета: {exc}",
        ) from exc

    if payload.date_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Столбец даты '{payload.date_column}' не найден в файле",
        )

    if payload.value_column not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Столбец значений '{payload.value_column}' не найден в файле",
        )

    try:
        processed_df, summary = preprocess_dataframe(
            df=df,
            date_column=payload.date_column,
            value_column=payload.value_column,
            drop_duplicate_rows=payload.drop_duplicate_rows,
            drop_duplicate_timestamps=payload.drop_duplicate_timestamps,
            sort_by_date=payload.sort_by_date,
            fill_method=payload.fill_method,
            returns_method=payload.returns_method,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    if processed_df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="После предобработки датасет оказался пустым",
        )

    processed_file_path = PROCESSED_DIR / f"dataset_{dataset.id}_processed.csv"
    processed_df.to_csv(processed_file_path, index=False)

    dataset.date_column = payload.date_column
    dataset.value_column = payload.value_column
    dataset.rows_count = summary["rows_after"]

    db.commit()
    db.refresh(dataset)

    preview_df = processed_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return DatasetPreprocessResponse(
        dataset_id=dataset.id,
        processed_file_path=str(processed_file_path),
        summary=summary,
        preview=preview_df.to_dict(orient="records"),
    )