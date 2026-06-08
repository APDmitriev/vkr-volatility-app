from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class DatasetResponse(BaseModel):
    id: int
    project_id: int
    name: str
    file_path: str
    date_column: Optional[str]
    value_column: Optional[str]
    rows_count: Optional[int]
    created_at: datetime

    class Config:
        orm_mode = True
        from_attributes = True


class DatasetUploadResponse(BaseModel):
    id: int
    project_id: int
    name: str
    file_path: str
    rows_count: int
    columns: list[str]
    preview: list[dict[str, Any]]


class DatasetValidationRequest(BaseModel):
    date_column: str
    value_column: str


class DatasetValidationIssue(BaseModel):
    code: str
    level: str
    message: str
    count: int | None = None


class DatasetValidationSummary(BaseModel):
    missing_date_count: int
    missing_value_count: int
    duplicate_rows_count: int
    duplicate_timestamps_count: int
    unsorted_timestamps: bool


class DatasetValidationResponse(BaseModel):
    dataset_id: int
    valid: bool
    rows_count: int
    date_column: str
    value_column: str
    summary: DatasetValidationSummary
    issues: list[DatasetValidationIssue]


class DatasetPreprocessRequest(BaseModel):
    date_column: str
    value_column: str
    drop_duplicate_rows: bool = True
    drop_duplicate_timestamps: bool = True
    sort_by_date: bool = True
    fill_method: str = "drop"
    returns_method: str = "simple"


class DatasetPreprocessSummary(BaseModel):
    rows_before: int
    rows_after: int
    invalid_dates_removed: int
    invalid_values_removed: int
    duplicate_rows_removed: int
    duplicate_timestamps_removed: int
    fill_method: str
    returns_method: str
    returns_column: str | None = None
    feature_columns: list[str] = Field(default_factory=list)
    features_count: int = 0


class DatasetPreprocessResponse(BaseModel):
    dataset_id: int
    processed_file_path: str
    summary: DatasetPreprocessSummary
    preview: list[dict[str, Any]]