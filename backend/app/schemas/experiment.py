from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ExperimentBase(BaseModel):
    project_id: int | None = None
    project_name: str | None = None
    dataset_id: int | None = None
    dataset_name: str | None = None

    model: str = Field(..., min_length=1, max_length=100)
    parameters: dict[str, Any] | None = None
    metrics: dict[str, Any] | None = None

    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    mape: float | None = None

    result_file_path: str | None = None
    status: str = "Успешно"
    raw_response: dict[str, Any] | None = None


class ExperimentCreate(ExperimentBase):
    pass


class ExperimentRead(ExperimentBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
        from_attributes = True
