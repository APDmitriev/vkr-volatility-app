from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, selectinload

from app.api.deps import get_db
from app.models.experiment import Experiment
from app.models.experiment_metric import ExperimentMetric
from app.models.experiment_parameter import ExperimentParameter
from app.schemas.experiment import ExperimentCreate, ExperimentRead


router = APIRouter(prefix="/experiments", tags=["experiments"])


def _base_query(db: Session):
    return db.query(Experiment).options(
        selectinload(Experiment.project),
        selectinload(Experiment.dataset),
        selectinload(Experiment.parameter_items),
        selectinload(Experiment.metric_items),
    )


def _as_float(value: Any) -> float | None:
    if value in (None, "", "—"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@router.get("", response_model=list[ExperimentRead], include_in_schema=False)
@router.get("/", response_model=list[ExperimentRead])
def list_experiments(
    project_id: int | None = Query(default=None),
    dataset_id: int | None = Query(default=None),
    model: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    query = _base_query(db)

    if project_id is not None:
        query = query.filter(Experiment.project_id == project_id)
    if dataset_id is not None:
        query = query.filter(Experiment.dataset_id == dataset_id)
    if model:
        query = query.filter(Experiment.model == model)

    return query.order_by(Experiment.created_at.desc()).limit(limit).all()


@router.post("/", response_model=ExperimentRead, status_code=status.HTTP_201_CREATED)
def create_experiment(payload: ExperimentCreate, db: Session = Depends(get_db)):
    metrics = dict(payload.metrics or {})

    # Для обратной совместимости с UI: если метрики пришли отдельными полями,
    # переносим их в нормализованную таблицу experiment_metrics.
    for metric_name, metric_value in {
        "mae": payload.mae,
        "mse": payload.mse,
        "rmse": payload.rmse,
        "mape": payload.mape,
    }.items():
        if metric_value is not None and metric_name not in metrics:
            metrics[metric_name] = metric_value

    experiment = Experiment(
        project_id=payload.project_id,
        dataset_id=payload.dataset_id,
        model=payload.model,
        result_file_path=payload.result_file_path,
        status=payload.status or "Успешно",
    )
    db.add(experiment)
    db.flush()

    for param_name, param_value in (payload.parameters or {}).items():
        db.add(
            ExperimentParameter(
                experiment_id=experiment.id,
                name=str(param_name),
                value=param_value,
            )
        )

    for metric_name, metric_value in metrics.items():
        db.add(
            ExperimentMetric(
                experiment_id=experiment.id,
                name=str(metric_name).lower(),
                value=_as_float(metric_value),
            )
        )

    db.commit()

    return _base_query(db).filter(Experiment.id == experiment.id).first()


@router.get("/{experiment_id}", response_model=ExperimentRead)
def get_experiment(experiment_id: int, db: Session = Depends(get_db)):
    experiment = _base_query(db).filter(Experiment.id == experiment_id).first()
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_experiment(experiment_id: int, db: Session = Depends(get_db)):
    experiment = db.query(Experiment).filter(Experiment.id == experiment_id).first()
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    db.delete(experiment)
    db.commit()
    return None


@router.delete("/", status_code=status.HTTP_204_NO_CONTENT)
def clear_experiments(
    project_id: int | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Experiment)
    if project_id is not None:
        query = query.filter(Experiment.project_id == project_id)
    for experiment in query.all():
        db.delete(experiment)
    db.commit()
    return None
