from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from app.schemas.analysis import (
    VolatilityRequest,
    VolatilityResponse,
    VolatilityProfileRequest,
    VolatilityProfileResponse,
    ArimaForecastRequest,
    ArimaForecastResponse,
    SarimaForecastRequest,
    SarimaForecastResponse,
    LinearRegressionForecastRequest,
    LinearRegressionForecastResponse,
    ArimaTuningRequest,
    ArimaTuningResponse,
    LinearRegressionTuningRequest,
    LinearRegressionTuningResponse,
    SarimaTuningRequest,
    SarimaTuningResponse,
    GarchTuningRequest,
    GarchTuningResponse,
    FuzzyTuningRequest,
    FuzzyTuningResponse,
    GarchForecastRequest,
    GarchForecastResponse,
    FuzzyForecastRequest,
    FuzzyForecastResponse,
    MLPForecastRequest,
    MLPForecastResponse,
    LSTMForecastRequest,
    LSTMForecastResponse,
    MLPTuningRequest,
    MLPTuningResponse,
    LSTMTuningRequest,
    LSTMTuningResponse,
)
from app.services.volatility_service import calculate_rolling_volatility
from app.services.volatility_profile_service import calculate_volatility_profile
from app.services.arima_service import run_arima_forecast
from app.services.sarima_service import run_sarima_forecast
from app.services.linear_regression_service import run_linear_regression_forecast
from app.services.mlp_service import run_mlp_forecast
from app.services.lstm_service import run_lstm_forecast
from app.services.model_tuning_service import (
    tune_linear_regression_forecast,
    tune_arima_forecast,
    tune_sarima_forecast,
    tune_garch_forecast,
    tune_fuzzy_first_order_forecast,
    tune_mlp_forecast,
    tune_lstm_forecast,
)
from app.services.garch_service import run_garch_forecast
from app.services.fuzzy_time_series_service import run_fuzzy_first_order_forecast

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "storage" / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/volatility", response_model=VolatilityResponse)
def calculate_volatility(payload: VolatilityRequest):
    try:
        result_df, summary = calculate_rolling_volatility(
            processed_file_path=payload.processed_file_path,
            window_size=payload.window_size,
            annualize=payload.annualize,
            periods_per_year=payload.periods_per_year,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось рассчитать волатильность: {exc}",
        ) from exc

    result_file_path = ANALYSIS_DIR / f"volatility_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)

    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return VolatilityResponse(
        result_file_path=str(result_file_path),
        method="rolling_std",
        window_size=payload.window_size,
        annualize=payload.annualize,
        periods_per_year=payload.periods_per_year,
        summary=summary,
        preview=preview_df.to_dict(orient="records"),
    )



@router.post("/volatility-profile", response_model=VolatilityProfileResponse)
def volatility_profile(payload: VolatilityProfileRequest):
    try:
        return calculate_volatility_profile(
            processed_file_path=payload.processed_file_path,
            window_size=payload.window_size,
            periods_per_year=payload.periods_per_year,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось оценить профиль волатильности: {exc}",
        ) from exc

@router.post("/forecast/linear-regression", response_model=LinearRegressionForecastResponse)
def forecast_linear_regression(payload: LinearRegressionForecastRequest):
    try:
        result_df, summary = run_linear_regression_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            window_size=payload.window_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось построить прогноз линейной регрессией: {exc}",
        ) from exc

    result_file_path = ANALYSIS_DIR / f"linear_regression_forecast_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)

    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return LinearRegressionForecastResponse(
        result_file_path=str(result_file_path),
        model=summary["model"],
        window_size=summary["window_size"],
        train_size=summary["train_size"],
        test_size=summary["test_size"],
        metrics=summary["metrics"],
        preview=preview_df.to_dict(orient="records"),
        future_preview=summary.get("future_preview", []),
    )


@router.post("/forecast/linear-regression/tune", response_model=LinearRegressionTuningResponse)
def tune_linear_regression(payload: LinearRegressionTuningRequest):
    try:
        result_df, summary = tune_linear_regression_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            window_values=payload.window_values,
            optimize_by=payload.optimize_by,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось подобрать Linear Regression: {exc}",
        ) from exc

    result_file_path = ANALYSIS_DIR / f"linear_regression_tuned_forecast_{uuid4().hex}.csv"
    candidates_file_path = ANALYSIS_DIR / f"linear_regression_candidates_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)
    pd.DataFrame(summary["candidates"]).to_csv(candidates_file_path, index=False)

    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return LinearRegressionTuningResponse(
        result_file_path=str(result_file_path),
        candidates_file_path=str(candidates_file_path),
        best_model=summary["best_model"],
        best_parameters=summary["best_parameters"],
        optimize_by=summary["optimize_by"],
        train_size=summary["train_size"],
        test_size=summary["test_size"],
        metrics=summary["metrics"],
        candidates=summary["candidates"][:10],
        preview=preview_df.to_dict(orient="records"),
    )


@router.post("/forecast/arima", response_model=ArimaForecastResponse)
def forecast_arima(payload: ArimaForecastRequest):
    try:
        result_df, summary = run_arima_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            p=payload.p,
            d=payload.d,
            q=payload.q,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось построить ARIMA-прогноз: {exc}",
        ) from exc

    result_file_path = ANALYSIS_DIR / f"arima_forecast_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)

    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return ArimaForecastResponse(
        result_file_path=str(result_file_path),
        model=summary["model"],
        order=summary["order"],
        train_size=summary["train_size"],
        test_size=summary["test_size"],
        metrics=summary["metrics"],
        preview=preview_df.to_dict(orient="records"),
        future_preview=summary.get("future_preview", []),
    )


@router.post("/forecast/sarima", response_model=SarimaForecastResponse)
def forecast_sarima(payload: SarimaForecastRequest):
    try:
        result_df, summary = run_sarima_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            p=payload.p,
            d=payload.d,
            q=payload.q,
            seasonal_p=payload.seasonal_p,
            seasonal_d=payload.seasonal_d,
            seasonal_q=payload.seasonal_q,
            seasonal_period=payload.seasonal_period,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось построить SARIMA-прогноз: {exc}",
        ) from exc

    result_file_path = ANALYSIS_DIR / f"sarima_forecast_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)

    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return SarimaForecastResponse(
        result_file_path=str(result_file_path),
        model=summary["model"],
        order=summary["order"],
        seasonal_order=summary["seasonal_order"],
        train_size=summary["train_size"],
        test_size=summary["test_size"],
        metrics=summary["metrics"],
        preview=preview_df.to_dict(orient="records"),
        future_preview=summary.get("future_preview", []),
    )


@router.post("/forecast/arima/tune", response_model=ArimaTuningResponse)
def tune_arima(payload: ArimaTuningRequest):
    try:
        result_df, summary = tune_arima_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            p_values=payload.p_values,
            d_values=payload.d_values,
            q_values=payload.q_values,
            optimize_by=payload.optimize_by,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось подобрать ARIMA-модель: {exc}",
        ) from exc

    result_file_path = ANALYSIS_DIR / f"arima_tuned_forecast_{uuid4().hex}.csv"
    candidates_file_path = ANALYSIS_DIR / f"arima_candidates_{uuid4().hex}.csv"

    result_df.to_csv(result_file_path, index=False)
    pd.DataFrame(summary["candidates"]).to_csv(candidates_file_path, index=False)

    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return ArimaTuningResponse(
        result_file_path=str(result_file_path),
        candidates_file_path=str(candidates_file_path),
        best_model=summary["best_model"],
        best_order=summary["best_order"],
        optimize_by=summary["optimize_by"],
        train_size=summary["train_size"],
        test_size=summary["test_size"],
        metrics=summary["metrics"],
        candidates=summary["candidates"][:10],
        preview=preview_df.to_dict(orient="records"),
    )


@router.post("/forecast/sarima/tune", response_model=SarimaTuningResponse)
def tune_sarima(payload: SarimaTuningRequest):
    try:
        result_df, summary = tune_sarima_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            p_values=payload.p_values,
            d_values=payload.d_values,
            q_values=payload.q_values,
            seasonal_p_values=payload.seasonal_p_values,
            seasonal_d_values=payload.seasonal_d_values,
            seasonal_q_values=payload.seasonal_q_values,
            seasonal_period_values=payload.seasonal_period_values,
            optimize_by=payload.optimize_by,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось подобрать SARIMA-модель: {exc}",
        ) from exc

    result_file_path = ANALYSIS_DIR / f"sarima_tuned_forecast_{uuid4().hex}.csv"
    candidates_file_path = ANALYSIS_DIR / f"sarima_candidates_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)
    pd.DataFrame(summary["candidates"]).to_csv(candidates_file_path, index=False)

    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return SarimaTuningResponse(
        result_file_path=str(result_file_path),
        candidates_file_path=str(candidates_file_path),
        best_model=summary["best_model"],
        best_order=summary["best_order"],
        best_seasonal_order=summary["best_seasonal_order"],
        best_parameters=summary["best_parameters"],
        optimize_by=summary["optimize_by"],
        train_size=summary["train_size"],
        test_size=summary["test_size"],
        metrics=summary["metrics"],
        candidates=summary["candidates"][:10],
        preview=preview_df.to_dict(orient="records"),
    )


@router.post("/forecast/garch/tune", response_model=GarchTuningResponse)
def tune_garch(payload: GarchTuningRequest):
    try:
        result_df, summary = tune_garch_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            p_values=payload.p_values,
            q_values=payload.q_values,
            annualize=payload.annualize,
            periods_per_year=payload.periods_per_year,
            optimize_by=payload.optimize_by,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось подобрать GARCH-модель: {exc}",
        ) from exc

    result_file_path = ANALYSIS_DIR / f"garch_tuned_forecast_{uuid4().hex}.csv"
    candidates_file_path = ANALYSIS_DIR / f"garch_candidates_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)
    pd.DataFrame(summary["candidates"]).to_csv(candidates_file_path, index=False)

    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return GarchTuningResponse(
        result_file_path=str(result_file_path),
        candidates_file_path=str(candidates_file_path),
        best_model=summary["best_model"],
        best_order=summary["best_order"],
        best_parameters=summary["best_parameters"],
        optimize_by=summary["optimize_by"],
        metrics=summary["metrics"],
        summary=summary["summary"],
        candidates=summary["candidates"][:10],
        preview=preview_df.to_dict(orient="records"),
    )


@router.post("/forecast/fuzzy-first-order/tune", response_model=FuzzyTuningResponse)
def tune_fuzzy_first_order(payload: FuzzyTuningRequest):
    try:
        result_df, summary = tune_fuzzy_first_order_forecast(
            processed_file_path=payload.processed_file_path,
            min_sets_values=payload.min_sets_values,
            max_sets_values=payload.max_sets_values,
            optimize_by=payload.optimize_by,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось подобрать Fuzzy Time Series: {exc}",
        ) from exc

    result_file_path = ANALYSIS_DIR / f"fuzzy_first_order_tuned_{uuid4().hex}.csv"
    candidates_file_path = ANALYSIS_DIR / f"fuzzy_first_order_candidates_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)
    pd.DataFrame(summary["candidates"]).to_csv(candidates_file_path, index=False)

    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return FuzzyTuningResponse(
        result_file_path=str(result_file_path),
        candidates_file_path=str(candidates_file_path),
        best_model=summary["best_model"],
        best_parameters=summary["best_parameters"],
        best_fuzzy_sets_count=summary.get("best_fuzzy_sets_count"),
        optimize_by=summary["optimize_by"],
        metrics=summary["metrics"],
        summary=summary["summary"],
        candidates=summary["candidates"][:10],
        preview=preview_df.to_dict(orient="records"),
    )


@router.post("/forecast/garch", response_model=GarchForecastResponse)
def forecast_garch(payload: GarchForecastRequest):
    try:
        result_df, summary = run_garch_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            p=payload.p,
            q=payload.q,
            annualize=payload.annualize,
            periods_per_year=payload.periods_per_year,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось построить GARCH-прогноз: {exc}",
        ) from exc

    result_file_path = ANALYSIS_DIR / f"garch_forecast_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)

    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return GarchForecastResponse(
        result_file_path=str(result_file_path),
        model="GARCH",
        order=[payload.p, payload.q],
        annualize=payload.annualize,
        periods_per_year=payload.periods_per_year,
        summary=summary,
        preview=preview_df.to_dict(orient="records"),
        future_preview=summary.get("future_preview", []),
    )


@router.post("/forecast/fuzzy-first-order", response_model=FuzzyForecastResponse)
def forecast_fuzzy_first_order(payload: FuzzyForecastRequest):
    try:
        result_df, summary = run_fuzzy_first_order_forecast(
            processed_file_path=payload.processed_file_path,
            min_sets=payload.min_sets,
            max_sets=payload.max_sets,
            forecast_horizon=payload.forecast_horizon,
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Не удалось построить нечёткий прогноз: {exc}",
        ) from exc

    result_file_path = ANALYSIS_DIR / f"fuzzy_first_order_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)

    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)

    return FuzzyForecastResponse(
        result_file_path=str(result_file_path),
        model=summary["model"],
        fuzzy_sets_count=summary["fuzzy_sets_count"],
        groups_count=summary["groups_count"],
        universe_min=summary["universe_min"],
        universe_max=summary["universe_max"],
        avg_distance=summary["avg_distance"],
        next_forecast=summary["next_forecast"],
        metrics=summary["metrics"],
        preview=preview_df.to_dict(orient="records"),
        future_preview=summary.get("future_preview", []),
    )

@router.post("/forecast/mlp", response_model=MLPForecastResponse)
def forecast_mlp(payload: MLPForecastRequest):
    try:
        result_df, summary = run_mlp_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            window_size=payload.window_size,
            hidden_layer_1=payload.hidden_layer_1,
            hidden_layer_2=payload.hidden_layer_2,
            max_iter=payload.max_iter,
            random_state=payload.random_state,
            include_exogenous=payload.include_exogenous,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Не удалось обучить MLP: {exc}") from exc

    result_file_path = ANALYSIS_DIR / f"mlp_forecast_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)
    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)
    return MLPForecastResponse(
        result_file_path=str(result_file_path),
        model=summary["model"],
        window_size=summary["window_size"],
        hidden_layers=summary["hidden_layers"],
        max_iter=summary["max_iter"],
        train_size=summary["train_size"],
        test_size=summary["test_size"],
        feature_columns=summary["feature_columns"],
        metrics=summary["metrics"],
        preview=preview_df.to_dict(orient="records"),
        future_preview=summary.get("future_preview", []),
    )


@router.post("/forecast/mlp/tune", response_model=MLPTuningResponse)
def tune_mlp(payload: MLPTuningRequest):
    try:
        result_df, summary = tune_mlp_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            window_values=payload.window_values,
            hidden_layer_1_values=payload.hidden_layer_1_values,
            hidden_layer_2_values=payload.hidden_layer_2_values,
            max_iter=payload.max_iter,
            random_state=payload.random_state,
            include_exogenous=payload.include_exogenous,
            optimize_by=payload.optimize_by,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Не удалось подобрать MLP: {exc}") from exc

    result_file_path = ANALYSIS_DIR / f"mlp_tuned_forecast_{uuid4().hex}.csv"
    candidates_file_path = ANALYSIS_DIR / f"mlp_candidates_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)
    pd.DataFrame(summary["candidates"]).to_csv(candidates_file_path, index=False)
    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)
    return MLPTuningResponse(
        result_file_path=str(result_file_path),
        candidates_file_path=str(candidates_file_path),
        best_model=summary["best_model"],
        best_parameters=summary["best_parameters"],
        optimize_by=summary["optimize_by"],
        train_size=summary["train_size"],
        test_size=summary["test_size"],
        feature_columns=summary["feature_columns"],
        metrics=summary["metrics"],
        candidates=summary["candidates"][:10],
        preview=preview_df.to_dict(orient="records"),
    )


@router.post("/forecast/lstm", response_model=LSTMForecastResponse)
def forecast_lstm(payload: LSTMForecastRequest):
    try:
        result_df, summary = run_lstm_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            window_size=payload.window_size,
            hidden_size=payload.hidden_size,
            num_layers=payload.num_layers,
            epochs=payload.epochs,
            learning_rate=payload.learning_rate,
            random_state=payload.random_state,
            include_exogenous=payload.include_exogenous,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Не удалось обучить LSTM: {exc}") from exc

    result_file_path = ANALYSIS_DIR / f"lstm_forecast_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)
    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)
    return LSTMForecastResponse(
        result_file_path=str(result_file_path),
        model=summary["model"],
        window_size=summary["window_size"],
        hidden_size=summary["hidden_size"],
        num_layers=summary["num_layers"],
        epochs=summary["epochs"],
        train_size=summary["train_size"],
        test_size=summary["test_size"],
        feature_columns=summary["feature_columns"],
        metrics=summary["metrics"],
        preview=preview_df.to_dict(orient="records"),
        future_preview=summary.get("future_preview", []),
    )


@router.post("/forecast/lstm/tune", response_model=LSTMTuningResponse)
def tune_lstm(payload: LSTMTuningRequest):
    try:
        result_df, summary = tune_lstm_forecast(
            processed_file_path=payload.processed_file_path,
            forecast_horizon=payload.forecast_horizon,
            window_values=payload.window_values,
            hidden_size_values=payload.hidden_size_values,
            num_layers_values=payload.num_layers_values,
            epochs=payload.epochs,
            learning_rate=payload.learning_rate,
            random_state=payload.random_state,
            include_exogenous=payload.include_exogenous,
            optimize_by=payload.optimize_by,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Не удалось подобрать LSTM: {exc}") from exc

    result_file_path = ANALYSIS_DIR / f"lstm_tuned_forecast_{uuid4().hex}.csv"
    candidates_file_path = ANALYSIS_DIR / f"lstm_candidates_{uuid4().hex}.csv"
    result_df.to_csv(result_file_path, index=False)
    pd.DataFrame(summary["candidates"]).to_csv(candidates_file_path, index=False)
    preview_df = result_df.head(10).copy()
    preview_df = preview_df.replace([np.inf, -np.inf], np.nan)
    preview_df = preview_df.astype(object).where(pd.notnull(preview_df), None)
    return LSTMTuningResponse(
        result_file_path=str(result_file_path),
        candidates_file_path=str(candidates_file_path),
        best_model=summary["best_model"],
        best_parameters=summary["best_parameters"],
        optimize_by=summary["optimize_by"],
        train_size=summary["train_size"],
        test_size=summary["test_size"],
        feature_columns=summary["feature_columns"],
        metrics=summary["metrics"],
        candidates=summary["candidates"][:10],
        preview=preview_df.to_dict(orient="records"),
    )
