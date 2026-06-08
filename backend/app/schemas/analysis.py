from typing import Any
from datetime import datetime
from pydantic import BaseModel, Field


class VolatilityRequest(BaseModel):
    processed_file_path: str
    window_size: int = Field(..., ge=2)
    annualize: bool = False
    periods_per_year: int = Field(252, ge=1)


class VolatilitySummary(BaseModel):
    rows_count: int
    returns_count: int
    volatility_count: int
    min_volatility: float | None = None
    max_volatility: float | None = None
    mean_volatility: float | None = None


class VolatilityResponse(BaseModel):
    result_file_path: str
    method: str
    window_size: int
    annualize: bool
    periods_per_year: int
    summary: VolatilitySummary
    preview: list[dict[str, Any]]


class ArimaForecastRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    p: int = Field(1, ge=0)
    d: int = Field(1, ge=0)
    q: int = Field(1, ge=0)


class ArimaMetrics(BaseModel):
    mae: float
    rmse: float
    mape: float | None = None


class ArimaForecastResponse(BaseModel):
    result_file_path: str
    model: str
    order: list[int]
    train_size: int
    test_size: int
    metrics: ArimaMetrics
    preview: list[dict[str, object]]
    future_preview: list[dict[str, object]] = Field(default_factory=list)



class SarimaForecastRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    p: int = Field(1, ge=0)
    d: int = Field(1, ge=0)
    q: int = Field(1, ge=0)
    seasonal_p: int = Field(1, ge=0)
    seasonal_d: int = Field(0, ge=0)
    seasonal_q: int = Field(1, ge=0)
    seasonal_period: int = Field(7, ge=2)


class SarimaForecastResponse(BaseModel):
    result_file_path: str
    model: str
    order: list[int]
    seasonal_order: list[int]
    train_size: int
    test_size: int
    metrics: ArimaMetrics
    preview: list[dict[str, object]]
    future_preview: list[dict[str, object]] = Field(default_factory=list)



class LinearRegressionForecastRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    window_size: int = Field(5, ge=1)


class LinearRegressionMetrics(BaseModel):
    mae: float
    mse: float
    rmse: float
    mape: float | None = None


class LinearRegressionForecastResponse(BaseModel):
    result_file_path: str
    model: str
    window_size: int
    train_size: int
    test_size: int
    metrics: LinearRegressionMetrics
    preview: list[dict[str, object]]
    future_preview: list[dict[str, object]] = Field(default_factory=list)


class ArimaTuningRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    p_values: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    d_values: list[int] = Field(default_factory=lambda: [0, 1, 2])
    q_values: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    optimize_by: str = "rmse"


class ArimaCandidateResult(BaseModel):
    order: list[int]
    mae: float
    rmse: float
    mape: float | None = None


class ArimaTuningResponse(BaseModel):
    result_file_path: str
    candidates_file_path: str
    best_model: str
    best_order: list[int]
    optimize_by: str
    train_size: int
    test_size: int
    metrics: ArimaMetrics
    candidates: list[ArimaCandidateResult]
    preview: list[dict[str, object]]


class GarchForecastRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    p: int = Field(1, ge=1)
    q: int = Field(1, ge=1)
    annualize: bool = False
    periods_per_year: int = Field(252, ge=1)


class GarchForecastSummary(BaseModel):
    train_size: int
    forecast_horizon: int
    min_forecast_volatility: float | None = None
    max_forecast_volatility: float | None = None
    mean_forecast_volatility: float | None = None


class GarchForecastResponse(BaseModel):
    result_file_path: str
    model: str
    order: list[int]
    annualize: bool
    periods_per_year: int
    summary: GarchForecastSummary
    preview: list[dict[str, object]]
    future_preview: list[dict[str, object]] = Field(default_factory=list)


class FuzzyForecastRequest(BaseModel):
    processed_file_path: str
    min_sets: int = Field(7, ge=3)
    max_sets: int = Field(30, ge=3)
    forecast_horizon: int = Field(1, ge=1)


class FuzzyForecastMetrics(BaseModel):
    mae: float
    mse: float
    mape: float | None = None


class FuzzyForecastResponse(BaseModel):
    result_file_path: str
    model: str
    fuzzy_sets_count: int
    groups_count: int
    universe_min: float
    universe_max: float
    avg_distance: float
    next_forecast: float
    metrics: FuzzyForecastMetrics
    preview: list[dict[str, object]]
    future_preview: list[dict[str, object]] = Field(default_factory=list)

class TuningCandidate(BaseModel):
    parameters: dict[str, Any]
    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    mape: float | None = None


class LinearRegressionTuningRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    window_values: list[int] = Field(default_factory=lambda: [2, 3, 5, 7, 10, 14])
    optimize_by: str = "rmse"


class LinearRegressionTuningResponse(BaseModel):
    result_file_path: str
    candidates_file_path: str
    best_model: str
    best_parameters: dict[str, Any]
    optimize_by: str
    train_size: int
    test_size: int
    metrics: LinearRegressionMetrics
    candidates: list[TuningCandidate]
    preview: list[dict[str, object]]


class SarimaTuningRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    p_values: list[int] = Field(default_factory=lambda: [0, 1, 2])
    d_values: list[int] = Field(default_factory=lambda: [0, 1])
    q_values: list[int] = Field(default_factory=lambda: [0, 1, 2])
    seasonal_p_values: list[int] = Field(default_factory=lambda: [0, 1])
    seasonal_d_values: list[int] = Field(default_factory=lambda: [0, 1])
    seasonal_q_values: list[int] = Field(default_factory=lambda: [0, 1])
    seasonal_period_values: list[int] = Field(default_factory=lambda: [7])
    optimize_by: str = "rmse"


class SarimaTuningResponse(BaseModel):
    result_file_path: str
    candidates_file_path: str
    best_model: str
    best_order: list[int]
    best_seasonal_order: list[int]
    best_parameters: dict[str, Any]
    optimize_by: str
    train_size: int
    test_size: int
    metrics: ArimaMetrics
    candidates: list[TuningCandidate]
    preview: list[dict[str, object]]


class GarchTuningRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    p_values: list[int] = Field(default_factory=lambda: [1, 2])
    q_values: list[int] = Field(default_factory=lambda: [1, 2])
    annualize: bool = False
    periods_per_year: int = Field(252, ge=1)
    optimize_by: str = "mae"


class GarchTuningMetrics(BaseModel):
    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    mape: float | None = None


class GarchTuningResponse(BaseModel):
    result_file_path: str
    candidates_file_path: str
    best_model: str
    best_order: list[int]
    best_parameters: dict[str, Any]
    optimize_by: str
    metrics: GarchTuningMetrics
    summary: GarchForecastSummary
    candidates: list[TuningCandidate]
    preview: list[dict[str, object]]


class FuzzyTuningRequest(BaseModel):
    processed_file_path: str
    min_sets_values: list[int] = Field(default_factory=lambda: [7])
    max_sets_values: list[int] = Field(default_factory=lambda: [7, 10, 15, 20, 25, 30])
    optimize_by: str = "rmse"


class FuzzyTuningMetrics(BaseModel):
    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    mape: float | None = None


class FuzzyTuningResponse(BaseModel):
    result_file_path: str
    candidates_file_path: str
    best_model: str
    best_parameters: dict[str, Any]
    best_fuzzy_sets_count: int | None = None
    optimize_by: str
    metrics: FuzzyTuningMetrics
    summary: dict[str, Any]
    candidates: list[TuningCandidate]
    preview: list[dict[str, object]]


class VolatilityProfileRequest(BaseModel):
    processed_file_path: str
    window_size: int = Field(7, ge=2)
    periods_per_year: int = Field(252, ge=1)


class ModelRecommendation(BaseModel):
    model: str
    score: float
    reason: str


class VolatilityProfileResponse(BaseModel):
    rows_count: int
    returns_count: int
    volatility_level: str
    volatility_level_ru: str
    std_returns: float
    mean_abs_return: float
    relative_volatility: float
    mean_rolling_volatility: float
    max_rolling_volatility: float
    volatility_cv: float
    spike_ratio: float
    max_drawdown: float
    acf_lag_1: float
    abs_returns_acf_lag_1: float
    abs_returns_acf_lag_5: float
    autocorrelation_detected: bool
    seasonality_detected: bool
    seasonal_period: int | None = None
    seasonal_strength: float
    volatility_clustering_detected: bool
    recommended_model: str
    alternative_models: list[str]
    model_ranking: list[ModelRecommendation]
    explanation: str


class NeuralNetworkMetrics(BaseModel):
    mae: float
    mse: float
    rmse: float
    mape: float | None = None


class MLPForecastRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    window_size: int = Field(7, ge=1)
    hidden_layer_1: int = Field(64, ge=1)
    hidden_layer_2: int = Field(32, ge=0)
    max_iter: int = Field(500, ge=100)
    random_state: int = Field(42, ge=0)
    include_exogenous: bool = True


class MLPForecastResponse(BaseModel):
    result_file_path: str
    model: str
    window_size: int
    hidden_layers: list[int]
    max_iter: int
    train_size: int
    test_size: int
    feature_columns: list[str]
    metrics: NeuralNetworkMetrics
    preview: list[dict[str, object]]
    future_preview: list[dict[str, object]] = Field(default_factory=list)


class LSTMForecastRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    window_size: int = Field(7, ge=1)
    hidden_size: int = Field(32, ge=4)
    num_layers: int = Field(1, ge=1)
    epochs: int = Field(60, ge=5)
    learning_rate: float = Field(0.01, gt=0)
    random_state: int = Field(42, ge=0)
    include_exogenous: bool = True


class LSTMForecastResponse(BaseModel):
    result_file_path: str
    model: str
    window_size: int
    hidden_size: int
    num_layers: int
    epochs: int
    train_size: int
    test_size: int
    feature_columns: list[str]
    metrics: NeuralNetworkMetrics
    preview: list[dict[str, object]]
    future_preview: list[dict[str, object]] = Field(default_factory=list)


class MLPTuningRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    window_values: list[int] = Field(default_factory=lambda: [3, 5, 7, 10])
    hidden_layer_1_values: list[int] = Field(default_factory=lambda: [32, 64])
    hidden_layer_2_values: list[int] = Field(default_factory=lambda: [0, 32])
    max_iter: int = Field(400, ge=100)
    random_state: int = Field(42, ge=0)
    include_exogenous: bool = True
    optimize_by: str = "rmse"


class MLPTuningResponse(BaseModel):
    result_file_path: str
    candidates_file_path: str
    best_model: str
    best_parameters: dict[str, Any]
    optimize_by: str
    train_size: int
    test_size: int
    feature_columns: list[str]
    metrics: NeuralNetworkMetrics
    candidates: list[TuningCandidate]
    preview: list[dict[str, object]]


class LSTMTuningRequest(BaseModel):
    processed_file_path: str
    forecast_horizon: int = Field(..., ge=1)
    window_values: list[int] = Field(default_factory=lambda: [3, 5, 7])
    hidden_size_values: list[int] = Field(default_factory=lambda: [16, 32])
    num_layers_values: list[int] = Field(default_factory=lambda: [1])
    epochs: int = Field(40, ge=5)
    learning_rate: float = Field(0.01, gt=0)
    random_state: int = Field(42, ge=0)
    include_exogenous: bool = True
    optimize_by: str = "rmse"


class LSTMTuningResponse(BaseModel):
    result_file_path: str
    candidates_file_path: str
    best_model: str
    best_parameters: dict[str, Any]
    optimize_by: str
    train_size: int
    test_size: int
    feature_columns: list[str]
    metrics: NeuralNetworkMetrics
    candidates: list[TuningCandidate]
    preview: list[dict[str, object]]
