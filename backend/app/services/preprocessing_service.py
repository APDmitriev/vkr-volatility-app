import math
import pandas as pd


def _safe_feature_name(column: str, used_names: set[str]) -> str:
    base = str(column).strip() or "feature"
    if base.lower() in {"timestamp", "value", "returns"}:
        base = f"feature_{base}"
    name = base
    idx = 2
    while name in used_names:
        name = f"{base}_{idx}"
        idx += 1
    used_names.add(name)
    return name


def preprocess_dataframe(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    drop_duplicate_rows: bool = True,
    drop_duplicate_timestamps: bool = True,
    sort_by_date: bool = True,
    fill_method: str = "drop",
    returns_method: str = "simple",
) -> tuple[pd.DataFrame, dict]:
    rows_before = int(len(df))

    work_df = df.copy()

    work_df[date_column] = pd.to_datetime(work_df[date_column], errors="coerce")
    work_df[value_column] = pd.to_numeric(work_df[value_column], errors="coerce")

    invalid_dates_removed = int(work_df[date_column].isna().sum())
    invalid_values_removed = int(work_df[value_column].isna().sum())

    if fill_method == "drop":
        work_df = work_df.dropna(subset=[date_column, value_column])
    elif fill_method == "ffill":
        work_df = work_df.sort_values(by=date_column)
        work_df[value_column] = work_df[value_column].ffill()
        work_df = work_df.dropna(subset=[date_column, value_column])
    elif fill_method == "bfill":
        work_df = work_df.sort_values(by=date_column)
        work_df[value_column] = work_df[value_column].bfill()
        work_df = work_df.dropna(subset=[date_column, value_column])
    else:
        raise ValueError("fill_method должен быть одним из: drop, ffill, bfill")

    duplicate_rows_removed = 0
    if drop_duplicate_rows:
        before = len(work_df)
        work_df = work_df.drop_duplicates()
        duplicate_rows_removed = int(before - len(work_df))

    duplicate_timestamps_removed = 0
    if drop_duplicate_timestamps:
        before = len(work_df)
        work_df = work_df.drop_duplicates(subset=[date_column], keep="last")
        duplicate_timestamps_removed = int(before - len(work_df))

    if sort_by_date:
        work_df = work_df.sort_values(by=date_column)



    used_names = {"timestamp", "value"}
    result_df = pd.DataFrame(
        {
            "timestamp": work_df[date_column].values,
            "value": work_df[value_column].values,
        }
    )

    feature_columns: list[str] = []
    for column in work_df.columns:
        if column in {date_column, value_column}:
            continue
        converted = pd.to_numeric(work_df[column], errors="coerce").replace([float("inf"), float("-inf")], None)
        valid_count = int(converted.notna().sum())
        if valid_count < max(3, int(len(work_df) * 0.5)):
            continue
        feature_name = _safe_feature_name(str(column), used_names)
        if fill_method == "drop":

            converted = converted.ffill().bfill()
        elif fill_method == "ffill":
            converted = converted.ffill().bfill()
        elif fill_method == "bfill":
            converted = converted.bfill().ffill()
        result_df[feature_name] = converted.reset_index(drop=True)
        feature_columns.append(feature_name)

    if returns_method == "none":
        returns_column = None
    elif returns_method == "simple":
        result_df["returns"] = result_df["value"].pct_change()
        result_df["returns"] = result_df["returns"].replace([float("inf"), float("-inf")], None)
        returns_column = "returns"
    elif returns_method == "log":
        result_df["returns"] = result_df["value"].apply(
            lambda x: math.log(x) if x > 0 else None
        )
        result_df["returns"] = result_df["returns"].diff()
        result_df["returns"] = result_df["returns"].replace([float("inf"), float("-inf")], None)
        returns_column = "returns"
    else:
        raise ValueError("returns_method должен быть одним из: none, simple, log")

    rows_after = int(len(result_df))

    summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "invalid_dates_removed": invalid_dates_removed,
        "invalid_values_removed": invalid_values_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
        "duplicate_timestamps_removed": duplicate_timestamps_removed,
        "fill_method": fill_method,
        "returns_method": returns_method,
        "returns_column": returns_column,
        "feature_columns": feature_columns,
        "features_count": len(feature_columns),
    }

    return result_df, summary
