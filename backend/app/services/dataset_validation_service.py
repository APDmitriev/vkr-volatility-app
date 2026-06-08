import pandas as pd


def validate_dataset_dataframe(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
) -> dict:
    parsed_dates = pd.to_datetime(df[date_column], errors="coerce")
    numeric_values = pd.to_numeric(df[value_column], errors="coerce")

    missing_date_count = int(parsed_dates.isna().sum())
    missing_value_count = int(numeric_values.isna().sum())
    duplicate_rows_count = int(df.duplicated().sum())

    non_null_dates = parsed_dates.dropna()
    duplicate_timestamps_count = int(non_null_dates.duplicated().sum())
    unsorted_timestamps = bool(not non_null_dates.is_monotonic_increasing)

    issues = []

    if missing_date_count > 0:
        issues.append(
            {
                "code": "missing_dates",
                "level": "error",
                "message": "В столбце даты есть пустые или некорректные значения.",
                "count": missing_date_count,
            }
        )

    if missing_value_count > 0:
        issues.append(
            {
                "code": "invalid_values",
                "level": "error",
                "message": "В столбце значений есть пустые или нечисловые значения.",
                "count": missing_value_count,
            }
        )

    if duplicate_rows_count > 0:
        issues.append(
            {
                "code": "duplicate_rows",
                "level": "warning",
                "message": "В датасете обнаружены полностью дублирующиеся строки.",
                "count": duplicate_rows_count,
            }
        )

    if duplicate_timestamps_count > 0:
        issues.append(
            {
                "code": "duplicate_timestamps",
                "level": "warning",
                "message": "В столбце даты обнаружены повторяющиеся временные метки.",
                "count": duplicate_timestamps_count,
            }
        )

    if unsorted_timestamps:
        issues.append(
            {
                "code": "unsorted_timestamps",
                "level": "warning",
                "message": "Временные метки не отсортированы по возрастанию.",
                "count": None,
            }
        )

    valid = not any(issue["level"] == "error" for issue in issues)

    return {
        "valid": valid,
        "rows_count": int(len(df)),
        "date_column": date_column,
        "value_column": value_column,
        "summary": {
            "missing_date_count": missing_date_count,
            "missing_value_count": missing_value_count,
            "duplicate_rows_count": duplicate_rows_count,
            "duplicate_timestamps_count": duplicate_timestamps_count,
            "unsorted_timestamps": unsorted_timestamps,
        },
        "issues": issues,
    }