from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
CLEANED_DATA_PATH = PROCESSED_DIR / "diabetes_cleaned.csv"
TRANSFORMED_DATA_PATH = PROCESSED_DIR / "diabetes_transformed.csv"
OUTLIER_REPORT_PATH = REPORTS_DIR / "outlier_report.csv"

EXPECTED_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

VALIDITY_RULES = {
    "Pregnancies": lambda s: s >= 0,
    "Glucose": lambda s: s > 0,
    "BloodPressure": lambda s: s > 0,
    "SkinThickness": lambda s: s > 0,
    "Insulin": lambda s: s > 0,
    "BMI": lambda s: s > 0,
    "DiabetesPedigreeFunction": lambda s: s >= 0,
    "Age": lambda s: s > 0,
}

TRANSFORMATION_MAP = {
    "Insulin": "log1p",
    "DiabetesPedigreeFunction": "sqrt",
    "Age": "log1p",
}

RANDOM_STATE = 42
IQR_MULTIPLIER = 1.5


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    return pd.read_csv(path)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = (
        result.columns
        .str.strip()
        .str.replace("*", "", regex=False)
    )
    unnamed_columns = [
        column for column in result.columns
        if column.lower().startswith("unnamed")
    ]
    return result.drop(columns=unnamed_columns)


def validate_structure(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in EXPECTED_FEATURES
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    return df[EXPECTED_FEATURES].copy()


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in EXPECTED_FEATURES:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def replace_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in ZERO_AS_MISSING:
        result[column] = result[column].replace(0, np.nan)
    return result


def replace_invalid_values_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column, is_valid in VALIDITY_RULES.items():
        invalid_mask = result[column].notna() & ~is_valid(result[column])
        result.loc[invalid_mask, column] = np.nan
    return result


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def get_observed_bounds(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lower_bounds = df[EXPECTED_FEATURES].min().to_numpy()
    upper_bounds = df[EXPECTED_FEATURES].max().to_numpy()
    return lower_bounds, upper_bounds


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    lower_bounds, upper_bounds = get_observed_bounds(df)
    imputer = IterativeImputer(
        random_state=RANDOM_STATE,
        min_value=lower_bounds,
        max_value=upper_bounds,
    )
    imputed_values = imputer.fit_transform(df[EXPECTED_FEATURES])
    return pd.DataFrame(
        imputed_values,
        columns=EXPECTED_FEATURES,
        index=df.index,
    )


def validate_dataset(df: pd.DataFrame) -> None:
    if df.isna().sum().sum() != 0:
        raise ValueError("Dataset still contains missing values")
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError("Dataset contains infinite values")
    for column, is_valid in VALIDITY_RULES.items():
        if not is_valid(df[column]).all():
            raise ValueError(f"Invalid values found in column: {column}")


def apply_transformations(
    df: pd.DataFrame,
    transformation_map: dict[str, str] = TRANSFORMATION_MAP,
) -> pd.DataFrame:
    result = df.copy()
    for feature, transformation in transformation_map.items():
        if transformation == "log1p":
            result[feature] = np.log1p(result[feature])
        elif transformation == "sqrt":
            result[feature] = np.sqrt(result[feature])
        else:
            raise ValueError(f"Unknown transformation: {transformation}")
    return result


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for feature in EXPECTED_FEATURES:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        mask = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        records.append({
            "feature": feature,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": int(mask.sum()),
            "outlier_percentage": mask.mean() * 100,
        })
    return pd.DataFrame(records)


def save_processed_data(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def cleaning_pipeline(
    raw_path: Path = RAW_DATA_PATH,
    cleaned_path: Path = CLEANED_DATA_PATH,
    transformed_path: Path = TRANSFORMED_DATA_PATH,
    outlier_report_path: Path = OUTLIER_REPORT_PATH,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_raw_data(raw_path)
    df = clean_column_names(df)
    df = validate_structure(df)
    df = convert_to_numeric(df)
    df = replace_zeros_with_nan(df)
    df = replace_invalid_values_with_nan(df)
    df = remove_duplicates(df)
    df_clean = impute_missing_values(df)
    validate_dataset(df_clean)
    df_transformed = apply_transformations(df_clean)
    validate_dataset(df_transformed)
    outlier_report = detect_outliers(df_transformed)

    if save:
        save_processed_data(df_clean, cleaned_path)
        save_processed_data(df_transformed, transformed_path)
        save_processed_data(outlier_report, outlier_report_path)

    return df_clean, df_transformed, outlier_report


if __name__ == "__main__":
    cleaned, transformed, report = cleaning_pipeline()
    print(f"Cleaned dataset    : {cleaned.shape}")
    print(f"Transformed dataset: {transformed.shape}")
    print(report[["feature", "outlier_count", "outlier_percentage"]])