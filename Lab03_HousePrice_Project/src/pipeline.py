from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreparedData:
    X: pd.DataFrame
    y_log: np.ndarray
    test: pd.DataFrame
    test_ids: pd.Series


HIGH_MISSING_DEFAULT = 0.80

ENGINEERED_COLUMNS = [
    "TotalSF",
    "TotalBathrooms",
    "TotalPorchSF",
    "HouseAge",
    "RemodAge",
    "GarageAge",
    "Total_sqr_footage",
    "OverallQual_GrLivArea",
    "OverallQual_TotalSF",
]

SELECTED_BASE = [
    "MSSubClass", "MSZoning", "Neighborhood", "OverallQual", "OverallCond",
    "YearBuilt", "YearRemodAdd", "LotArea", "LotFrontage", "GrLivArea",
    "1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "GarageCars", "GarageArea",
    "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "KitchenQual", "BsmtQual", "GarageFinish", "SaleType", "SaleCondition",
]


def load_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Chưa có train.csv/test.csv trong thư mục data/. "
            "Hãy tải dữ liệu cuộc thi Kaggle và đặt hai file này vào data/."
        )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def add_features(df: pd.DataFrame, use_engineered: bool = True) -> pd.DataFrame:
    df = df.copy()

    # MSSubClass là mã loại công trình nên coi như biến phân loại.
    if "MSSubClass" in df.columns:
        df["MSSubClass"] = df["MSSubClass"].astype(str)

    if not use_engineered:
        return df

    def num(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
        return pd.Series(0.0, index=df.index)

    df["TotalSF"] = num("TotalBsmtSF") + num("1stFlrSF") + num("2ndFlrSF")
    df["TotalBathrooms"] = (
        num("FullBath") + 0.5 * num("HalfBath") +
        num("BsmtFullBath") + 0.5 * num("BsmtHalfBath")
    )
    df["TotalPorchSF"] = (
        num("OpenPorchSF") + num("EnclosedPorch") +
        num("3SsnPorch") + num("ScreenPorch") + num("WoodDeckSF")
    )
    df["HouseAge"] = (num("YrSold") - num("YearBuilt")).clip(lower=0)
    df["RemodAge"] = (num("YrSold") - num("YearRemodAdd")).clip(lower=0)
    df["GarageAge"] = (num("YrSold") - num("GarageYrBlt")).clip(lower=0)
    df["Total_sqr_footage"] = num("BsmtFinSF1") + num("BsmtFinSF2") + num("1stFlrSF") + num("2ndFlrSF")
    df["OverallQual_GrLivArea"] = num("OverallQual") * num("GrLivArea")
    df["OverallQual_TotalSF"] = num("OverallQual") * df["TotalSF"]
    return df


def drop_sparse_columns(train: pd.DataFrame, test: pd.DataFrame, threshold: float = HIGH_MISSING_DEFAULT):
    train = train.copy()
    test = test.copy()
    target = train["SalePrice"].copy()
    train_x = train.drop(columns=["SalePrice"])
    missing_ratio = train_x.isna().mean()
    sparse = missing_ratio[missing_ratio > threshold].index.tolist()

    # ID không mang ý nghĩa dự đoán trực tiếp.
    drop_cols = sorted(set(sparse + ["Id"]))
    train_x = train_x.drop(columns=drop_cols, errors="ignore")
    test_x = test.drop(columns=drop_cols, errors="ignore")
    return train_x, target, test_x, drop_cols


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")


def make_variant(train: pd.DataFrame, test: pd.DataFrame, variant: str):
    use_engineered = variant in {"engineered", "engineered_log", "selected"}
    train2 = add_features(train, use_engineered=use_engineered)
    test2 = add_features(test, use_engineered=use_engineered)

    if variant == "selected":
        available = [c for c in SELECTED_BASE + ENGINEERED_COLUMNS if c in train2.columns]
        train2 = train2[available + ["SalePrice"]]
        test2 = test2[available]

    train_x, y, test_x, dropped = drop_sparse_columns(train2, test2)
    y_log = np.log1p(y.to_numpy(dtype=float))
    return train_x, y_log, test_x, dropped


def apply_log_features(df: pd.DataFrame, X: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Log-transform positive-skewed numeric columns using the training frame's skewness.

    This function is intentionally simple for an undergraduate lab. The list of
    transformed columns is returned so the same transformation can be applied to test data.
    """
    out = df.copy()
    source = X if X is not None else df
    numeric_cols = source.select_dtypes(exclude=["object", "category"]).columns
    skewed = []
    for c in numeric_cols:
        s = pd.to_numeric(source[c], errors="coerce")
        if s.notna().sum() > 0 and s.min(skipna=True) >= 0 and abs(s.skew(skipna=True)) > 0.75:
            skewed.append(c)
            out[c] = np.log1p(pd.to_numeric(out[c], errors="coerce").clip(lower=0))
    return out, skewed


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names.extend(trans.get_feature_names_out(cols).tolist())
    return names
