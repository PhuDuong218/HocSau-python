"""Kiểm tra nhanh cấu trúc pipeline mà không cần dữ liệu Kaggle."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pipeline import add_features, build_preprocessor

rng = np.random.default_rng(42)
n = 20
sample = pd.DataFrame({
    "Id": np.arange(1, n + 1),
    "MSSubClass": rng.integers(20, 120, n),
    "Neighborhood": rng.choice(["NAmes", "CollgCr", "OldTown"], n),
    "OverallQual": rng.integers(3, 10, n),
    "GrLivArea": rng.integers(700, 2500, n),
    "TotalBsmtSF": rng.integers(0, 1800, n),
    "1stFlrSF": rng.integers(500, 1800, n),
    "2ndFlrSF": rng.integers(0, 1000, n),
    "FullBath": rng.integers(1, 3, n),
    "HalfBath": rng.integers(0, 2, n),
    "BsmtFullBath": rng.integers(0, 2, n),
    "BsmtHalfBath": rng.integers(0, 2, n),
    "OpenPorchSF": rng.integers(0, 300, n),
    "EnclosedPorch": rng.integers(0, 300, n),
    "3SsnPorch": rng.integers(0, 100, n),
    "ScreenPorch": rng.integers(0, 200, n),
    "WoodDeckSF": rng.integers(0, 400, n),
    "YrSold": rng.integers(2007, 2011, n),
    "YearBuilt": rng.integers(1950, 2010, n),
    "YearRemodAdd": rng.integers(1950, 2010, n),
    "GarageYrBlt": rng.integers(1950, 2010, n).astype(float),
})

engineered = add_features(sample, use_engineered=True)
assert all(c in engineered.columns for c in ["TotalSF", "TotalBathrooms", "HouseAge", "OverallQual_TotalSF"])
pre = build_preprocessor(engineered, scale_numeric=True)
mat = pre.fit_transform(engineered)
assert np.isfinite(mat).all()
print("SMOKE TEST PASSED")
print("Input columns:", engineered.shape[1])
print("Transformed columns:", mat.shape[1])
