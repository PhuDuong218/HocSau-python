from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from models import predict_mlp, sklearn_model, train_mlp  # noqa: E402
from pipeline import (  # noqa: E402
    add_features,
    apply_log_features,
    build_preprocessor,
    get_feature_names,
    load_data,
    make_variant,
    seed_everything,
)

RESULTS = ROOT / "results"
MODELS = ROOT / "models"
DATA = ROOT / "data"
RESULTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


def rmsle_from_log(y_true_log, y_pred_log):
    return float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))


def summarize(y_true_log, y_pred_log):
    return {
        "RMSLE": rmsle_from_log(y_true_log, y_pred_log),
        "MAE_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "RMSE_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
    }


def plot_history(history, path):
    df = pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss"])
    df.to_csv(RESULTS / "mlp_history.csv", index=False)
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["epoch"], df["train_loss"], label="Train")
    plt.plot(df["epoch"], df["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE trên log1p(SalePrice)")
    plt.title("MLP learning curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    seed_everything(42)
    train_raw, test_raw = load_data(DATA)
    print("Train shape:", train_raw.shape)
    print("Test shape:", test_raw.shape)

    rows = []
    predictions = {}

    # ------------------------------------------------------------------
    # 1. Thực nghiệm đặc trưng cho RandomForest (scikit-learn)
    # ------------------------------------------------------------------
    variants = ["baseline", "engineered", "engineered_log", "selected"]
    rf_objects = {}
    for variant in variants:
        print(f"\n===== SKLEARN / {variant.upper()} =====")
        X, y_log, X_test, dropped = make_variant(train_raw, test_raw, variant)

        if variant == "engineered_log":
            X_log, log_cols = apply_log_features(X, X)
            X_test_log, _ = apply_log_features(X_test, X)
            X, X_test = X_log, X_test_log
        else:
            log_cols = []

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y_log, test_size=0.20, random_state=42
        )
        pre = build_preprocessor(X_tr, scale_numeric=False)
        Xtr_np = pre.fit_transform(X_tr)
        Xva_np = pre.transform(X_va)
        Xt_np = pre.transform(X_test)
        model = sklearn_model("random_forest")
        model.fit(Xtr_np, y_tr)
        pred_va = model.predict(Xva_np)
        metrics = summarize(y_va, pred_va)
        print(metrics)
        rows.append({
            "Model": "RandomForestRegressor",
            "Variant": variant,
            **metrics,
            "n_raw_features": X.shape[1],
            "n_transformed_features": Xtr_np.shape[1],
            "dropped_sparse": ", ".join(dropped),
            "log_features": ", ".join(log_cols),
        })

        # Lưu mô hình của từng biến thể để kiểm tra/reuse.
        joblib.dump({"preprocessor": pre, "model": model, "variant": variant}, MODELS / f"rf_{variant}.joblib")
        rf_objects[variant] = (pre, model, Xt_np)

        if variant == "engineered_log":
            predictions["sklearn"] = np.expm1(model.predict(Xt_np))

    # ------------------------------------------------------------------
    # 2. MLP PyTorch trên biến thể engineered_log
    # ------------------------------------------------------------------
    print("\n===== PYTORCH MLP / ENGINEERED_LOG =====")
    X, y_log, X_test, dropped = make_variant(train_raw, test_raw, "engineered_log")
    X, log_cols = apply_log_features(X, X)
    X_test, _ = apply_log_features(X_test, X)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y_log, test_size=0.20, random_state=42
    )
    pre = build_preprocessor(X_tr, scale_numeric=True)
    Xtr_np = pre.fit_transform(X_tr).astype(np.float32)
    Xva_np = pre.transform(X_va).astype(np.float32)
    Xt_np = pre.transform(X_test).astype(np.float32)
    model, history, device = train_mlp(Xtr_np, y_tr, Xva_np, y_va)
    pred_va = predict_mlp(model, Xva_np, device)
    metrics = summarize(y_va, pred_va)
    print(metrics)
    rows.append({
        "Model": "PyTorch MLP",
        "Variant": "engineered_log",
        **metrics,
        "n_raw_features": X.shape[1],
        "n_transformed_features": Xtr_np.shape[1],
        "dropped_sparse": ", ".join(dropped),
        "log_features": ", ".join(log_cols),
    })

    torch_path = MODELS / "mlp_engineered_log.pth"
    import torch
    torch.save({"model_state_dict": model.state_dict(), "input_dim": Xtr_np.shape[1]}, torch_path)
    joblib.dump(pre, MODELS / "mlp_preprocessor.joblib")
    plot_history(history, RESULTS / "mlp_learning_curve.png")
    predictions["mlp"] = np.expm1(predict_mlp(model, Xt_np, device))

    # ------------------------------------------------------------------
    # 3. Feature importance của RF engineered_log
    # ------------------------------------------------------------------
    pre_rf, model_rf, Xt_rf = rf_objects["engineered_log"]
    feature_names = get_feature_names(pre_rf)
    importance = pd.Series(model_rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
    importance.sort_values().plot(kind="barh", figsize=(8, 6))
    plt.xlabel("Feature importance")
    plt.title("Top 20 features - Random Forest")
    plt.tight_layout()
    plt.savefig(RESULTS / "rf_top20_features.png", dpi=160)
    plt.close()
    importance.rename("importance").to_csv(RESULTS / "rf_top20_features.csv")

    # ------------------------------------------------------------------
    # 4. Save validation results and Kaggle submissions
    # ------------------------------------------------------------------
    result_df = pd.DataFrame(rows)
    result_df.to_csv(RESULTS / "results_summary.csv", index=False)
    result_df.to_json(RESULTS / "results_summary.json", orient="records", indent=2)

    sample_path = DATA / "sample_submission.csv"
    if sample_path.exists():
        sub = pd.read_csv(sample_path)
    else:
        sub = pd.DataFrame({"Id": test_raw["Id"].to_numpy()})
    sub["SalePrice"] = predictions["sklearn"]
    sub.to_csv(RESULTS / "submission_sklearn.csv", index=False)
    sub["SalePrice"] = predictions["mlp"]
    sub.to_csv(RESULTS / "submission_mlp.csv", index=False)

    # Model check theo tinh thần paper: dự đoán lại quan sát đầu tiên của train.
    X_check, y_check, _, _ = make_variant(train_raw, test_raw, "engineered_log")
    X_check, _ = apply_log_features(X_check, X_check)
    pre_check = build_preprocessor(X_check, scale_numeric=False)
    X_check_np = pre_check.fit_transform(X_check).astype(np.float32)
    first_pred_log = float(model_rf.predict(X_check_np[:1])[0])
    first_actual = float(train_raw.iloc[0]["SalePrice"])
    first_pred = float(np.expm1(first_pred_log))
    manual = {
        "first_observation_id": int(train_raw.iloc[0]["Id"]),
        "actual_saleprice": first_actual,
        "predicted_saleprice": first_pred,
        "absolute_error": abs(first_actual - first_pred),
        "note": "Đây là kiểm tra tự động để rà lỗi pipeline. Báo cáo nên trình bày phép tính tay theo yêu cầu giảng viên nếu đề bắt buộc.",
    }
    (RESULTS / "manual_check.txt").write_text(json.dumps(manual, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nHoàn tất. Xem results/results_summary.csv và hai file submission_*.csv")


if __name__ == "__main__":
    main()
