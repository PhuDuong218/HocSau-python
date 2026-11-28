"""Aggregate Keras vs PyTorch model configs and benchmark metrics.

Reads PyTorch outputs from `results/*/metrics.json` (including `EXP_01` and
`T03_mnist`) and `notebooks/04_segmentation.ipynb`, reads documented Keras
figures from `docs/CNN_CONVERSION.md` and `results/lstm_summation_seed42/metrics.json`,
and merges real Keras measurements from `results/keras_benchmark/keras_benchmark.json`
when available.

Exports `results/keras_benchmark/benchmark_t08_summary.csv`, renders
`results/plots/benchmark_t08_comparison.png`, and optionally updates
`docs/BENCHMARK_T08.md` with `--update-doc`.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_CONFIG_SPECS = {
    "Func_apxm_MLP": {
        "task_and_input": "Hồi quy Gaussian 2D (`2` → `1`, `[-1, 1]` theo EXP_01)",
        "architecture": "`Dense(16, sigmoid) → Dense(1, sigmoid)` (gốc `B=8..512`)",
        "loss": "MSE",
        "optimizer": "SGD (lr=0.01)",
        "batch": 32,
        "keras_epochs": "50",
        "pytorch_epochs": "50",
    },
    "MNIST_MLP": {
        "task_and_input": "Phân loại 10 lớp (`(28, 28)` / `[1, 28, 28]`, `[0, 1]`)",
        "architecture": "`Flatten → Dense(512, ReLU) → Drop(0.2) → Dense(512, ReLU) → Drop(0.2) → Dense(10)`",
        "loss": "CrossEntropy",
        "optimizer": "SGD (lr=0.01)",
        "batch": 128,
        "keras_epochs": "30",
        "pytorch_epochs": "30",
    },
    "MNIST_CNN": {
        "task_and_input": "Phân loại 10 lớp (`(28, 28, 1)` / `[1, 28, 28]`, `[0, 1]`)",
        "architecture": "`3×[Conv(64/128/256, 3×3, same, ReLU) → Pool(2) → Drop(0.2)] → Dense(512→512→10)`",
        "loss": "CrossEntropy",
        "optimizer": "Adam (lr=0.001)",
        "batch": 128,
        "keras_epochs": "4",
        "pytorch_epochs": "4",
    },
    "Human_Gender_CNN": {
        "task_and_input": "Phân loại 2 lớp (`(64, 64, 1)` / `[1, 64, 64]`, `rot90 k=3`, `[0, 255]`)",
        "architecture": "`5 khối Conv(16×2, 32×3, 64×3, 128×3, 256×3, same, ReLU) + Pool(2) + Drop(0.2) → Dense(1024→1024→2)`",
        "loss": "CrossEntropy",
        "optimizer": "Adam (lr=0.001) + ReduceLR(0.1, p=5) + EarlyStop(p=5)",
        "batch": 128,
        "keras_epochs": "30",
        "pytorch_epochs": "30",
    },
    "FER2013_CNN": {
        "task_and_input": "Phân loại 7 lớp (`(48, 48, 1)` / `[1, 48, 48]` + Flip, Rot ±45°, Trans 0.2)",
        "architecture": "`Conv(64s→64v) + Pool + Drop(0.25) → Conv(128s→128v→218s→218v) + Pool + Drop(0.25) → Dense(1024, Drop 0.5)→7`",
        "loss": "CrossEntropy",
        "optimizer": "Adam (lr=0.0001) + EarlyStop(p=5)",
        "batch": 128,
        "keras_epochs": "30",
        "pytorch_epochs": "1",
    },
    "MNIST_CNN_LSTM": {
        "task_and_input": "Hồi quy tổng 1..8 chữ số (`(8, 28, 28, 1)` / `[8, 1, 28, 28]`, `[0, 1]`)",
        "architecture": "`TimeDistributed(Conv 64→128→256 + Pool + Drop 0.2) → LSTM(128, seq) → LSTM(128) → Drop(0.22) → Dense(1)`",
        "loss": "MAE (L1Loss)",
        "optimizer": "SGD (lr=0.01)",
        "batch": 128,
        "keras_epochs": "30",
        "pytorch_epochs": "30",
    },
    "Hand_Depth_SegNet": {
        "task_and_input": "Phân đoạn 7 lớp (`(96, 96, 1)` / `[1, 96, 96]` → Mask LUT `[96, 96]`)",
        "architecture": "`Enc-Dec 3 tầng: Conv(64→128→256 + BN + ReLU + Pool) → Up + Conv(256→128→64 + BN + ReLU) → Conv1×1(7)`",
        "loss": "CrossEntropy",
        "optimizer": "Adam (lr=0.0001)",
        "batch": 8,
        "keras_epochs": "15",
        "pytorch_epochs": "15",
    },
}


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_segnet_notebook(nb_path):
    nb = _load_json(nb_path)
    text_chunks = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                text_chunks.append("".join(out.get("text", [])))
    full_text = "\n".join(text_chunks)

    params_m = re.search(r"Tổng số tham số trainable:\s*([\d,]+)", full_text)
    time_m = re.search(r"Thời gian huấn luyện:\s*([\d.]+)s", full_text)
    loss_m = re.search(r"Test Loss\s*:\s*([\d.]+)", full_text)
    acc_m = re.search(r"Test Pixel Accuracy\s*:\s*([\d.]+)%", full_text)
    device_m = re.search(r"Device:\s*([^\n\r]+)", full_text)

    epochs = 15
    total_sec = float(time_m.group(1))
    return {
        "parameters": int(params_m.group(1).replace(",", "")),
        "mean_epoch_seconds": total_sec / epochs,
        "test_loss": float(loss_m.group(1)),
        "test_accuracy": float(acc_m.group(1)),
        "epochs_completed": epochs,
        "batch_size": 8,
        "device": device_m.group(1).strip() if device_m else "cpu",
    }


def _parse_cnn_conversion_keras(doc_path):
    """Extract Keras validation accuracies recorded in docs/CNN_CONVERSION.md."""
    text = Path(doc_path).read_text(encoding="utf-8")
    m = re.search(
        r"98[.,]85%\s*\(MNIST\),\s*88[.,]17%\s*\(Gender[^)]*\)\s*và\s*61[.,]83%\s*\(FER2013",
        text,
    )
    if not m:
        return {}
    return {
        "MNIST_CNN": 98.85,
        "Human_Gender_CNN": 88.17,
        "FER2013_CNN": 61.83,
    }


def _fmt_num(val, suffix="", is_int=False):
    if val == "" or val is None:
        return "—"
    if is_int:
        return f"`{int(val):,}`"
    if isinstance(val, float) and suffix == "%":
        return f"`{val:.2f}%`"
    if isinstance(val, float) and suffix == "s":
        return f"`{val:.2f}s`"
    if isinstance(val, float):
        return f"`{val:.4f}`"
    return f"`{val}{suffix}`"


def _format_config_row(model_name, keras_models, pt_meta):
    spec = MODEL_CONFIG_SPECS[model_name]
    k_entry = keras_models.get(model_name, {})
    p_entry = pt_meta.get(model_name, {})

    k_ep = k_entry.get("epochs", spec["keras_epochs"])
    p_ep = p_entry.get("epochs", spec["pytorch_epochs"])
    k_bs = k_entry.get("batch_size") or spec["batch"]
    p_bs = p_entry.get("batch_size") or spec["batch"]
    bs_part = f"`bs={p_bs}`" if k_bs == p_bs else f"`bs=K:{k_bs}/PT:{p_bs}`"

    ep_part = f"K: `{k_ep} ep`, PT: `{p_ep} ep`"
    train_cfg = f"{spec['loss']} · {spec['optimizer']} · {bs_part} · {ep_part}"
    return {
        "model_name": model_name,
        "task_and_input": spec["task_and_input"],
        "architecture": spec["architecture"],
        "train_cfg": train_cfg,
    }


def _write_markdown(rows, out_path, keras_device=None, keras_models=None, pt_meta=None):
    keras_models = keras_models or {}
    pt_meta = pt_meta or {}

    lines = [
        "# Đối sánh kỹ thuật Keras vs PyTorch\n",
        "Tài liệu đối chiếu **cấu hình/kiến trúc mô hình** (Bảng 1) và **kết quả huấn luyện** (Bảng 2: **Loss**, **Acc / MAE / MSE**, **số params**, **thời gian train/epoch**) giữa **Keras** và **PyTorch**, lấy từ mã nguồn, cấu hình và kết quả đo trong `BTL03_Keras_to_PyTorch`.\n",
    ]
    if keras_device:
        lines.append(
            f"> **Nguồn đo thực nghiệm:** Số liệu **PyTorch** được tổng hợp từ kết quả thực nghiệm của cả 7 mô hình trong `results/` và `notebooks/04_segmentation.ipynb`; số liệu **Keras** ở Bảng 2 đọc từ `{keras_device}` (`results/keras_benchmark/keras_benchmark.json`).\n"
        )
    else:
        lines.append(
            "> **Nguồn đo thực nghiệm:** Số liệu **PyTorch** được tổng hợp từ `results/` và `notebooks/04_segmentation.ipynb`; số liệu **Keras** được lấy từ các cấu hình và kết quả đã xác nhận trong repository.\n"
        )

    lines.extend([
        "---\n",
        "## 1. Bảng so sánh cấu hình và kiến trúc mô hình (Keras vs PyTorch)\n",
        "| Mô hình | Bài toán & Đầu vào (Keras / PyTorch) | Kiến trúc mạng chính | Cấu hình huấn luyện (Loss, Optimizer, Batch, Epoch) |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for r in rows:
        cfg = _format_config_row(r["model_name"], keras_models, pt_meta)
        lines.append(
            f"| `{cfg['model_name']}` | {cfg['task_and_input']} | {cfg['architecture']} | {cfg['train_cfg']} |"
        )

    lines.extend([
        "\n---\n",
        "## 2. Bảng đối sánh trực tiếp theo cặp (Keras / PyTorch)\n",
        "| Mô hình | Loss (Keras / PyTorch) | Acc hoặc MAE/MSE (Keras / PyTorch) | Số params (Keras / PyTorch) | Thời gian train/epoch (Keras / PyTorch) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ])

    for r in rows:
        sfx = "%" if r["metric_label"] == "Acc (%)" else ""
        tag = (
            f" ({r['metric_label']})"
            if r["metric_label"] in {"MAE", "MSE"} and (r["keras_metric"] != "" or r["pytorch_metric"] != "")
            else ""
        )
        lines.append(
            f"| `{r['model_name']}` | {_fmt_num(r['keras_loss'])} / {_fmt_num(r['pytorch_loss'])} | "
            f"{_fmt_num(r['keras_metric'], sfx)} / {_fmt_num(r['pytorch_metric'], sfx)}{tag} | "
            f"{_fmt_num(r['keras_params'], is_int=True)} / {_fmt_num(r['pytorch_params'], is_int=True)} | "
            f"{_fmt_num(r['keras_epoch_sec'], 's')} / {_fmt_num(r['pytorch_epoch_sec'], 's')} |"
        )

    lines.extend([
        "\n---\n",
        "## 3. Nguồn trích xuất & Biểu đồ tổng hợp\n",
        "- **PyTorch:** [`results/EXP_01/metrics.json`](../results/EXP_01/metrics.json), [`results/T03_mnist/metrics.json`](../results/T03_mnist/metrics.json), [`results/mnist_cpu_seed42/metrics.json`](../results/mnist_cpu_seed42/metrics.json), [`results/gender_full_seed42/metrics.json`](../results/gender_full_seed42/metrics.json), [`results/fer2013_cpu_seed42/metrics.json`](../results/fer2013_cpu_seed42/metrics.json), [`results/lstm_summation_seed42/metrics.json`](../results/lstm_summation_seed42/metrics.json) và [`notebooks/04_segmentation.ipynb`](../notebooks/04_segmentation.ipynb).",
        "- **Keras:** Chạy thực nghiệm bằng [`notebooks/05_keras_benchmark_colab.ipynb`](../notebooks/05_keras_benchmark_colab.ipynb) trên Colab T4 (lưu vào [`results/keras_benchmark/keras_benchmark.json`](../results/keras_benchmark/keras_benchmark.json)) kết hợp [`docs/CNN_CONVERSION.md`](CNN_CONVERSION.md) và [`src/lstm_models.py`](../src/lstm_models.py).",
        "- **File CSV & script tái tạo:** [`results/keras_benchmark/benchmark_t08_summary.csv`](../results/keras_benchmark/benchmark_t08_summary.csv), [`scripts/build_benchmark_t08.py`](../scripts/build_benchmark_t08.py).\n",
        "![Biểu đồ đối sánh Keras vs PyTorch](../results/plots/benchmark_t08_comparison.png)\n",
    ])
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def build_benchmark(root_dir=Path("."), *, update_doc=False):
    root = Path(root_dir)
    exp01_pt = _load_json(root / "results/EXP_01/metrics.json")
    mlp_mnist_pt = _load_json(root / "results/T03_mnist/metrics.json")
    mnist_pt = _load_json(root / "results/mnist_cpu_seed42/metrics.json")
    gender_pt = _load_json(root / "results/gender_full_seed42/metrics.json")
    fer_pt = _load_json(root / "results/fer2013_cpu_seed42/metrics.json")
    lstm_pt = _load_json(root / "results/lstm_summation_seed42/metrics.json")
    seg_pt = _parse_segnet_notebook(root / "notebooks/04_segmentation.ipynb")
    keras_cnn_acc = _parse_cnn_conversion_keras(root / "docs/CNN_CONVERSION.md")

    keras_json_path = root / "results/keras_benchmark/keras_benchmark.json"
    keras_models = {}
    keras_device = None
    if keras_json_path.is_file():
        kdata = _load_json(keras_json_path)
        keras_models = kdata.get("models", {})
        keras_device = kdata.get("device")

    def _k(model_name, field, fallback=""):
        return keras_models.get(model_name, {}).get(field, fallback)

    pt_meta = {
        "Func_apxm_MLP": {
            "device": exp01_pt["config"].get("device"),
            "epochs": exp01_pt["config"].get("epochs"),
            "batch_size": exp01_pt["config"].get("batch_size"),
        },
        "MNIST_MLP": {
            "device": mlp_mnist_pt["config"].get("device"),
            "epochs": mlp_mnist_pt["config"].get("epochs"),
            "batch_size": mlp_mnist_pt["config"].get("batch_size"),
        },
        "MNIST_CNN": {
            "device": mnist_pt.get("device_name") or mnist_pt.get("device"),
            "epochs": mnist_pt.get("epochs_completed"),
            "batch_size": mnist_pt.get("batch_size"),
        },
        "Human_Gender_CNN": {
            "device": gender_pt.get("device_name") or gender_pt.get("device"),
            "epochs": gender_pt.get("epochs_completed"),
            "batch_size": gender_pt.get("batch_size"),
        },
        "FER2013_CNN": {
            "device": fer_pt.get("device_name") or fer_pt.get("device"),
            "epochs": fer_pt.get("epochs_completed"),
            "batch_size": fer_pt.get("batch_size"),
        },
        "MNIST_CNN_LSTM": {
            "device": lstm_pt.get("device_name") or lstm_pt.get("device"),
            "epochs": lstm_pt.get("epochs_completed"),
            "batch_size": lstm_pt.get("batch_size"),
        },
        "Hand_Depth_SegNet": {
            "device": seg_pt.get("device"),
            "epochs": seg_pt.get("epochs_completed"),
            "batch_size": seg_pt.get("batch_size"),
        },
    }

    rows = [
        {
            "model_name": "Func_apxm_MLP",
            "keras_loss": _k("Func_apxm_MLP", "loss"),
            "pytorch_loss": round(exp01_pt["test_mse"], 4),
            "metric_label": "MSE",
            "keras_metric": _k("Func_apxm_MLP", "metric"),
            "pytorch_metric": round(exp01_pt["test_mse"], 4),
            "keras_params": _k("Func_apxm_MLP", "params"),
            "pytorch_params": exp01_pt["params"],
            "keras_epoch_sec": _k("Func_apxm_MLP", "mean_epoch_seconds"),
            "pytorch_epoch_sec": round(exp01_pt["training_seconds"] / exp01_pt["config"]["epochs"], 2),
        },
        {
            "model_name": "MNIST_MLP",
            "keras_loss": _k("MNIST_MLP", "loss"),
            "pytorch_loss": round(mlp_mnist_pt["test_loss"], 4),
            "metric_label": "Acc (%)",
            "keras_metric": _k("MNIST_MLP", "metric"),
            "pytorch_metric": round(mlp_mnist_pt["test_accuracy"] * 100, 2),
            "keras_params": _k("MNIST_MLP", "params"),
            "pytorch_params": mlp_mnist_pt["params"],
            "keras_epoch_sec": _k("MNIST_MLP", "mean_epoch_seconds"),
            "pytorch_epoch_sec": round(mlp_mnist_pt["seconds_per_epoch"], 2),
        },
        {
            "model_name": "MNIST_CNN",
            "keras_loss": _k("MNIST_CNN", "loss"),
            "pytorch_loss": round(mnist_pt["test"]["loss"], 4),
            "metric_label": "Acc (%)",
            "keras_metric": _k("MNIST_CNN", "metric", keras_cnn_acc.get("MNIST_CNN", "")),
            "pytorch_metric": round(mnist_pt["test"]["accuracy"] * 100, 2),
            "keras_params": _k("MNIST_CNN", "params", mnist_pt["parameters"]),
            "pytorch_params": mnist_pt["parameters"],
            "keras_epoch_sec": _k("MNIST_CNN", "mean_epoch_seconds"),
            "pytorch_epoch_sec": round(mnist_pt["training_seconds"] / mnist_pt["epochs_completed"], 2),
        },
        {
            "model_name": "Human_Gender_CNN",
            "keras_loss": _k("Human_Gender_CNN", "loss"),
            "pytorch_loss": round(gender_pt["validation"]["loss"], 4),
            "metric_label": "Acc (%)",
            "keras_metric": _k("Human_Gender_CNN", "metric", keras_cnn_acc.get("Human_Gender_CNN", "")),
            "pytorch_metric": round(gender_pt["validation"]["accuracy"] * 100, 2),
            "keras_params": _k("Human_Gender_CNN", "params", gender_pt["parameters"]),
            "pytorch_params": gender_pt["parameters"],
            "keras_epoch_sec": _k("Human_Gender_CNN", "mean_epoch_seconds"),
            "pytorch_epoch_sec": round(gender_pt["training_seconds"] / gender_pt["epochs_completed"], 2),
        },
        {
            "model_name": "FER2013_CNN",
            "keras_loss": _k("FER2013_CNN", "loss"),
            "pytorch_loss": round(fer_pt["test"]["loss"], 4),
            "metric_label": "Acc (%)",
            "keras_metric": _k("FER2013_CNN", "metric", keras_cnn_acc.get("FER2013_CNN", "")),
            "pytorch_metric": round(fer_pt["test"]["accuracy"] * 100, 2),
            "keras_params": _k("FER2013_CNN", "params", fer_pt["parameters"]),
            "pytorch_params": fer_pt["parameters"],
            "keras_epoch_sec": _k("FER2013_CNN", "mean_epoch_seconds"),
            "pytorch_epoch_sec": round(fer_pt["training_seconds"] / fer_pt["epochs_completed"], 2),
        },
        {
            "model_name": "MNIST_CNN_LSTM",
            "keras_loss": _k("MNIST_CNN_LSTM", "loss"),
            "pytorch_loss": round(lstm_pt["test"]["loss"], 4),
            "metric_label": "MAE",
            "keras_metric": _k("MNIST_CNN_LSTM", "metric"),
            "pytorch_metric": round(lstm_pt["test"]["loss"], 4),
            "keras_params": _k("MNIST_CNN_LSTM", "params", lstm_pt["keras_equivalent_parameters"]),
            "pytorch_params": lstm_pt["parameters"],
            "keras_epoch_sec": _k("MNIST_CNN_LSTM", "mean_epoch_seconds"),
            "pytorch_epoch_sec": round(lstm_pt["mean_epoch_seconds"], 2),
        },
        {
            "model_name": "Hand_Depth_SegNet",
            "keras_loss": _k("Hand_Depth_SegNet", "loss"),
            "pytorch_loss": seg_pt["test_loss"],
            "metric_label": "Acc (%)",
            "keras_metric": _k("Hand_Depth_SegNet", "metric"),
            "pytorch_metric": seg_pt["test_accuracy"],
            "keras_params": _k("Hand_Depth_SegNet", "params"),
            "pytorch_params": seg_pt["parameters"],
            "keras_epoch_sec": _k("Hand_Depth_SegNet", "mean_epoch_seconds"),
            "pytorch_epoch_sec": round(seg_pt["mean_epoch_seconds"], 2),
        },
    ]

    fieldnames = [
        "model_name",
        "keras_loss", "pytorch_loss",
        "metric_label", "keras_metric", "pytorch_metric",
        "keras_params", "pytorch_params",
        "keras_epoch_sec", "pytorch_epoch_sec",
    ]
    csv_path = root / "results/keras_benchmark/benchmark_t08_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    _render_chart(rows, root / "results/plots/benchmark_t08_comparison.png")
    if update_doc:
        _write_markdown(
            rows,
            root / "docs/BENCHMARK_T08.md",
            keras_device=keras_device,
            keras_models=keras_models,
            pt_meta=pt_meta,
        )
        print(f"Updated Markdown: {root / 'docs/BENCHMARK_T08.md'}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Plot: {root / 'results/plots/benchmark_t08_comparison.png'}")


def _fmt_params_short(val):
    v = float(val)
    if v >= 10_000_000:
        return f"{v / 1e6:.1f}M"
    if v >= 1_000_000:
        return f"{v / 1e6:.2f}M"
    if v >= 10_000:
        return f"{round(v / 1e3)}K"
    return f"{int(v)}"


def _render_chart(rows, out_path):
    completed = [r for r in rows if r["pytorch_params"] != ""]
    acc_rows = [r for r in completed if r["metric_label"] == "Acc (%)"]
    has_keras_time = any(r["keras_epoch_sec"] != "" for r in completed)

    fig, axes = plt.subplots(1, 3, figsize=(20.0, 5.4))
    w = 0.34
    offset = 0.20

    # Panel 1: Accuracy (%) Keras vs PyTorch (chỉ 5 mô hình phân loại / phân đoạn)
    labels_acc = [r["model_name"].replace("_", "\n") for r in acc_rows]
    x1 = np.arange(len(labels_acc))
    k_acc = [float(r["keras_metric"]) if r["keras_metric"] != "" else 0.0 for r in acc_rows]
    p_acc = [float(r["pytorch_metric"]) for r in acc_rows]
    b1 = axes[0].bar(x1 - offset, k_acc, w, label="Keras", color="#4C78A8")
    b2 = axes[0].bar(x1 + offset, p_acc, w, label="PyTorch", color="#F58518")
    axes[0].set_xticks(x1)
    axes[0].set_xticklabels(labels_acc, fontsize=8.5)
    axes[0].set_ylim(0, 114)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Accuracy (%): 5 mô hình phân loại & phân đoạn")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    for idx, bar in enumerate(b1):
        h = bar.get_height()
        label = f"{h:.2f}%" if acc_rows[idx]["keras_metric"] != "" else "—"
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.5, label, ha="center", va="bottom", fontsize=7.0)
    for bar in b2:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.2f}%", ha="center", va="bottom", fontsize=7.0)

    # Panel 2: Parameter count (thang log để thấy rõ từ 65 tới 19.0M params)
    labels_all = [r["model_name"].replace("_", "\n") for r in completed]
    x2 = np.arange(len(labels_all))
    k_par = [float(r["keras_params"]) if r["keras_params"] != "" else 0.0 for r in completed]
    p_par = [float(r["pytorch_params"]) for r in completed]
    b3 = axes[1].bar(x2 - offset, k_par, w, label="Keras", color="#4C78A8")
    b4 = axes[1].bar(x2 + offset, p_par, w, label="PyTorch", color="#F58518")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels_all, fontsize=8)
    min_par = min(v for v in (k_par + p_par) if v > 0)
    max_par = max(k_par + p_par)
    axes[1].set_ylim(max(1.0, min_par * 0.2), max_par * 4.5)
    axes[1].set_ylabel("Số params (thang log)")
    axes[1].set_title("Số tham số (Params): Keras vs PyTorch")
    axes[1].grid(axis="y", which="both", alpha=0.3)
    axes[1].legend(fontsize=8)
    for idx, bar in enumerate(b3):
        h = bar.get_height()
        raw_k = completed[idx]["keras_params"]
        label = "—" if raw_k == "" else _fmt_params_short(raw_k)
        y_pos = h * 1.18 if h > 0 else max(1.0, min_par * 0.2) * 1.25
        axes[1].text(bar.get_x() + bar.get_width() / 2, y_pos, label, ha="center", va="bottom", fontsize=6.2)
    for idx, bar in enumerate(b4):
        h = bar.get_height()
        label = _fmt_params_short(completed[idx]["pytorch_params"])
        axes[1].text(bar.get_x() + bar.get_width() / 2, h * 1.18, label, ha="center", va="bottom", fontsize=6.2)

    # Panel 3: Training Time / Epoch (seconds)
    p_sec = [float(r["pytorch_epoch_sec"]) for r in completed]
    if has_keras_time:
        k_sec = [float(r["keras_epoch_sec"]) if r["keras_epoch_sec"] != "" else 0.0 for r in completed]
        b5 = axes[2].bar(x2 - offset, k_sec, w, label="Keras", color="#4C78A8")
        b6 = axes[2].bar(x2 + offset, p_sec, w, label="PyTorch", color="#F58518")
        axes[2].set_yscale("log")
        min_pos = min(v for v in (k_sec + p_sec) if v > 0)
        axes[2].set_ylim(min(0.5, min_pos * 0.4), max(k_sec + p_sec) * 2.8)
        for idx, bar in enumerate(b5):
            h = bar.get_height()
            if completed[idx]["keras_epoch_sec"] != "":
                lbl = f"{h:.2f}s" if h < 1.0 else f"{h:.1f}s"
                axes[2].text(bar.get_x() + bar.get_width() / 2, h * 1.15, lbl, ha="center", va="bottom", fontsize=6.8)
            else:
                axes[2].text(bar.get_x() + bar.get_width() / 2, min(0.5, min_pos * 0.4) * 1.25, "—", ha="center", va="bottom", fontsize=7)
        for bar in b6:
            h = bar.get_height()
            lbl = f"{h:.2f}s" if h < 1.0 else f"{h:.1f}s"
            axes[2].text(bar.get_x() + bar.get_width() / 2, h * 1.15, lbl, ha="center", va="bottom", fontsize=6.8)
        axes[2].set_title("Thời gian train / epoch: Keras vs PyTorch")
    else:
        b5 = axes[2].bar(x2, p_sec, width=0.5, color="#54A24B", label="PyTorch (đã đo)")
        axes[2].set_yscale("log")
        min_pos = min(v for v in p_sec if v > 0)
        axes[2].set_ylim(min(0.5, min_pos * 0.4), max(p_sec) * 2.5)
        for bar in b5:
            h = bar.get_height()
            axes[2].text(bar.get_x() + bar.get_width() / 2, h * 1.15, f"{h:.2f}s", ha="center", va="bottom", fontsize=7.5)
        axes[2].set_title("Thời gian train / epoch (PyTorch)")

    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(labels_all, fontsize=8)
    axes[2].set_ylabel("Thời gian / epoch (giây, thang log)")
    axes[2].grid(axis="y", which="both", alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.suptitle("Đối sánh Keras vs PyTorch (BTL03 — 7 mô hình)", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-doc", action="store_true", help="Regenerate docs/BENCHMARK_T08.md")
    args = parser.parse_args()
    build_benchmark(Path(__file__).resolve().parent.parent, update_doc=args.update_doc)
