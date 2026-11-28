"""Reproduce both T03 notebooks, writing configuration and measured evidence."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from src.mlp_data import gaussian_grid, make_mlp_mnist_loaders
from src.mlp_models import GaussianMLP, MNISTMLP, parameter_count

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def provenance():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable (exported archive)"
    files = [Path(__file__), ROOT / "src/mlp_models.py", ROOT / "src/mlp_data.py", ROOT / "src/datasets.py"]
    return {"base_commit": commit, "python": platform.python_version(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": np.__version__, "platform": platform.platform(),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def synchronize(device):
    if device.type == "cuda": torch.cuda.synchronize()


def epoch_pass(model, loader, criterion, device, optimizer=None, classification=False):
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            if optimizer is not None:
                loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(x)
            if classification: correct += int((pred.argmax(1) == y).sum())
            count += len(x)
    return total_loss / count, correct / count if classification else None


def save_checkpoint(path, model, config, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k,v in model.state_dict().items()},
                "config": config, "metrics": metrics}, path)


def run_function(args, device):
    folder = ROOT / "results" / "T03_function"
    folder.mkdir(parents=True, exist_ok=True)
    x, y = gaussian_grid(32); xt, yt = gaussian_grid(10)
    # Preserve the original notebook's pool range(1,1001), although grid has 1024 rows.
    pool = np.arange(1, 1001)
    configs = [("samples", a, 16, 100) for a in range(100, 1001, 100)]
    configs += [("hidden", 800, b, 100) for b in [8,16,32,64,128,256,512]]
    configs += [("epochs", 800, 128, c) for c in range(10, 200, 10)]
    summary, all_history = [], []
    logs = []
    for sweep, n, hidden, epochs in configs:
        seed_everything(args.seed)
        indices = np.random.default_rng(args.seed).choice(pool, n, replace=False)
        loader = DataLoader(TensorDataset(x[indices], y[indices]), batch_size=128,
                            shuffle=True, generator=torch.Generator().manual_seed(args.seed))
        full = DataLoader(TensorDataset(x,y), batch_size=1024)
        test = DataLoader(TensorDataset(xt,yt), batch_size=100)
        model = GaussianMLP(hidden).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        synchronize(device); start = time.perf_counter()
        experiment = f"{sweep}_A{n}_B{hidden}_C{epochs}"
        history = []
        for epoch in range(1, epochs+1):
            train_loss, _ = epoch_pass(model, loader, criterion, device, optimizer)
            test_loss, _ = epoch_pass(model, test, criterion, device)
            history.append({"experiment_id": experiment, "epoch": epoch,
                            "train_loss": train_loss, "test_loss": test_loss})
        synchronize(device); elapsed = time.perf_counter() - start
        full_loss, _ = epoch_pass(model, full, criterion, device)
        test_loss, _ = epoch_pass(model, test, criterion, device)
        with torch.no_grad():
            predictions = model(xt.to(device)).cpu().numpy().ravel()
        row = {"experiment_id": experiment, "sweep": sweep, "samples": n,
               "hidden": hidden, "epochs": epochs, "params": parameter_count(model),
               "train_grid_mse": full_loss, "test_mse": test_loss,
               "test_mae": float(np.mean(np.abs(predictions-yt.numpy().ravel()))),
               "training_seconds": elapsed, "seconds_per_epoch": elapsed/epochs}
        summary.append(row); all_history.extend(history)
        message = f"{experiment}: train_grid_mse={full_loss:.8f} test_mse={test_loss:.8f} seconds={elapsed:.2f}"
        logs.append(message); print(message, flush=True)
        if sweep == "epochs" and epochs == 100:
            config = {"task": "function", "hidden": hidden, "samples": n, "epochs": epochs,
                      "batch_size": 128, "learning_rate": 0.01, "optimizer": "SGD",
                      "loss": "MSELoss", "seed": args.seed, "device": str(device)}
            save_checkpoint(ROOT / "checkpoints/T03_function.pth", model, config, row)
            write_csv(folder / "predictions.csv", [{"x1": float(xx[0]), "x2": float(xx[1]),
                       "true": float(yy), "predicted": float(pp)} for xx,yy,pp in zip(xt.numpy(),yt.numpy().ravel(),predictions)])
            write_csv(folder / "representative_history.csv", history)
            write_json(folder / "sample_indices.json", indices.tolist())
            plot_function(model, folder, device)
    write_csv(folder / "sweeps.csv", summary)
    write_csv(folder / "all_history.csv", all_history)
    (folder / "train.log").write_text("\n".join(logs)+"\n", encoding="utf-8")
    fig, axes = plt.subplots(1,3,figsize=(12,3.7))
    for ax,sweep,key,label in zip(axes,["samples","hidden","epochs"],["samples","hidden","epochs"],["Training samples A","Hidden units B","Epochs C"]):
        subset = [r for r in summary if r["sweep"] == sweep]
        ax.plot([r[key] for r in subset],[r["train_grid_mse"] for r in subset],"o-",label="Full train grid")
        ax.plot([r[key] for r in subset],[r["test_mse"] for r in subset],"s-",label="Test grid")
        ax.set(xlabel=label,ylabel="MSE"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(folder / "sensitivity.png",dpi=160); plt.close(fig)
    write_json(folder / "metadata.json", {**provenance(), "device": str(device), "seed": args.seed,
               "experiments": len(configs), "train_grid": [32,32], "test_grid": [10,10],
               "sampling_pool": "indices 1..1000 inclusive, matching source notebook",
               "representative_experiment": "epochs_A800_B128_C100",
               "total_training_seconds": sum(r["training_seconds"] for r in summary),
               "comparison_note": "Same seed and sample selection across B/C sweeps; original notebook did not fix a seed. Train-grid MSE includes unsampled rows. Test grid is monitored, not used for checkpoint selection."})


def plot_function(model, folder, device):
    x,y = gaussian_grid(32)
    with torch.no_grad(): pred = model(x.to(device)).cpu().numpy().reshape(32,32)
    fig = plt.figure(figsize=(10,4.2))
    for i,(z,title) in enumerate([(y.numpy().reshape(32,32),"True Gaussian"),(pred,"PyTorch MLP prediction")]):
        ax = fig.add_subplot(1,2,i+1,projection="3d")
        ax.plot_surface(x[:,0].numpy().reshape(32,32),x[:,1].numpy().reshape(32,32),z,cmap="viridis")
        ax.set(xlabel="x1",ylabel="x2",zlim=(0,0.17),title=title)
    fig.tight_layout(); fig.savefig(folder / "function_surfaces.png",dpi=160); plt.close(fig)


def run_mnist(args, device):
    seed_everything(args.seed)
    folder = ROOT / "results/T03_mnist"; folder.mkdir(parents=True,exist_ok=True)
    loaders = make_mlp_mnist_loaders(ROOT / "data/mnist", 128, args.seed)
    model = MNISTMLP().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    history, logs = [], []
    synchronize(device); start = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        synchronize(device); epoch_start = time.perf_counter()
        train_loss, train_acc = epoch_pass(model,loaders["train"],criterion,device,optimizer,True)
        test_loss, test_acc = epoch_pass(model,loaders["test"],criterion,device,classification=True)
        synchronize(device)
        history.append({"epoch": epoch, "train_loss": train_loss,"train_accuracy": train_acc,
                        "test_loss": test_loss,"test_accuracy": test_acc,
                        "seconds": time.perf_counter()-epoch_start})
        message = f"epoch={epoch:02d}/{args.epochs} train_loss={train_loss:.6f} train_acc={train_acc:.4%} test_loss={test_loss:.6f} test_acc={test_acc:.4%} seconds={history[-1]['seconds']:.2f}"
        logs.append(message); print(message,flush=True)
        write_csv(folder / "history.csv",history)
        (folder / "train.log").write_text("\n".join(logs)+"\n",encoding="utf-8")
    synchronize(device); elapsed = time.perf_counter()-start
    config = {"task":"mnist","epochs":args.epochs,"batch_size":128,"optimizer":"SGD",
              "learning_rate":0.01,"seed":args.seed,"dropout":0.2,"device":str(device),
              "normalization":"raw uint8 / 255 via local adapter of T02 dataset",
              "checkpoint_selection":"fixed final epoch; no best-test selection"}
    metrics = {"test_loss":test_loss,"test_accuracy":test_acc,"params":parameter_count(model),
               "training_seconds":elapsed,"seconds_per_epoch":elapsed/args.epochs,
               "train_samples":60000,"test_samples":10000}
    save_checkpoint(ROOT / "checkpoints/T03_mnist.pth",model,config,metrics)
    write_json(folder / "metrics.json",{**metrics,"config":config})
    write_json(folder / "metadata.json",provenance())
    predictions, labels = [], []
    model.eval()
    with torch.no_grad():
        for x,y in loaders["test"]:
            predictions.extend(model(x.to(device)).argmax(1).cpu().tolist()); labels.extend(y.tolist())
    write_csv(folder / "predictions.csv",[{"index":i,"true":y,"predicted":p} for i,(y,p) in enumerate(zip(labels,predictions))])
    cm = np.zeros((10,10),dtype=int)
    np.add.at(cm,(np.asarray(labels),np.asarray(predictions)),1)
    np.savetxt(folder / "confusion_matrix.csv",cm,delimiter=",",fmt="%d")
    fig,axes = plt.subplots(1,2,figsize=(10,3.6))
    for ax,keys,ylabel in [(axes[0],["train_loss","test_loss"],"Cross entropy"),(axes[1],["train_accuracy","test_accuracy"],"Accuracy")]:
        for key in keys: ax.plot([r['epoch'] for r in history],[r[key] for r in history],label=key)
        ax.set(xlabel="Epoch",ylabel=ylabel); ax.legend()
    fig.tight_layout(); fig.savefig(folder / "learning_curves.png",dpi=160); plt.close(fig)
    fig,ax = plt.subplots(figsize=(6,5))
    image = ax.imshow(cm,cmap="Blues"); fig.colorbar(image,ax=ax)
    ax.set(xticks=range(10),yticks=range(10),xlabel="Predicted",ylabel="True")
    for i in range(10):
        for j in range(10): ax.text(j,i,str(cm[i,j]),ha="center",va="center",fontsize=7,color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.tight_layout(); fig.savefig(folder / "confusion_matrix.png",dpi=160); plt.close(fig)
    data = loaders["test"].dataset.tensors[0]
    wrong = [i for i,(y,p) in enumerate(zip(labels,predictions)) if y != p]
    chosen = list(range(8))+wrong[:8]
    fig,axes=plt.subplots(2,8,figsize=(12,3.5))
    for ax,i in zip(axes.ravel(),chosen):
        ax.imshow(data[i,0],cmap="gray"); ax.set_title(f"True {labels[i]} / Pred {predictions[i]}",fontsize=9); ax.axis("off")
    fig.tight_layout(); fig.savefig(folder / "prediction_examples.png",dpi=160); plt.close(fig)
    print(json.dumps(metrics),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--task",choices=["all","function","mnist"],default="all")
    parser.add_argument("--epochs",type=int,default=30)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--download-only",action="store_true")
    args=parser.parse_args()
    if args.epochs < 1: parser.error("epochs must be positive")
    if args.device == "cuda" and not torch.cuda.is_available(): parser.error("CUDA unavailable")
    seed_everything(args.seed)
    if args.download_only:
        loaders = make_mlp_mnist_loaders(ROOT / "data/mnist",128,args.seed)
        print({k:len(v.dataset) for k,v in loaders.items()},flush=True); return
    device=torch.device(args.device)
    if args.task in ["function","all"]: run_function(args,device)
    if args.task in ["mnist","all"]: run_mnist(args,device)

if __name__ == "__main__": main()
