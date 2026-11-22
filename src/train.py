"""Train the verified BaselineCNN, selecting weights using validation only.

Uses the verified dataset manifest, drops exact pixel overlap with held-out test,
and keeps identical train images in the same split to avoid duplicate leakage.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import random
import time
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .models import BaselineCNN
    from .dataset import CLASSES, ROOT
except ImportError:
    from models import BaselineCNN
    from dataset import CLASSES, ROOT


def write_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    os.replace(temp, path)


def make_splits(seed, val_fraction):
    with (ROOT/'results/data_audit/manifest.csv').open(encoding='utf-8') as f:
        records = list(csv.DictReader(f))
    # Recheck source-file hashes so the manifest cannot silently describe a
    # different collection than the images actually consumed by training.
    test_hashes = {r['pixel_sha256'] for r in records if r['original_split']=='test'}
    groups = {}
    for r in records:
        if r['original_split']=='train':
            groups.setdefault(r['pixel_sha256'], []).append(r)
    eligible = {key: group for key,group in groups.items()
                if key not in test_hashes and len({r['label'] for r in group})==1}
    keys = sorted(eligible)
    train_keys, val_keys = train_test_split(
        keys, test_size=val_fraction, random_state=seed,
        stratify=[eligible[key][0]['label'] for key in keys])
    train_keys, val_keys = set(train_keys), set(val_keys)
    assert not train_keys & val_keys
    assert not (train_keys | val_keys) & test_hashes
    for r in records:
        key = r['pixel_sha256']
        if r['original_split']=='test':
            r['assigned_split']='test'
        elif key in test_hashes:
            r['assigned_split']='excluded_test_pixel_overlap'
        elif key not in eligible:
            r['assigned_split']='excluded_conflicting_train_labels'
        else:
            r['assigned_split']='train' if key in train_keys else 'validation'
    return records


def load_tensors(rows):
    images = torch.empty((len(rows), 1, 48, 48), dtype=torch.uint8)
    labels = torch.tensor([int(r['label']) for r in rows], dtype=torch.long)
    # Reading this 60 MB verified archive once avoids tens of thousands of
    # filesystem opens on Windows; each member must still match audit hashes.
    with ZipFile(BytesIO((ROOT/'data/downloads/fer2013.zip').read_bytes())) as archive:
        for i,r in enumerate(rows):
            raw = archive.read(r['path'])
            if hashlib.sha256(raw).hexdigest() != r['file_sha256']:
                raise ValueError(f'Archive member changed after verification: {r["path"]}')
            with Image.open(BytesIO(raw)) as im:
                im.load()
                if im.mode!='L' or im.size!=(48,48):
                    raise ValueError(f'Invalid image: {r["path"]}')
                pixels = np.array(im, dtype=np.uint8, copy=True)
                if hashlib.sha256(pixels.tobytes()).hexdigest() != r['pixel_sha256']:
                    raise ValueError('Decoded pixels disagree with audit manifest')
                images[i,0]=torch.from_numpy(pixels)
    if set(labels.tolist()) != set(range(7)):
        raise ValueError('Every split must contain all seven labels.')
    return TensorDataset(images, labels)


def train(epochs=3, seed=42, batch_size=128, lr=0.001, val_fraction=0.15, threads=4):
    if epochs < 1 or batch_size < 1 or not 0 < val_fraction < 1 or lr<=0:
        raise ValueError('Invalid training arguments')
    run_dir = ROOT/f'results/training/baseline_seed{seed}'
    checkpoint = ROOT/'checkpoints/baseline_best.pth'
    if checkpoint.exists() or (run_dir/'history.csv').exists():
        raise FileExistsError('Existing checkpoint/training run found; refusing to overwrite.')
    audit = json.loads((ROOT/'results/data_audit/dataset_audit.json').read_text(encoding='utf-8'))
    if hashlib.sha256((ROOT/'data/downloads/fer2013.zip').read_bytes()).hexdigest() != audit['archive_sha256']:
        raise ValueError('Archive changed after dataset audit.')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    records = make_splits(seed, val_fraction)
    train_rows = [r for r in records if r['assigned_split']=='train']
    val_rows = [r for r in records if r['assigned_split']=='validation']
    print(f'Loading verified images: train={len(train_rows)}, validation={len(val_rows)}', flush=True)
    train_data, val_data = load_tensors(train_rows), load_tensors(val_rows)
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    val_loader=DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    with (run_dir/'split_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    counts={split:{c:sum(r['assigned_split']==split and r['class']==c for r in records) for c in CLASSES}
            for split in sorted({r['assigned_split'] for r in records})}
    metadata={
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'run_type':'new_training_run_by_XuanAnh_not_original_Nhi_checkpoint',
        'architecture':'BaselineCNN matching notebooks/02_baseline.ipynb',
        'class_to_idx':{c:i for i,c in enumerate(CLASSES)},
        'seed':seed, 'epochs_requested':epochs, 'batch_size':batch_size, 'learning_rate':lr,
        'optimizer':'Adam', 'criterion':'CrossEntropyLoss', 'device':'cpu', 'threads':threads,
        'torch_version':str(torch.__version__), 'train_counts':counts,
        'dataset_source':audit['source_url'], 'archive_sha256':audit['archive_sha256'],
        'selection_metric':'minimum validation cross-entropy', 'test_used_for_selection':False,
        'train_transforms':'grayscale uint8 to float32 /255; independent horizontal flip p=0.5 per image; normalize (x-0.5)/0.5',
        'validation_transforms':'grayscale uint8 to float32 /255; normalize (x-0.5)/0.5; no augmentation',
        'split_policy':'seeded stratified split of unique decoded-pixel groups from train; excludes train pixel overlap with test and conflicting train-label groups',
        'model_source_sha256':hashlib.sha256((ROOT/'src/models.py').read_bytes()).hexdigest(),
        'trainer_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'manifest_sha256':hashlib.sha256((run_dir/'split_manifest.csv').read_bytes()).hexdigest(),
    }
    write_json(run_dir/'run_metadata.json', metadata)
    model=BaselineCNN()
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    best_loss=float('inf'); history=[]; started=time.perf_counter()
    for epoch in range(1,epochs+1):
        epoch_start=time.perf_counter(); model.train()
        total_loss=0.0; correct=0; seen=0
        for batch,(images,labels) in enumerate(train_loader,1):
            x=images.float().div_(255.0)
            mask=torch.rand(len(x))<0.5
            x[mask]=x[mask].flip(-1)
            x=x.sub_(0.5).div_(0.5)
            optimizer.zero_grad(set_to_none=True)
            output=model(x); loss=F.cross_entropy(output,labels)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss; no checkpoint claimed.')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Non-finite gradient')
            optimizer.step()
            total_loss+=loss.item()*len(labels)
            correct+=(output.argmax(1)==labels).sum().item(); seen+=len(labels)
            if batch%25==0 or batch==len(train_loader):
                progress={'epoch':epoch,'batch':batch,'batches':len(train_loader),
                          'samples_seen':seen,'train_loss_so_far':total_loss/seen,
                          'elapsed_seconds':time.perf_counter()-started}
                write_json(run_dir/'progress.json',progress)
                print(f'Epoch {epoch}/{epochs} batch {batch}/{len(train_loader)} loss={total_loss/seen:.4f}',flush=True)
        model.eval(); val_loss=0.0; actual=[]; predicted=[]
        with torch.inference_mode():
            for images,labels in val_loader:
                output=model(images.float().div(255.0).sub(0.5).div(0.5))
                val_loss+=F.cross_entropy(output,labels,reduction='sum').item()
                actual.extend(labels.tolist()); predicted.extend(output.argmax(1).tolist())
        val_loss/=len(val_data)
        if not np.isfinite(val_loss):
            raise ValueError('Non-finite validation loss')
        row={'epoch':epoch,'train_loss':total_loss/seen,'train_accuracy':correct/seen,
             'val_loss':val_loss,'val_accuracy':float(np.mean(np.array(actual)==np.array(predicted))),
             'val_macro_f1':f1_score(actual,predicted,labels=list(range(7)),average='macro',zero_division=0),
             'seconds':time.perf_counter()-epoch_start}
        history.append(row)
        if val_loss < best_loss:
            best_loss=val_loss
            tmp=checkpoint.with_suffix('.pth.tmp'); torch.save(model.state_dict(),tmp); os.replace(tmp,checkpoint)
            metadata.update(best_epoch=epoch,best_val_loss=val_loss,best_val_accuracy=row['val_accuracy'],
                            best_val_macro_f1=row['val_macro_f1'],
                            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            write_json(checkpoint.with_suffix('.metadata.json'),metadata)
        with (run_dir/'history.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        print('EPOCH_RESULT '+json.dumps(row),flush=True)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    metadata.update(epochs_completed=epochs,elapsed_seconds=time.perf_counter()-started,
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    checkpoint_reload_verified=True)
    write_json(run_dir/'run_metadata.json',metadata)
    write_json(checkpoint.with_suffix('.metadata.json'),metadata)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for key,label in [('train_loss','Train'),('val_loss','Validation')]:
        axes[0].plot([r['epoch'] for r in history],[r[key] for r in history],marker='o',label=label)
    for key,label in [('train_accuracy','Train'),('val_accuracy','Validation')]:
        axes[1].plot([r['epoch'] for r in history],[r[key] for r in history],marker='o',label=label)
    for ax,title in zip(axes,['Cross-entropy','Accuracy']):
        ax.set(xlabel='Epoch',title=title,xticks=list(range(1,epochs+1)));ax.legend();ax.grid(alpha=0.25)
    fig.tight_layout();fig.savefig(run_dir/'learning_curve.png',dpi=180);plt.close(fig)
    print(f'TRAINING_COMPLETE {checkpoint} best_epoch={metadata["best_epoch"]}',flush=True)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--val-fraction',type=float,default=0.15)
    p.add_argument('--threads',type=int,default=4)
    args=p.parse_args()
    train(args.epochs,args.seed,args.batch_size,args.lr,args.val_fraction,args.threads)
