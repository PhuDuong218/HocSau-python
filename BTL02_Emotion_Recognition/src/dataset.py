"""
Module quản lý và tiền xử lý dữ liệu FER2013:
- Cung cấp transforms và DataLoaders cho huấn luyện/đánh giá các mô hình (Viet)
- Cung cấp công cụ kiểm định (audit), tính hash sha256 và tách manifest reproducible (XuanAnh)
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
EXPECTED = {
    'train': dict(zip(CLASSES, [3995, 436, 4097, 7215, 4965, 4830, 3171])),
    'test': dict(zip(CLASSES, [958, 111, 1024, 1774, 1233, 1247, 831])),
}


def get_transforms():
    """
    Trả về transforms cho tập train (có Augmentation) và tập test (chỉ chuẩn hóa)
    Dữ liệu FER2013 là ảnh mức xám (Grayscale 1 kênh), kích thước 48x48.
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    return train_transform, test_transform


def get_dataloaders(train_dir="../data/raw/train", test_dir="../data/raw/test", batch_size=128, num_workers=0):
    """
    Khởi tạo DataLoader cho cả train và test dựa trên cấu trúc thư mục ImageFolder.
    """
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.classes


def prepare(archive=None):
    """
    Xác thực kho lưu trữ nén fer2013.zip và tạo manifest dữ liệu kiểm định reproducible.
    """
    archive = Path(archive or ROOT / 'data/downloads/fer2013.zip')
    raw = ROOT / 'data/raw'
    report_dir = ROOT / 'results/data_audit'
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    with ZipFile(archive) as z:
        counts = {s: Counter() for s in EXPECTED}
        infos = z.infolist()
        for info in infos:
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) != 3 or parts[0] not in EXPECTED or parts[1] not in CLASSES:
                raise ValueError(f'Unexpected archive member: {info.filename}')
            target = (raw / info.filename).resolve()
            if not target.is_relative_to(raw.resolve()):
                raise ValueError('Unsafe archive path')
            counts[parts[0]][parts[1]] += 1
        if {s: dict(c) for s, c in counts.items()} != EXPECTED:
            raise ValueError(f'Class counts disagree with supplied slides: {counts}')
        bad = z.testzip()
        if bad:
            raise ValueError(f'ZIP CRC check failed: {bad}')
        raw.mkdir(parents=True, exist_ok=True)
        # Refuse to mix an existing image collection with the verified archive.
        expected_names = {i.filename for i in infos if not i.is_dir()}
        existing = {p.relative_to(raw).as_posix() for p in raw.rglob('*') if p.is_file()}
        if existing - expected_names:
            raise ValueError('data/raw has extra files; use a clean input directory.')
        z.extractall(raw)
    records = []
    pixels = defaultdict(list)
    for index, info in enumerate(infos):
        if info.is_dir():
            continue
        path = raw / info.filename
        with Image.open(path) as img:
            img.load()
            if img.size != (48, 48) or img.mode != 'L':
                raise ValueError(f'Expected 48x48 grayscale image: {path}, {img.size}, {img.mode}')
            pixel_hash = hashlib.sha256(img.tobytes()).hexdigest()
        split, label, filename = Path(info.filename).parts
        record = {'path': info.filename, 'original_split': split, 'class': label,
                  'label': CLASSES.index(label), 'pixel_sha256': pixel_hash,
                  'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        records.append(record)
        pixels[pixel_hash].append(record)
        if (index + 1) % 10000 == 0:
            print(f'Validated {index + 1} images', flush=True)
    cross = [group for group in pixels.values() if len({r['original_split'] for r in group}) > 1]
    conflicts = [group for group in pixels.values() if len({r['class'] for r in group}) > 1]
    public = sum('PublicTest_' in r['path'] for r in records if r['original_split'] == 'test')
    private = sum('PrivateTest_' in r['path'] for r in records if r['original_split'] == 'test')
    if public != 3589 or private != 3589:
        raise ValueError('Public/Private test filenames disagree with 3589/3589 split.')
    report = {
        'source_url': 'https://www.kaggle.com/datasets/msambare/fer2013',
        'download_url': 'https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013',
        'archive_sha256': archive_hash, 'archive_crc_check': 'passed',
        'image_validation': 'All images decoded; all grayscale L, 48x48',
        'class_to_idx': {name: i for i, name in enumerate(CLASSES)},
        'original_counts': EXPECTED, 'total_images': len(records),
        'test_public': public, 'test_private': private,
        'duplicate_pixel_groups': sum(len(group) > 1 for group in pixels.values()),
        'train_test_duplicate_pixel_groups': len(cross),
        'train_images_sharing_pixels_with_test': sum(r['original_split'] == 'train' for group in cross for r in group),
        'conflicting_label_pixel_groups': len(conflicts),
        'duplicate_definition': 'Exact SHA-256 match of decoded 48x48 grayscale pixels; not perceptual similarity',
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / 'manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (report_dir / 'dataset_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    prepare(parser.parse_args().archive)
