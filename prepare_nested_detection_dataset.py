import argparse
import json
import random
import shutil
from collections import defaultdict, Counter
from pathlib import Path
import yaml

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']

# Parent Google Drive folder names mapped to YOLO class names.
# no violation is background only, so it is not a detected class.
CLASS_FOLDER_MAP = {
    'construction materials': 'construction_materials',
    'garbage debris': 'garbage_debris',
    'illegal parking': 'illegal_parking',
    'road obstruction': 'road_obstruction',
    'sidewalk obstruction': 'sidewalk_obstruction',
    'no violation': None,
}

GLOBAL_CLASSES = [
    'construction_materials',
    'garbage_debris',
    'illegal_parking',
    'road_obstruction',
    'sidewalk_obstruction',
]

SPLIT_ALIASES = {
    'train': ['train'],
    'valid': ['valid', 'val'],
    'test': ['test'],
}


def slugify(text: str) -> str:
    return text.lower().replace(' ', '_').replace('-', '_').replace('.', '_').replace('/', '_')


def ensure_output_dirs(output_root: Path):
    for split in ['train', 'valid', 'test']:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)


def list_images(folder: Path):
    if not folder.exists():
        return []
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(folder.rglob(f'*{ext}'))
        images.extend(folder.rglob(f'*{ext.upper()}'))
    return sorted(set([p for p in images if p.is_file()]))


def find_split_dirs(class_root: Path, split_key: str):
    split_names = SPLIT_ALIASES[split_key]
    found = []
    for split_name in split_names:
        found.extend([p for p in class_root.rglob(split_name) if p.is_dir() and p.name == split_name])
    # Avoid nested duplicates if a train folder contains another train folder somehow
    return sorted(set(found))


def find_coco_json(split_dir: Path):
    candidates = [
        split_dir / '_annotations.coco.json',
        split_dir / '_annotations.json',
        split_dir / 'annotations.coco.json',
        split_dir / 'annotations.json',
    ]
    for c in candidates:
        if c.exists():
            return c
    coco_files = list(split_dir.rglob('*coco*.json')) + list(split_dir.rglob('*annotations*.json'))
    return coco_files[0] if coco_files else None


def copy_image_and_label(image_path: Path, label_lines, output_root: Path, target_split: str, prefix: str):
    images_dir = output_root / target_split / 'images'
    labels_dir = output_root / target_split / 'labels'

    safe_stem = f'{prefix}_{image_path.stem}'
    new_image = images_dir / f'{safe_stem}{image_path.suffix.lower()}'
    new_label = labels_dir / f'{safe_stem}.txt'

    counter = 1
    while new_image.exists() or new_label.exists():
        safe_stem = f'{prefix}_{image_path.stem}_{counter}'
        new_image = images_dir / f'{safe_stem}{image_path.suffix.lower()}'
        new_label = labels_dir / f'{safe_stem}.txt'
        counter += 1

    shutil.copy2(image_path, new_image)
    new_label.write_text(('\n'.join(label_lines).strip() + '\n') if label_lines else '')


def coco_bbox_to_yolo(bbox, width, height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    return x_center, y_center, w / width, h / height


def process_coco_split(split_dir: Path, output_root: Path, target_split: str, folder_label, prefix: str):
    coco_path = find_coco_json(split_dir)
    if coco_path is None:
        return 0, 0, False

    with open(coco_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    image_by_id = {img['id']: img for img in coco.get('images', [])}
    anns_by_image = defaultdict(list)
    for ann in coco.get('annotations', []):
        anns_by_image[ann.get('image_id')].append(ann)

    image_folder = split_dir / 'images' if (split_dir / 'images').exists() else split_dir
    class_id = GLOBAL_CLASSES.index(folder_label) if folder_label is not None else None

    copied = 0
    boxes = 0

    for image_id, info in image_by_id.items():
        file_name = info.get('file_name')
        width = info.get('width')
        height = info.get('height')
        if not file_name or not width or not height:
            continue

        image_path = image_folder / file_name
        if not image_path.exists():
            matches = list(image_folder.rglob(Path(file_name).name))
            if not matches:
                continue
            image_path = matches[0]

        label_lines = []
        if folder_label is not None:
            for ann in anns_by_image.get(image_id, []):
                bbox = ann.get('bbox')
                if not bbox or len(bbox) != 4:
                    continue
                x, y, w, h = coco_bbox_to_yolo(bbox, width, height)
                if w <= 0 or h <= 0:
                    continue
                # Force all nested COCO categories under this parent folder into the parent violation class.
                label_lines.append(f'{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
                boxes += 1

        copy_image_and_label(image_path, label_lines, output_root, target_split, prefix)
        copied += 1

    return copied, boxes, True


def process_yolo_split(split_dir: Path, output_root: Path, target_split: str, folder_label, prefix: str):
    images_dir = split_dir / 'images'
    labels_dir = split_dir / 'labels'
    if not images_dir.exists():
        return 0, 0, False

    images = [p for p in list_images(images_dir) if labels_dir not in p.parents]
    if not images:
        return 0, 0, False

    class_id = GLOBAL_CLASSES.index(folder_label) if folder_label is not None else None
    copied = 0
    boxes = 0

    for image_path in images:
        label_lines = []
        if folder_label is not None:
            old_label = labels_dir / f'{image_path.stem}.txt'
            if old_label.exists():
                for line in old_label.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        label_lines.append(f'{class_id} {" ".join(parts[1:5])}')
                        boxes += 1
        copy_image_and_label(image_path, label_lines, output_root, target_split, prefix)
        copied += 1

    return copied, boxes, True


def create_data_yaml(output_root: Path):
    data = {
        'path': str(output_root.resolve()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'names': {i: name for i, name in enumerate(GLOBAL_CLASSES)},
    }
    path = output_root / 'data.yaml'
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, sort_keys=False)
    return path


def summarize(output_root: Path):
    summary = {}
    for split in ['train', 'valid', 'test']:
        labels_dir = output_root / split / 'labels'
        images_dir = output_root / split / 'images'
        counters = Counter()
        empty = 0
        images = len(list_images(images_dir))
        for label_file in labels_dir.glob('*.txt'):
            lines = label_file.read_text().strip().splitlines()
            if not lines:
                empty += 1
                continue
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    cid = int(float(parts[0]))
                    counters[GLOBAL_CLASSES[cid]] += 1
        summary[split] = {'images': images, 'empty_labels': empty, 'boxes': dict(counters)}
    return summary


def maybe_create_valid_from_train(output_root: Path, ratio=0.15, seed=42):
    valid_images_dir = output_root / 'valid' / 'images'
    if list_images(valid_images_dir):
        return
    train_images_dir = output_root / 'train' / 'images'
    train_labels_dir = output_root / 'train' / 'labels'
    valid_labels_dir = output_root / 'valid' / 'labels'
    train_images = list_images(train_images_dir)
    if len(train_images) < 10:
        return
    random.seed(seed)
    random.shuffle(train_images)
    move_count = max(1, int(len(train_images) * ratio))
    for image_path in train_images[:move_count]:
        label_path = train_labels_dir / f'{image_path.stem}.txt'
        shutil.move(str(image_path), str(valid_images_dir / image_path.name))
        if label_path.exists():
            shutil.move(str(label_path), str(valid_labels_dir / label_path.name))


def main():
    parser = argparse.ArgumentParser(description='Prepare nested Roboflow/COCO/YOLO datasets for CIVICLEAR YOLO detection training.')
    parser.add_argument('--source-root', type=Path, default=Path('/content/drive/MyDrive/CIVICLEAR DATASETS'))
    parser.add_argument('--output-root', type=Path, default=Path('/content/drive/MyDrive/CIVICLEAR_MERGED_YOLO'))
    parser.add_argument('--reset', action='store_true', help='Delete output folder before rebuilding.')
    parser.add_argument('--make-valid-if-missing', action='store_true', help='If valid is empty, move 15% of train into valid.')
    args = parser.parse_args()

    if not args.source_root.exists():
        raise FileNotFoundError(f'Missing source folder: {args.source_root}')
    if args.reset and args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_output_dirs(args.output_root)

    totals = defaultdict(lambda: defaultdict(int))

    for folder_name, folder_label in CLASS_FOLDER_MAP.items():
        class_root = args.source_root / folder_name
        if not class_root.exists():
            print(f'WARNING: Missing parent class folder: {class_root}')
            continue

        print(f'\n=== {folder_name} -> {folder_label or "background/no_violation"} ===')
        prefix_base = slugify(folder_name)
        for target_split in ['train', 'valid', 'test']:
            split_dirs = find_split_dirs(class_root, target_split)
            if not split_dirs:
                print(f'  {target_split}: no split folders found')
                continue

            for idx, split_dir in enumerate(split_dirs):
                prefix = f'{prefix_base}_{idx}'
                copied, boxes, ok = process_yolo_split(split_dir, args.output_root, target_split, folder_label, prefix)
                fmt = 'YOLO' if ok else None
                if not ok:
                    copied, boxes, ok = process_coco_split(split_dir, args.output_root, target_split, folder_label, prefix)
                    fmt = 'COCO' if ok else None
                if ok:
                    totals[folder_name][target_split] += copied
                    print(f'  {target_split}: {fmt} {split_dir} -> {copied} images, {boxes} boxes')
                else:
                    print(f'  {target_split}: skipped {split_dir} (no readable COCO/YOLO annotations)')

    if args.make_valid_if_missing:
        maybe_create_valid_from_train(args.output_root)

    yaml_path = create_data_yaml(args.output_root)
    print('\nCreated:', yaml_path)

    print('\nMerged image totals by parent folder:')
    for folder_name in CLASS_FOLDER_MAP:
        print(folder_name, dict(totals[folder_name]))

    print('\nFinal YOLO summary:')
    final_summary = summarize(args.output_root)
    for split, info in final_summary.items():
        print(split, info)

if __name__ == '__main__':
    main()
