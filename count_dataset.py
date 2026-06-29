from pathlib import Path
from collections import defaultdict

SOURCE_ROOT = Path('/content/drive/MyDrive/CIVICLEAR DATASETS')  # change for local use
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
CLASS_FOLDERS = [
    'construction materials',
    'garbage debris',
    'illegal parking',
    'no violation',
    'road obstruction',
    'sidewalk obstruction',
]
SPLITS = ['train', 'valid', 'val', 'test']

def count_images_under_split(class_path: Path, split: str) -> int:
    images = []
    for ext in IMAGE_EXTENSIONS:
        images += list(class_path.rglob(f'{split}/**/*{ext}'))
        images += list(class_path.rglob(f'{split}/**/*{ext.upper()}'))
    return len(set(images))

def main():
    print(f'SOURCE_ROOT: {SOURCE_ROOT}')
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f'Missing dataset folder: {SOURCE_ROOT}')

    print('\nRecursive image count by class:')
    print('| Class | Train | Valid/Val | Test |')
    print('|---|---:|---:|---:|')

    for class_name in CLASS_FOLDERS:
        class_path = SOURCE_ROOT / class_name
        if not class_path.exists():
            print(f'| {class_name} | MISSING | MISSING | MISSING |')
            continue
        train = count_images_under_split(class_path, 'train')
        valid = count_images_under_split(class_path, 'valid') + count_images_under_split(class_path, 'val')
        test = count_images_under_split(class_path, 'test')
        print(f'| {class_name} | {train} | {valid} | {test} |')

if __name__ == '__main__':
    main()
