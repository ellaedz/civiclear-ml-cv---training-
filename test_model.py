import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Test CIVICLEAR YOLO model on test set and save predictions.')
    parser.add_argument('--weights', type=Path, default=Path('/content/drive/MyDrive/CIVICLEAR_TRAINING_RUNS/civiclear_yolov8n/weights/best.pt'))
    parser.add_argument('--data', type=Path, default=Path('/content/drive/MyDrive/CIVICLEAR_MERGED_YOLO/data.yaml'))
    parser.add_argument('--source', type=Path, default=Path('/content/drive/MyDrive/CIVICLEAR_MERGED_YOLO/test/images'))
    parser.add_argument('--project', type=Path, default=Path('/content/drive/MyDrive/CIVICLEAR_TRAINING_RUNS'))
    parser.add_argument('--name', type=str, default='sample_predictions')
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f'Missing weights: {args.weights}')

    model = YOLO(str(args.weights))
    metrics = model.val(data=str(args.data), split='test', imgsz=640)
    print('mAP50:', metrics.box.map50)
    print('mAP50-95:', metrics.box.map)
    print('Precision:', metrics.box.mp)
    print('Recall:', metrics.box.mr)

    model.predict(
        source=str(args.source),
        conf=0.25,
        save=True,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )
    print('Predictions saved to:', args.project / args.name)

if __name__ == '__main__':
    main()
