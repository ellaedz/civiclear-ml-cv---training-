import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Train CIVICLEAR YOLO detection model.')
    parser.add_argument('--data', type=Path, default=Path('/content/drive/MyDrive/CIVICLEAR_MERGED_YOLO/data.yaml'))
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--project', type=Path, default=Path('/content/drive/MyDrive/CIVICLEAR_TRAINING_RUNS'))
    parser.add_argument('--name', type=str, default='civiclear_yolov8n')
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f'Missing data.yaml: {args.data}')

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=15,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )

    print('Training complete.')
    print('Best model should be in:')
    print(args.project / args.name / 'weights' / 'best.pt')

if __name__ == '__main__':
    main()
