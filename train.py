import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 detector on a local dataset YAML."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="datasets/helmet_data.yaml",
        help="Path to dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model checkpoint (e.g., yolov8n.pt).",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Directory where training outputs will be saved.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="helmet_50ep",
        help="Run name under the project directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on (cpu, 0, 0,1, etc.).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
