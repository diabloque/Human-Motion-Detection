import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 inference on images, video, folder, or webcam."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/helmet_50ep/weights/best.pt",
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Inference source: file, directory, URL, or webcam index (e.g., 0).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs",
        help="Directory where prediction outputs are saved.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="predict",
        help="Run name under the project directory.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display predictions in a live window.",
    )
    return parser.parse_args()


def _normalize_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {model_path}. "
            "Train first or pass --model to an existing .pt file."
        )

    model = YOLO(str(model_path))
    model.predict(
        source=_normalize_source(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        show=args.show,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
