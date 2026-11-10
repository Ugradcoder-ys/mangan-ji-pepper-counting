"""YOLO 系モデル（v5 / v8 / 11 / 12）の比較実験スクリプト。

使用例:
    # 複数モデルを順に学習・検証
    python detect/detect_yolo.py train --data path/to/data.yaml --models yolov8m.pt yolo12m.pt

    # 推論（画像フォルダ or 動画）
    python detect/detect_yolo.py predict --weights best.pt --source path/to/images --conf 0.25
"""

import argparse
import os

# 比較対象のモデル（v5: n/m, v8 / 11 / 12: m）
DEFAULT_MODELS = [
    "yolov5nu.pt",
    "yolov5mu.pt",
    "yolov8m.pt",
    "yolo11m.pt",
    "yolo12m.pt",
]


def train_one(model_variant, args):
    """1つのモデルを学習し、検証指標を表示する。"""
    from ultralytics import YOLO

    print(f"\n--- Initializing Experiment: {model_variant} ---")
    model = YOLO(model_variant)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.output,
        name=model_variant.split(".")[0],
        exist_ok=True,
    )

    metrics = model.val()
    print(f"Metrics for {model_variant}:")
    print(f"mAP 50-95: {metrics.box.map:.4f}")
    print(f"mAP 50: {metrics.box.map50:.4f}")


def train(args):
    for variant in args.models:
        try:
            train_one(variant, args)
        except Exception as e:
            print(f"Failed to process {variant}: {e}")


def predict(args):
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        save_txt=True,
        project=args.output,
        name=args.name,
    )
    print(f"Inference completed. Results saved in: {os.path.join(args.output, args.name)}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO 系モデルの学習・比較・推論")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="指定したモデルを順に学習・検証する")
    p_train.add_argument("--data", required=True, help="データセット設定ファイル (data.yaml)")
    p_train.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                         help=f"学習するモデル (default: {' '.join(DEFAULT_MODELS)})")
    p_train.add_argument("--epochs", type=int, default=200, help="エポック数 (default: 200)")
    p_train.add_argument("--imgsz", type=int, default=640, help="入力画像サイズ (default: 640)")
    p_train.add_argument("--output", default="runs/yolo", help="出力先ディレクトリ (default: runs/yolo)")
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict", help="学習済み重みで推論する")
    p_pred.add_argument("--weights", required=True, help="学習済み重み (.pt)")
    p_pred.add_argument("--source", required=True, help="入力（画像・画像フォルダ・動画のパス）")
    p_pred.add_argument("--conf", type=float, default=0.25, help="信頼度閾値 (default: 0.25)")
    p_pred.add_argument("--output", default="runs/yolo", help="出力先ディレクトリ (default: runs/yolo)")
    p_pred.add_argument("--name", default="inference_results", help="結果フォルダ名 (default: inference_results)")
    p_pred.set_defaults(func=predict)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
