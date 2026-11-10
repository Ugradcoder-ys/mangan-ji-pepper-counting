"""RT-DETR の学習・推論スクリプト。

使用例:
    # 学習
    python detect/detect_rtdetr.py train --data path/to/data.yaml --output runs/rtdetr

    # 推論（画像フォルダ or 動画）
    python detect/detect_rtdetr.py predict --weights best.pt --source path/to/images --conf 0.25
"""

import argparse
import os


def train(args):
    from ultralytics import RTDETR

    print(f"\n--- Initializing RT-DETR Training: {args.model} ---")
    model = RTDETR(args.model)

    # バッチサイズ・画像サイズはGPUメモリに応じて調整
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.output,
        name=args.name,
        plots=True,
    )

    best_model_path = os.path.join(results.save_dir, "weights", "best.pt")
    print(f"Training completed. Best model saved at: {best_model_path}")


def predict(args):
    from ultralytics import RTDETR

    model = RTDETR(args.weights)

    # 結果（バウンディングボックス・クラス確率・ラベル）を保存
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
    parser = argparse.ArgumentParser(description="RT-DETR の学習・推論")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="RT-DETR を学習する")
    p_train.add_argument("--data", required=True, help="データセット設定ファイル (data.yaml)")
    p_train.add_argument("--model", default="rtdetr-l.pt", help="事前学習済みモデル (default: rtdetr-l.pt)")
    p_train.add_argument("--epochs", type=int, default=200, help="エポック数 (default: 200)")
    p_train.add_argument("--imgsz", type=int, default=640, help="入力画像サイズ (default: 640)")
    p_train.add_argument("--batch", type=int, default=8, help="バッチサイズ (default: 8)")
    p_train.add_argument("--output", default="runs/rtdetr", help="出力先ディレクトリ (default: runs/rtdetr)")
    p_train.add_argument("--name", default="rtdetr_experiment", help="実験名 (default: rtdetr_experiment)")
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict", help="学習済み重みで推論する")
    p_pred.add_argument("--weights", required=True, help="学習済み重み (.pt)")
    p_pred.add_argument("--source", required=True, help="入力（画像・画像フォルダ・動画のパス）")
    p_pred.add_argument("--conf", type=float, default=0.25, help="信頼度閾値 (default: 0.25)")
    p_pred.add_argument("--output", default="runs/rtdetr", help="出力先ディレクトリ (default: runs/rtdetr)")
    p_pred.add_argument("--name", default="inference_results", help="結果フォルダ名 (default: inference_results)")
    p_pred.set_defaults(func=predict)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.func(args)
