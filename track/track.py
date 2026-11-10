"""RT-DETR による検出 + 多物体追跡で、果実・花の個体数（ユニークID数）をカウントする。

トラッカーは --tracker で切り替える:
    bytetrack / botsort : Ultralytics 組み込みトラッカー
    boosttrack          : BoxMOT の BoostTrack
    deepsort            : deep_sort_realtime の DeepSORT

使用例:
    python track/track.py --video input.mp4 --weights rtdetr_best.pt --tracker boosttrack --conf 0.75
    python track/track.py --video input.mp4 --weights rtdetr_best.pt --tracker bytetrack --output out/bytetrack.mp4
"""

import argparse
import os

# 万願寺唐辛子のクラス定義と描画色 (BGR)
CLASS_NAMES = {0: "dried_flowers", 1: "flowers", 2: "fruits"}
COLORS = {0: (92, 113, 166), 1: (220, 220, 255), 2: (0, 255, 0)}

TRACKERS = ["bytetrack", "botsort", "boosttrack", "deepsort"]

# BoT-SORT のカスタム設定（カメラ動き補正・Re-ID は無効）
BOTSORT_CONFIG = {
    "tracker_type": "botsort",
    "gmc_method": "none",
    "track_high_thresh": 0.3,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.4,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "with_reid": False,
    "proximity_thresh": 0.5,
    "appearance_thresh": 0.25,
    "fuse_score": True,
}


def draw(frame, cv2, box, cls_id, track_id):
    x1, y1, x2, y2 = map(int, box)
    color = COLORS.get(cls_id, (255, 255, 255))
    label = f"{CLASS_NAMES[cls_id]} ID:{track_id}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def detect(model, frame, conf_threshold):
    """RT-DETR で検出し、閾値以上の [x1, y1, x2, y2, conf, cls_id] を返す。"""
    results = model(frame, verbose=False)[0]
    dets = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        if conf >= conf_threshold and cls_id in CLASS_NAMES:
            dets.append([x1, y1, x2, y2, conf, cls_id])
    return dets


def build_step(args, model, cv2):
    """トラッカーを初期化し、1フレームを処理して (box, track_id, cls_id) を返す関数を作る。"""
    if args.tracker in ("bytetrack", "botsort"):
        tracker_cfg = "bytetrack.yaml"
        if args.tracker == "botsort":
            import yaml

            tracker_cfg = os.path.join(os.path.dirname(args.output) or ".", "botsort_custom.yaml")
            with open(tracker_cfg, "w") as f:
                yaml.dump(BOTSORT_CONFIG, f)

        def step(frame):
            results = model.track(
                source=frame, tracker=tracker_cfg, persist=True, conf=args.conf, verbose=False
            )[0]
            if results.boxes is None or results.boxes.id is None:
                return []
            ids = results.boxes.id.cpu().numpy().astype(int)
            bboxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            return [(b, int(t), int(c)) for b, t, c in zip(bboxes, ids, classes)]

        return step

    if args.tracker == "boosttrack":
        import numpy as np
        import torch
        from pathlib import Path
        from boxmot import BoostTrack

        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tracker = BoostTrack(reid_weights=Path(args.reid_weights), device=device, half=False)

        def step(frame):
            dets = detect(model, frame, args.conf)
            dets_array = np.array(dets, dtype=np.float32) if dets else np.empty((0, 6))
            # 出力形式: [x1, y1, x2, y2, track_id, conf, cls_id, ind]
            tracks = tracker.update(dets_array, frame)
            tracks = np.asarray(tracks) if tracks is not None else np.empty((0, 8))
            return [(tr[0:4], int(tr[4]), int(tr[6])) for tr in tracks]

        return step

    if args.tracker == "deepsort":
        from deep_sort_realtime.deepsort_tracker import DeepSort

        tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)

        def step(frame):
            # DeepSORT 形式: ([left, top, w, h], confidence, class_id)
            dets = [([x1, y1, x2 - x1, y2 - y1], conf, cls_id)
                    for x1, y1, x2, y2, conf, cls_id in detect(model, frame, args.conf)]
            out = []
            for track in tracker.update_tracks(dets, frame=frame):
                if not track.is_confirmed():
                    continue
                cls_id = track.get_det_class()
                if cls_id is None:
                    continue
                out.append((track.to_ltrb(), track.track_id, int(cls_id)))
            return out

        return step

    raise ValueError(f"Unknown tracker: {args.tracker}")


def run(args):
    import cv2
    from ultralytics import RTDETR

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    model = RTDETR(args.weights)
    step = build_step(args, model, cv2)

    # クラスごとのユニークID集合（= カウント結果）
    unique_id_sets = {cid: set() for cid in CLASS_NAMES}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"動画を開けません: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        for box, track_id, cls_id in step(frame):
            if cls_id in CLASS_NAMES:
                unique_id_sets[cls_id].add(track_id)
                draw(frame, cv2, box, cls_id, track_id)
        writer.write(frame)

    cap.release()
    writer.release()

    print(f"\n--- Final Counts (Unique IDs, tracker={args.tracker}, conf={args.conf}) ---")
    for cid, name in CLASS_NAMES.items():
        print(f"{name}: {len(unique_id_sets[cid])}")
    print(f"Output video: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETR + 多物体追跡による果実・花の個体数カウント")
    parser.add_argument("--video", required=True, help="入力動画のパス")
    parser.add_argument("--weights", required=True, help="学習済み RT-DETR 重み (.pt)")
    parser.add_argument("--tracker", choices=TRACKERS, default="bytetrack",
                        help="追跡手法 (default: bytetrack)")
    parser.add_argument("--conf", type=float, default=0.45,
                        help="信頼度閾値。実験では 0.45 / 0.60 / 0.75 / 0.90 を比較 (default: 0.45)")
    parser.add_argument("--output", default=None,
                        help="出力動画のパス (default: outputs/tracked_<tracker>.mp4)")
    parser.add_argument("--reid-weights", default="osnet_x0_25_msmt17.pt",
                        help="BoostTrack 用 Re-ID 重み (default: osnet_x0_25_msmt17.pt)")
    parser.add_argument("--device", default=None, help="BoostTrack の実行デバイス (例: cuda, cpu)")
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join("outputs", f"tracked_{args.tracker}.mp4")
    return args


if __name__ == "__main__":
    run(parse_args())
