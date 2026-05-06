# 万願寺唐辛子の果実・花 自動カウントシステム
### Automated Fruit & Flower Counting for Manganji Pepper using Object Detection and Multi-Object Tracking

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-green)](https://github.com/ultralytics/ultralytics)
<!--[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)-->

> 京都大学農学部 卒業研究（2025年度）  
> フィールドロボティクス研究室

---

## 概要

日本農業における担い手不足・高齢化を背景に、**京都府特産の万願寺唐辛子**を対象とした
果実・花の個体数自動計測システムを構築した。

圃場で撮影した動画に対し、物体検出（Object Detection）と多物体追跡（Multi-Object Tracking）を組み合わせることで、重複カウントを防ぎながら果実・花の個体数をリアルタイムに推定する。

![デモ画像](images/demo/demo.gif)
<!-- 実際の検出結果のGIFをここに配置 -->

---

## 研究の背景と課題

| 課題 | 詳細 |
|------|------|
| **目視カウントの限界** | 果実数・花数の把握は経験と目視に依存し、作業負担が大きく精度のばらつきも課題 |
| **万願寺唐辛子特有の困難** | 果実と葉が同色（緑）・細長い形状・オクルージョン（葉による遮蔽）が頻発 |
| **重複カウント問題** | 動画フレームをまたいで同一個体を別個体として数えてしまう |

---

## 使用技術

### 物体検出モデル
| モデル | アーキテクチャ | 特徴 |
|--------|--------------|------|
| YOLOv5mu | CNN (CSP-Darknet) | 実績あるベースライン |
| YOLOv8m | Anchor-free CNN | 高速・高精度のバランス |
| YOLO11m | 軽量最適化CNN | 処理速度と精度のバランス |
| YOLO12m | Attention機構導入 | 複雑背景での精度向上 |
| **RT-DETR** | **Transformer** | **追跡部分にて採用・最良性能** |

### 追跡モデル (Multi-Object Tracking)
| モデル | 特徴 |
|--------|------|
| **ByteTrack** | 低スコア検出も活用する2段階マッチング → **花クラスで最良** |
| **BoostTrack** | 一時的な検出消失への耐性が高い → **果実クラスで最良** |
| BoT-SORT | ByteTrackと同等の結果 |
| DeepSORT | Re-IDベース（今回の対象には不向きと判明） |

### その他
- **アノテーション**: Roboflow（3クラス：fruits / flowers / dried flowers）
- **実験環境**: Google Colab
- **撮影機材**: GoPro MAX, GoPro9
- **撮影場所**: 京都府農林水産技術センター内の万願寺とうがらし栽培用温室（亀岡市）

---

## 実験結果

### 物体検出モデルの比較（mサイズ統一）

| モデル | Precision | Recall | mAP@50 |
|--------|-----------|--------|--------|
| YOLOv5mu | 0.557 | 0.403 | 0.405 |
| YOLOv8m | 0.567 | 0.361 | 0.393 |
| YOLO11m | 0.491 | 0.284 | 0.297 |
| YOLO12m | 0.633 | 0.370 | 0.424 |
| **RT-DETR** | **0.671** | **0.416** | **0.470** |

→ RT-DETRがYOLO系全モデルを上回り最高精度を達成

### 追跡モデルの比較（conf=0.75 時点・最良組み合わせ）

| クラス | 最適追跡モデル | MAE | 相対誤差 |
|--------|-------------|-----|---------|
| 果実 (fruits) | **BoostTrack** | 7.0 | 29.2% |
| 花 (flowers) | **ByteTrack / BoT-SORT** | 3.4 | 39.5% |
| 枯花 (dried flowers) | **ByteTrack / BoT-SORT** | 2.6 | 34.2% |

→ 対象クラスの特性（外観・運動）に応じて最適な追跡手法が異なることを確認

---

## 主な知見・考察

- **RT-DETRの有効性**: TransformerのSelf-Attentionが、オクルージョンの多い農業環境での特徴抽出に効果的
- **BoostTrackの果実への適合**: 一時的な検出消失（遮蔽）後のID再接続が、葉に隠れやすい果実の追跡に貢献
- **ByteTrackの花への適合**: 小サイズ・低信頼度スコアの検出も2段階マッチングで活用できる点が、サイズの小さい花クラスに有効
- **信頼度閾値の重要性**: conf=0.75が最適値。クラスごとに動的な閾値設定が必要

---

## データセット

プライバシー・研究上の理由から画像・動画データセットは非公開です。

- **学習用静止画**: 2025年9月24日撮影
- **追跡評価用動画**: 2025年12月12日撮影
- **アノテーション数**: fruits 681件 / flowers 1,844件 / dried flowers 415件

---

## 環境構築

```bash
git clone https://github.com/あなたのユーザー名/mangan-ji-pepper-counting.git
cd mangan-ji-pepper-counting
pip install -r requirements.txt
```

---

## ディレクトリ構成

```
mangan-ji-pepper-counting/
├── train・detect/        # 学習・検出スクリプト (RT-DETR, YOLO)
├── track/         # 追跡スクリプト (ByteTrack, BoostTrack等)
├── images/        # デモ画像・実験結果グラフ
└── requirements.txt
```

---

## 参考文献

- Ultralytics YOLO Documentation
- RT-DETR: DETRs Beat YOLOs on Real-time Object Detection
- BoostTrack: GitHub
- ByteTrack: Multi-Object Tracking by Associating Every Detection Box

---

## 著者

**山内 翔太**  
京都大学農学部地域環境工学科 フィールドロボティクス研究室  
📧 yamauchi.shota.87r@st.kyoto-u.ac.jp