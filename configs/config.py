from pathlib import Path

# プロジェクトのルートディレクトリ
ROOT_DIR = Path(__file__).parent.parent

# データ関連のパス
DATA_DIR = ROOT_DIR / "data"
VIDEO_DIR = DATA_DIR / "videos"
FRAME_DIR = DATA_DIR / "frames"
OUTPUT_DIR = DATA_DIR / "output"
MODEL_DIR = ROOT_DIR / "models"


# フレーム抽出の設定
FRAME_EXTRACTION_FPS = 2  # 1秒あたりのフレーム数
FRAME_QUALITY = 95  # JPEGの品質（0-100）

# 物体検出の設定
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# レーン領域の定義（画面の相対座標）
LANES = {
    "top": [(0, 0), (0.33, 0.33)],
    "mid": [(0.33, 0.33), (0.66, 0.66)],
    "bottom": [(0.66, 0.66), (1, 1)],
    "jungle": [(0, 0), (1, 1)]  # ジャングル全体
}

# イベント検出の閾値
EVENT_THRESHOLDS = {
    "kill_detection": 0.8,
    "tower_destruction": 0.85,
    "teamfight": 3  # 一定範囲内のヒーロー数
}

# トラッキングの設定
TRACKING_CONFIG = {
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3
}

# 可視化の設定
VISUALIZATION_CONFIG = {
    "heatmap_resolution": (100, 100),
    "trajectory_line_width": 2,
    "hero_marker_size": 10
}

# ヒーローの色分け（チーム別）
TEAM_COLORS = {
    "ally": "#00ff00",  # 緑
    "enemy": "#ff0000"  # 赤
} 