# MLBB Game Analyzer

Mobile Legends: Bang Bang（MLBB）の試合動画を分析し、プレイヤーの動きやゲーム展開を可視化・分析するツールです。

## 機能

- YouTube動画の自動ダウンロードと処理
- ヒーローの検出とトラッキング
- レーン移動の分析
- チームファイトの検出と分析
- 目標物（タワーなど）の攻略分析
- チームパフォーマンスの評価
- 分析結果の可視化とエクスポート

## 必要条件

- Python 3.9以上
- 必要なパッケージは`requirements.txt`に記載

## インストール

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/mlbb_analyze_move.git
cd mlbb_analyze_move

# 仮想環境の作成と有効化
python -m venv venv
source venv/bin/activate  # Linuxの場合
# または
.\venv\Scripts\activate  # Windowsの場合

# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

### Streamlitアプリケーションの起動

```bash
streamlit run src/main.py
```

### コマンドラインからの実行

```python
from src.main import MLBBAnalyzer
from pathlib import Path

analyzer = MLBBAnalyzer()
video_path = Path("path/to/your/video.mp4")
analyzer.process_video(video_path)
```

## プロジェクト構造

```
mlbb_analyze_move/
├── src/
│   ├── main.py              # メインアプリケーション
│   ├── downloader.py        # YouTube動画ダウンローダー
│   ├── frame_extractor.py   # フレーム抽出
│   ├── detection.py         # 物体検出とトラッキング
│   ├── lane_mapper.py       # レーンマッピング
│   ├── event_extractor.py   # イベント検出
│   └── analyzer.py          # データ分析
├── models/                  # YOLOv8モデルファイル
├── configs/                 # 設定ファイル
├── data/                    # データ保存ディレクトリ
│   ├── videos/             # ダウンロードした動画
│   ├── frames/             # 抽出したフレーム
│   └── output/             # 分析結果
└── requirements.txt        # 依存パッケージ
```

## カスタマイズ

### YOLOv8モデルのカスタマイズ

1. MLBBのゲーム画面でヒーローやオブジェクトをアノテーション
2. YOLOv8でモデルを学習
3. 学習済みモデルを`models/`ディレクトリに配置
4. `configs/config.py`でモデルパスを更新

### レーン定義のカスタマイズ

`configs/config.py`の`LANES`辞書を編集して、レーン領域を調整できます：

```python
LANES = {
    "top": [(0, 0), (0.33, 0.33)],
    "mid": [(0.33, 0.33), (0.66, 0.66)],
    "bottom": [(0.66, 0.66), (1, 1)],
    "jungle": [(0, 0), (1, 1)]
}
```

## 分析結果

分析結果は以下の形式で出力されます：

- `analysis_results.json`: 詳細な分析結果（JSON形式）
- `team_performance.csv`: チームパフォーマンス指標（CSV形式）
- ヒートマップや軌跡の可視化（画像ファイル）

## 注意事項

- 動画の解像度や品質によって検出精度が変わる可能性があります
- 処理時間は動画の長さと設定に依存します
- GPUがある場合は自動的に使用されます

## ライセンス

MITライセンス

## 貢献

1. Forkする
2. 機能ブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. Pull Requestを作成

## 作者

Your Name (@yourusername) 