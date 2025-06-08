# MLBB Move Analyzer

Mobile Legends: Bang Bang（MLBB）の試合動画を分析するツールです。YOLOv8とDeepSORTを使用して、プレイヤーやオブジェクトの検出とトラッキングを行います。

## 機能

- YouTubeの動画URLまたはローカルファイルからの動画入力
- YOLOv8を使用したオブジェクト検出
- DeepSORTによるオブジェクトトラッキング
- Streamlitベースのユーザーインターフェース
- 分析結果のJSON形式での保存

## インストール

1. リポジトリをクローン
```bash
git clone https://github.com/[your-username]/mlbb_analyze_move.git
cd mlbb_analyze_move
```

2. 仮想環境を作成してアクティベート
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows
```

3. 必要なパッケージをインストール
```bash
pip install -r requirements.txt
```

## 使用方法

1. アプリケーションを起動
```bash
streamlit run main.py
```

2. ブラウザで表示されるインターフェースから：
   - YouTubeのURLを入力するか、ローカルの動画ファイルをアップロード
   - 検出信頼度閾値を調整
   - トラッキングの有効/無効を選択
   - 分析結果の保存の有効/無効を選択
   - 「分析開始」ボタンをクリック

3. 分析結果は`data/output`ディレクトリにJSON形式で保存されます

## 分析結果のフォーマット

```json
{
    "video_info": {
        "title": "動画タイトル",
        "duration": "動画の長さ（秒）",
        "view_count": "視聴回数（YouTubeの場合）",
        "url": "YouTubeURL（YouTubeの場合）"
    },
    "analysis_settings": {
        "confidence_threshold": "検出信頼度閾値",
        "tracking_enabled": "トラッキングの有効/無効"
    },
    "frames": [
        {
            "frame_number": "フレーム番号",
            "detections": [
                {
                    "bbox": [x1, y1, x2, y2],
                    "score": "検出スコア",
                    "class_id": "クラスID"
                }
            ],
            "tracks": [
                {
                    "track_id": "トラッキングID",
                    "bbox": [x1, y1, x2, y2]
                }
            ]
        }
    ]
}
```

## ライセンス

MIT License
