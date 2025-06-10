# MLBB Move Analyzer

Mobile Legends: Bang Bang（MLBB）の試合動画を分析するツールです。ミニマップ上のプレイヤー位置を検出し、チームの動きを分析します。

## 主な機能
- YouTubeの動画URLまたはローカルファイルからの動画入力
- ミニマップ上のプレイヤー位置の検出
  - 青チーム（味方）と赤チーム（敵）の識別
  - プレイヤーアイコンの追跡
  - 3x3グリッドによるマップ分割とゾーン分析
- プレイヤー位置の可視化
  - プレイヤー番号表示（A1, A2, ... / E1, E2, ...）
  - ゾーンごとのプレイヤー数表示
- Streamlitベースのユーザーインターフェース
- 分析結果のJSON形式での保存

20250611：Heat mapが最も分析しやすい
![スクリーンショット 2025-06-11 0 21 30](https://github.com/user-attachments/assets/a4b71650-d883-45b2-bcf5-c1515fe544ae)

### 懸念点
ミニマップからのデータを取得しているが、解像度の問題で正しく検知することが難しい。
そのため、ターレットやジャングラーモンスターがキャラクターとして認識されることがある。
＝上記にあるヒートマップを参考に連続的に動いているものをキャラクターとして見ると分析がしやすいと考えている。

## 必要条件

### ローカル環境での実行

- Python 3.12以上
- OpenCV
- Tesseract OCR
- その他requirements.txtに記載のパッケージ

### Docker環境での実行

- Docker
- Docker Compose

## インストール方法

### ローカル環境

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

### Docker環境

1. リポジトリをクローン
```bash
git clone https://github.com/shonagase/mlbb_analyze_move.git
cd mlbb_analyze_move
```

2. Dockerイメージをビルドして起動
```bash
docker-compose up --build
```

## 使用方法

1. アプリケーションを起動
   - ローカル環境の場合：
     ```bash
     streamlit run app.py
     ```
   - Docker環境の場合：
     ```bash
     docker-compose up
     ```

2. ブラウザで表示されるインターフェースから：
   - YouTubeのURLを入力するか、ローカルの動画ファイルをアップロード
   - ミニマップ検出のパラメータを調整（必要に応じて）
   - 「分析開始」ボタンをクリック

3. 分析結果は`data/output`ディレクトリにJSON形式で保存されます

## プレイヤー検出の仕様

### 色範囲設定
- 青チーム（味方）：H=95-125, S≥150, V≥150
- 赤チーム（敵）：H=0-10または170-180, S≥150, V≥150

### プレイヤーアイコンの特徴
- サイズ：50-300ピクセル
- 円形度：0.7以上

### 検出処理の流れ
1. ミニマップ領域の抽出
2. HSVカラー空間での色マスク生成
3. ガウシアンブラーによるノイズ除去
4. モルフォロジー処理による形状の整形
5. 輪郭検出とフィルタリング
6. プレイヤー位置の特定とチーム識別

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
        "minimap_detection": {
            "blue_team_hsv": [[95, 150, 150], [125, 255, 255]],
            "red_team_hsv": [[0, 150, 150], [10, 255, 255], [170, 150, 150], [180, 255, 255]]
        }
    },
    "frames": [
        {
            "frame_number": "フレーム番号",
            "minimap": {
                "blue_team": [
                    {
                        "id": "プレイヤーID（A1-A5）",
                        "position": [x, y],
                        "zone": "ゾーン番号（1-9）"
                    }
                ],
                "red_team": [
                    {
                        "id": "プレイヤーID（E1-E5）",
                        "position": [x, y],
                        "zone": "ゾーン番号（1-9）"
                    }
                ],
                "zone_stats": {
                    "1": {"blue": 数, "red": 数},
                    "2": {"blue": 数, "red": 数},
                    // ... ゾーン9まで
                }
            }
        }
    ]
}
```

## ライセンス

MIT License
 
