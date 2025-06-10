import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import pytesseract  # OCRのために追加
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ミニマップの定数
MINIMAP_CONSTANTS = {
    'PLAYER_ICON': {
        'MIN_AREA': 50,      # プレイヤーアイコンの最小面積
        'MAX_AREA': 300,     # プレイヤーアイコンの最大面積
        'MIN_CIRCULARITY': 0.7,  # プレイヤーアイコンの最小円形度
    },
    'COLORS': {
        'ALLY': {
            'PRIMARY': np.array([100, 150, 150]),  # 青チームの主要色
            'SECONDARY': np.array([120, 255, 255])
        },
        'ENEMY': {
            'PRIMARY': np.array([0, 150, 150]),    # 赤チームの主要色
            'SECONDARY': np.array([10, 255, 255])
        },
        'ENEMY2': {
            'PRIMARY': np.array([170, 150, 150]),  # 赤チームの補助色範囲
            'SECONDARY': np.array([180, 255, 255])
        }
    },
    'TOWER': {
        'MIN_AREA': 100,     # タワーアイコンの最小面積
        'MAX_AREA': 400,     # タワーアイコンの最大面積
        'ASPECT_RATIO': 0.5  # タワーアイコンのアスペクト比（高さ/幅）
    },
    'TIME_ROI': {
        'x': 0.474,  # 画面中央から少し左
        'y': 0.0,  # 画面上部
        'width': 0.05,  # ROIの幅
        'height': 0.06  # ROIの高さ
    }
}

class PositionPredictor(nn.Module):
    def __init__(self):
        super(PositionPredictor, self).__init__()
        # シンプルなCNN + RNNモデル
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.rnn = nn.LSTM(input_size=16* (image_height//2) * (image_width//2), hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, image_height * image_width)  # 出力: 位置の確率マップ
    
    def forward(self, x):  # x: バッチ x シーケンス長 x チャネル x 高さ x 幅
        batch_size, seq_len, c, h, w = x.size()
        cnn_out = []
        for t in range(seq_len):
            cnn_out.append(self.cnn(x[:, t]))  # 各フレームをCNNで処理
        cnn_out = torch.stack(cnn_out).view(batch_size, seq_len, -1)  # RNN入力に整形
        rnn_out, _ = self.rnn(cnn_out)
        out = self.fc(rnn_out[:, -1])  # 最終出力
        return F.softmax(out.view(batch_size, image_height, image_width), dim=1)  # 確率分布

class MinimapAnalyzer:
    def __init__(self):
        """
        MLBBのミニマップを分析するクラス
        """
        # ミニマップのROI（画面の左上）
        self.minimap_roi = {
            'x': 0.0,  # 左端から0%
            'y': 0.0,  # 上端から0%
            'width': 0.179,  # 画面幅の１７.9%
            'height': 0.315  # 画面高さの31.5%
        }
        
        # 時間表示のROI
        self.time_roi = MINIMAP_CONSTANTS['TIME_ROI']
        
        # チームの色定義（HSV形式）
        self.team_colors = {
            'ally': {
                'lower': MINIMAP_CONSTANTS['COLORS']['ALLY']['PRIMARY'],
                'upper': MINIMAP_CONSTANTS['COLORS']['ALLY']['SECONDARY']
            },
            'enemy': {
                'lower': MINIMAP_CONSTANTS['COLORS']['ENEMY']['PRIMARY'],
                'upper': MINIMAP_CONSTANTS['COLORS']['ENEMY']['SECONDARY']
            }
        }
        
        # 赤色の第2範囲（HSVの色相は循環するため）
        self.enemy_color_range2 = {
            'lower': MINIMAP_CONSTANTS['COLORS']['ENEMY2']['PRIMARY'],
            'upper': MINIMAP_CONSTANTS['COLORS']['ENEMY2']['SECONDARY']
        }
        
        # 移動履歴の保存用
        self.movement_history = {
            'ally': [],
            'enemy': []
        }
        
        # ミニマップのサイズ（初期値）
        self.minimap_width = 0
        self.minimap_height = 0

        self.kernel_size = (5, 5)
        self.threshold = 30
        self.min_area = 50
        self.max_area = 500
        self.target_size = (200, 200)
        
        # ベース画像を読み込む
        self.base_minimap = cv2.imread('/Users/shoyanagatomo/Documents/git/mlbb_analyze_move/base_picture/Minimap.webp')
        if self.base_minimap is not None:
            self.base_minimap = cv2.resize(self.base_minimap, self.target_size)

        # オブジェクトマスクの作成
        if self.base_minimap is not None:
            self.object_mask = self._create_object_mask()

    def _is_player_icon(self, contour: np.ndarray) -> bool:
        """
        輪郭がプレイヤーアイコンかどうかを判定

        Args:
            contour (np.ndarray): 輪郭データ

        Returns:
            bool: プレイヤーアイコンならTrue
        """
        # 面積チェック
        area = cv2.contourArea(contour)
        if area < MINIMAP_CONSTANTS['PLAYER_ICON']['MIN_AREA'] or \
           area > MINIMAP_CONSTANTS['PLAYER_ICON']['MAX_AREA']:
            return False

        # 円形度チェック
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MINIMAP_CONSTANTS['PLAYER_ICON']['MIN_CIRCULARITY']:
            return False

        return True

    def _check_color_consistency(self, roi_hsv: np.ndarray) -> bool:
        """
        色の一貫性をチェック

        Args:
            roi_hsv (np.ndarray): HSV色空間のROI

        Returns:
            bool: 色が一貫していればTrue
        """
        if roi_hsv.size == 0:
            return False

        # 色の標準偏差を計算
        std_color = np.std(roi_hsv, axis=(0,1))
        return std_color[0] < 15 and std_color[1] < 30 and std_color[2] < 30

    def extract_minimap(self, frame: np.ndarray) -> np.ndarray:
        """
        フレームからミニマップ領域を抽出

        Args:
            frame (np.ndarray): 入力フレーム

        Returns:
            np.ndarray: 抽出されたミニマップ領域
        """
        height, width = frame.shape[:2]
        x = int(width * self.minimap_roi['x'])
        y = int(height * self.minimap_roi['y'])
        w = int(width * self.minimap_roi['width'])
        h = int(height * self.minimap_roi['height'])
        
        # ミニマップのサイズを更新
        self.minimap_width = w
        self.minimap_height = h
        
        return frame[y:y+h, x:x+w]

    def detect_team_positions(self, minimap: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """
        ミニマップ上のチームの位置を検出

        Args:
            minimap (np.ndarray): ミニマップ画像

        Returns:
            Dict[str, List[Tuple[int, int]]]: チームごとの位置座標リスト
        """
        positions = {
            'ally': [],
            'enemy': []
        }

        # 画像の前処理
        blurred = cv2.GaussianBlur(minimap, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 各チームの色マスクを作成
        # 味方（青）の検出
        ally_mask = cv2.inRange(hsv, self.team_colors['ally']['lower'], self.team_colors['ally']['upper'])
        
        # 敵（赤）の検出 - 両方の赤色範囲を組み合わせる
        enemy_mask1 = cv2.inRange(hsv, self.team_colors['enemy']['lower'], self.team_colors['enemy']['upper'])
        enemy_mask2 = cv2.inRange(hsv, self.enemy_color_range2['lower'], self.enemy_color_range2['upper'])
        enemy_mask = cv2.bitwise_or(enemy_mask1, enemy_mask2)

        # 各マスクに対してモルフォロジー処理
        kernel = np.ones((3,3), np.uint8)
        for mask in [ally_mask, enemy_mask]:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 各チームの位置を検出
        for team, mask in [('ally', ally_mask), ('enemy', enemy_mask)]:
            # 輪郭検出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # プレイヤーアイコンの判定
                if not self._is_player_icon(contour):
                    continue

                # 輪郭の中心を計算
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # 中心点の色を確認
                roi_size = 3
                x1 = max(0, cx - roi_size)
                y1 = max(0, cy - roi_size)
                x2 = min(minimap.shape[1], cx + roi_size + 1)
                y2 = min(minimap.shape[0], cy + roi_size + 1)
                
                roi_hsv = hsv[y1:y2, x1:x2]
                
                # 色の一貫性をチェック
                if not self._check_color_consistency(roi_hsv):
                    continue

                # 位置を追加
                positions[team].append((cx, cy))

        # 各チームの検出数を5つまでに制限（距離に基づいてフィルタリング）
        for team in positions:
            if len(positions[team]) > 5:
                # 互いに最も離れた5つの点を選択
                filtered_positions = []
                current_positions = positions[team].copy()
                
                while len(filtered_positions) < 5 and current_positions:
                    if not filtered_positions:
                        # 最初の点を追加
                        filtered_positions.append(current_positions.pop(0))
                    else:
                        # 既存の点から最も離れた点を見つける
                        max_dist = -1
                        best_point = None
                        best_idx = -1
                        
                        for i, point in enumerate(current_positions):
                            min_dist = float('inf')
                            for existing_point in filtered_positions:
                                dist = np.sqrt((point[0] - existing_point[0])**2 + 
                                             (point[1] - existing_point[1])**2)
                                min_dist = min(min_dist, dist)
                            
                            if min_dist > max_dist:
                                max_dist = min_dist
                                best_point = point
                                best_idx = i
                        
                        if best_point is not None:
                            filtered_positions.append(best_point)
                            current_positions.pop(best_idx)
                
                positions[team] = filtered_positions

        return positions

    def update_movement_history(self, positions: Dict[str, List[Tuple[int, int]]], frame_number: int):
        """
        移動履歴を更新

        Args:
            positions (Dict[str, List[Tuple[int, int]]]): チームごとの位置座標
            frame_number (int): フレーム番号
        """
        for team in ['ally', 'enemy']:
            for pos in positions[team]:
                self.movement_history[team].append({
                    'frame': frame_number,
                    'position': pos
                })

    def analyze_movement_patterns(self) -> Dict[str, Dict]:
        """
        移動パターンを分析

        Returns:
            Dict[str, Dict]: チームごとの移動パターン分析結果
        """
        patterns = {
            'ally': {'zones': defaultdict(int), 'transitions': []},
            'enemy': {'zones': defaultdict(int), 'transitions': []}
        }
        
        for team in ['ally', 'enemy']:
            prev_zone = None
            for movement in self.movement_history[team]:
                pos = movement['position']
                # ミニマップを9つのゾーンに分割
                zone = (
                    'top' if pos[1] < self.minimap_height/3 else 
                    'middle' if pos[1] < 2*self.minimap_height/3 else 
                    'bottom',
                    'left' if pos[0] < self.minimap_width/3 else 
                    'center' if pos[0] < 2*self.minimap_width/3 else 
                    'right'
                )
                
                patterns[team]['zones'][zone] += 1
                
                if prev_zone and prev_zone != zone:
                    patterns[team]['transitions'].append((prev_zone, zone))
                
                prev_zone = zone
        
        return patterns

    def extract_game_time(self, frame: np.ndarray) -> str:
        """
        フレームから試合経過時間を抽出

        Args:
            frame (np.ndarray): 入力フレーム

        Returns:
            str: 検出された時間文字列（例: "12:34"）
        """
        try:
            height, width = frame.shape[:2]
            
            # ROIの座標を計算
            x = int(width * self.time_roi['x'])
            y = int(height * self.time_roi['y'])
            w = int(width * self.time_roi['width'])
            h = int(height * self.time_roi['height'])
            
            # ROI領域を切り出し
            time_area = frame[y:y+h, x:x+w]
            
            # グレースケールに変換
            gray = cv2.cvtColor(time_area, cv2.COLOR_BGR2GRAY)
            
            # 二値化
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # ノイズ除去
            kernel = np.ones((2,2), np.uint8)
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # OCRで時間を抽出
            text = pytesseract.image_to_string(processed, config='--psm 7 -c tessedit_char_whitelist=0123456789:')
            
            # 時間形式（mm:ss）の文字列を抽出
            import re
            time_match = re.search(r'\d{1,2}:\d{2}', text)
            
            if time_match:
                return time_match.group(0)
            return None
            
        except Exception as e:
            logging.error(f"Error extracting game time: {str(e)}")
            return None

    def process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, int]]], str]:
        """
        フレームを処理してミニマップ分析を実行

        Args:
            frame (np.ndarray): 入力フレーム
            frame_number (int): フレーム番号

        Returns:
            Tuple[np.ndarray, Dict[str, List[Tuple[int, int]]], str]: 
            可視化されたミニマップ、検出された位置情報、ゲーム経過時間
        """
        # ミニマップ領域の抽出
        minimap = self.extract_minimap(frame)
        
        # チームの位置を検出
        positions = self.detect_team_positions(minimap)
        
        # 移動履歴の更新
        self.update_movement_history(positions, frame_number)
        
        # 経過時間の検出
        game_time = self.extract_game_time(frame)
        
        # 結果の可視化（時間情報も追加）
        vis_map = self.visualize_minimap(minimap, positions, game_time)
        
        return vis_map, positions, game_time

    def visualize_minimap(self, minimap: np.ndarray, positions: Dict[str, List[Tuple[int, int]]], game_time: str = "") -> np.ndarray:
        """
        ミニマップ上の検出結果を可視化

        Args:
            minimap (np.ndarray): ミニマップ画像
            positions (Dict[str, List[Tuple[int, int]]]): チームごとの位置座標
            game_time (str): 検出された試合経過時間

        Returns:
            np.ndarray: 可視化されたミニマップ
        """
        vis_map = minimap.copy()
        
        # チームごとに異なる色で描画
        colors = {
            'ally': (255, 100, 0),  # 青色（BGR）
            'enemy': (0, 0, 255)    # 赤色（BGR）
        }
        
        # グリッドを描画（3x3）
        h, w = vis_map.shape[:2]
        grid_color = (128, 128, 128)  # グレー
        
        # 縦線
        for x in range(1, 3):
            cv2.line(vis_map, (w * x // 3, 0), (w * x // 3, h), grid_color, 1)
        
        # 横線
        for y in range(1, 3):
            cv2.line(vis_map, (0, h * y // 3), (w, h * y // 3), grid_color, 1)
        
        # 検出位置の可視化を改善
        for team, pos_list in positions.items():
            for i, pos in enumerate(pos_list):
                # 外側の円（輪郭）
                cv2.circle(vis_map, pos, 8, colors[team], 2)
                # 内側の円（塗りつぶし）
                cv2.circle(vis_map, pos, 4, colors[team], -1)
                
                # プレイヤー番号を表示
                label = f"{team[0].upper()}{i+1}"  # A1, A2, ... または E1, E2, ...
                cv2.putText(vis_map, label, 
                          (pos[0] - 10, pos[1] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 
                          0.4, colors[team], 1)
        
        # 検出数とゾーン情報を表示
        info_text = []
        info_text.append(f"Ally: {len(positions['ally'])} Enemy: {len(positions['enemy'])}")
        
        # ゾーンごとのプレイヤー数を計算
        zones = {
            'top': {'ally': 0, 'enemy': 0},
            'mid': {'ally': 0, 'enemy': 0},
            'bot': {'ally': 0, 'enemy': 0}
        }
        
        for team, pos_list in positions.items():
            for pos in pos_list:
                y = pos[1]
                if y < h/3:
                    zones['top'][team] += 1
                elif y < 2*h/3:
                    zones['mid'][team] += 1
                else:
                    zones['bot'][team] += 1
        
        # ゾーン情報を追加
        for zone in ['top', 'mid', 'bot']:
            if zones[zone]['ally'] > 0 or zones[zone]['enemy'] > 0:
                info_text.append(f"{zone.upper()}: A{zones[zone]['ally']} E{zones[zone]['enemy']}")
        
        # 時間情報を追加
        if game_time:
            info_text.insert(0, f"Time: {game_time}")
        
        # 情報テキストを描画
        for i, text in enumerate(info_text):
            y_pos = vis_map.shape[0] - 10 - (i * 15)
            cv2.putText(vis_map, text,
                       (10, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.4, (255, 255, 255), 1)
        
        return vis_map 

    def predict_position_probabilities(self, image_sequence: List[np.ndarray]) -> np.ndarray:
        # image_sequence: [過去A, 過去B, 現在A, 現在B, 将来A, 将来B]
        model = PositionPredictor()  # モデルインスタンス
        model.eval()  # 評価モード
        # シーケンスをテンソルに変換（簡易実装、実際は前処理が必要）
        input_tensor = torch.from_numpy(np.array(image_sequence)).permute(0, 3, 1, 2).unsqueeze(0).float()  # バッチ追加
        with torch.no_grad():
            output = model(input_tensor)
            return output.numpy()[0]  # 確率マップを返す 

    def analyze_frame_differences(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        3つの連続フレーム間の差分を分析し、変化の確率分布を計算

        Args:
            frames (List[np.ndarray]): 3つの連続フレーム [frame1, frame2, frame3]

        Returns:
            Tuple[np.ndarray, Dict]: 確率マップと変化の統計情報
        """
        if len(frames) != 3:
            raise ValueError("3つのフレームが必要です")

        # フレームをグレースケールに変換
        grays = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        # 差分を計算
        diff1 = cv2.absdiff(grays[0], grays[1])
        diff2 = cv2.absdiff(grays[1], grays[2])
        
        # ミニマップ用に閾値を調整（より小さな変化も検出）
        _, thresh1 = cv2.threshold(diff1, 20, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(diff2, 20, 255, cv2.THRESH_BINARY)
        
        # 2つの差分の組み合わせ
        combined_diff = cv2.bitwise_or(thresh1, thresh2)
        
        # ノイズ除去（カーネルサイズを小さくしてミニマップの細かい変化を保持）
        kernel = np.ones((2,2), np.uint8)
        combined_diff = cv2.morphologyEx(combined_diff, cv2.MORPH_OPEN, kernel)
        
        # 変化領域の検出
        contours, _ = cv2.findContours(combined_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 確率マップの作成（ヒートマップ形式）
        height, width = combined_diff.shape
        prob_map = np.zeros((height, width), dtype=np.float32)
        
        # 変化の統計情報
        changes = {
            'total_changes': len(contours),
            'change_areas': [],
            'movement_vectors': []
        }
        
        # 各変化領域の分析（ミニマップ用に面積の閾値を調整）
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 5:  # ミニマップでの最小面積を調整
                continue
                
            # 変化領域の中心を計算
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # ガウシアン分布で確率を表現（σをミニマップのスケールに合わせて調整）
                sigma = np.sqrt(area) / 4  # より狭い範囲に
                y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
                gaussian = np.exp(-(x*x + y*y) / (2*sigma*sigma))
                prob_map = np.maximum(prob_map, gaussian)
                
                # 変化情報を記録
                changes['change_areas'].append({
                    'center': (cx, cy),
                    'area': area,
                    'intensity': np.mean(combined_diff[cy-2:cy+3, cx-2:cx+3])
                })
                
                # フレーム間の移動ベクトルを計算（探索範囲をミニマップに合わせて調整）
                if len(changes['movement_vectors']) < 5:
                    prev_pos = self._find_corresponding_position(grays[0], (cx, cy), search_radius=10)
                    next_pos = self._find_corresponding_position(grays[2], (cx, cy), search_radius=10)
                    if prev_pos and next_pos:
                        movement = {
                            'start': prev_pos,
                            'mid': (cx, cy),
                            'end': next_pos,
                            'velocity': np.sqrt((next_pos[0]-prev_pos[0])**2 + (next_pos[1]-prev_pos[1])**2)
                        }
                        changes['movement_vectors'].append(movement)
        
        # 確率マップの正規化
        if np.max(prob_map) > 0:
            prob_map = prob_map / np.max(prob_map)
        
        return prob_map, changes

    def _find_corresponding_position(self, frame: np.ndarray, center: Tuple[int, int], 
                                   search_radius: int = 20) -> Optional[Tuple[int, int]]:
        """
        指定されたフレームで対応する位置を探索
        """
        cx, cy = center
        h, w = frame.shape
        
        # 探索範囲を制限
        x1 = max(0, cx - search_radius)
        y1 = max(0, cy - search_radius)
        x2 = min(w, cx + search_radius + 1)
        y2 = min(h, cy + search_radius + 1)
        
        # テンプレートマッチングで最も類似した位置を探索
        template = frame[cy-2:cy+3, cx-2:cx+3]  # 5x5のテンプレート
        if template.shape[0] > 0 and template.shape[1] > 0:
            roi = frame[y1:y2, x1:x2]
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            # 探索範囲内での相対位置を全体座標に変換
            return (x1 + max_loc[0], y1 + max_loc[1])
        
        return None

    def visualize_changes(self, prob_map: np.ndarray, changes: Dict) -> np.ndarray:
        """
        変化の確率マップと統計情報を可視化
        """
        # 確率マップをヒートマップとしてカラー化
        heatmap = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 移動ベクトルを描画
        for movement in changes['movement_vectors']:
            # 移動の軌跡を矢印で表示
            cv2.arrowedLine(heatmap, movement['start'], movement['end'], 
                          (255, 255, 255), 2, tipLength=0.3)
            
            # 中間点をマーク
            cv2.circle(heatmap, movement['mid'], 3, (0, 255, 0), -1)
        
        # 変化領域の中心をマーク
        for area in changes['change_areas']:
            cv2.circle(heatmap, area['center'], 5, (0, 0, 255), -1)
        
        return heatmap 

    def compare_with_base(self, current_minimap):
        """
        ベースミニマップと現在のミニマップを比較し、差分を検出する
        
        Args:
            current_minimap: 現在のフレームから抽出したミニマップ画像
        
        Returns:
            diff_map: 差分を可視化した画像
            diff_stats: 差分の統計情報
        """
        if self.base_minimap is None:
            return None, {"error": "Base minimap not loaded"}

        # 現在のミニマップをベースミニマップと同じサイズにリサイズ
        current_minimap = cv2.resize(current_minimap, self.target_size)

        # グレースケールに変換
        base_gray = cv2.cvtColor(self.base_minimap, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_minimap, cv2.COLOR_BGR2GRAY)

        # 差分を計算
        diff = cv2.absdiff(base_gray, current_gray)
        
        # 閾値処理
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 差分領域の検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 差分の可視化
        diff_map = current_minimap.copy()
        diff_areas = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                # 差分領域の中心を計算
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 差分領域を赤色で描画
                    cv2.drawContours(diff_map, [contour], -1, (0, 0, 255), 2)
                    cv2.circle(diff_map, (cx, cy), 3, (255, 0, 0), -1)
                    
                    diff_areas.append({
                        "center": (cx, cy),
                        "area": area,
                        "contour": contour.tolist()
                    })

        diff_stats = {
            "total_differences": len(diff_areas),
            "diff_areas": diff_areas,
            "total_diff_area": sum(area["area"] for area in diff_areas)
        }

        return diff_map, diff_stats

    def analyze_map_changes(self, current_minimap):
        """
        現在のミニマップを3つのベース画像と比較して変化を分析する
        
        Args:
            current_minimap: 現在のフレームから抽出したミニマップ画像
            
        Returns:
            dict: 分析結果を含む辞書
        """
        current_minimap = cv2.resize(current_minimap, self.target_size)
        
        # 1. 地形との比較（固定オブジェクトの検出）
        terrain_diff, terrain_stats = self._compare_images(self.base_minimap, current_minimap)
        
        # 2. オブジェクトありの画像との比較（動的オブジェクトの検出）
        objects_diff, objects_stats = self._compare_images(self.base_minimap, current_minimap)
        
        # 3. バトル状態の画像との比較（キャラクターの移動検出）
        battle_diff, battle_stats = self._compare_images(self.base_minimap, current_minimap)
        
        # 変化領域の分類
        classified_changes = self._classify_changes(
            terrain_stats['diff_areas'],
            objects_stats['diff_areas'],
            battle_stats['diff_areas']
        )
        
        return {
            'terrain_comparison': {
                'diff_map': terrain_diff,
                'stats': terrain_stats
            },
            'objects_comparison': {
                'diff_map': objects_diff,
                'stats': objects_stats
            },
            'battle_comparison': {
                'diff_map': battle_diff,
                'stats': battle_stats
            },
            'classified_changes': classified_changes
        }

    def _compare_images(self, base_img, current_img):
        """
        2つの画像を比較し、差分を検出する
        """
        if base_img is None:
            return None, {"error": "Base image not loaded"}

        # グレースケールに変換
        base_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

        # 差分を計算
        diff = cv2.absdiff(base_gray, current_gray)
        
        # 閾値処理
        _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 差分領域の検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 差分の可視化
        diff_map = current_img.copy()
        diff_areas = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                # 差分領域の中心を計算
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 差分領域を赤色で描画
                    cv2.drawContours(diff_map, [contour], -1, (0, 0, 255), 2)
                    cv2.circle(diff_map, (cx, cy), 3, (255, 0, 0), -1)
                    
                    diff_areas.append({
                        "center": (cx, cy),
                        "area": area,
                        "contour": contour.tolist()
                    })

        return diff_map, {
            "total_differences": len(diff_areas),
            "diff_areas": diff_areas,
            "total_diff_area": sum(area["area"] for area in diff_areas)
        }

    def _classify_changes(self, terrain_changes, object_changes, battle_changes):
        """
        検出された変化を分類する
        """
        def calculate_overlap(area1, area2):
            # 中心点間の距離で簡易的な重なり判定
            c1 = np.array(area1["center"])
            c2 = np.array(area2["center"])
            return np.linalg.norm(c1 - c2)

        classified = {
            "static_objects": [],  # 地形との比較でのみ検出された変化
            "dynamic_objects": [], # オブジェクトありの画像との比較で新たに検出された変化
            "characters": [],      # バトル画像との比較で新たに検出された変化
            "movements": []        # 前フレームからの移動
        }

        # 重なり判定の閾値
        overlap_threshold = 20

        # 地形との差分をまず静的オブジェクトとして分類
        for change in terrain_changes:
            classified["static_objects"].append({
                "position": change["center"],
                "size": change["area"],
                "type": "static"
            })

        # オブジェクトありの画像との差分を動的オブジェクトとして分類
        for change in object_changes:
            is_new = True
            for static in classified["static_objects"]:
                if calculate_overlap(change, {"center": static["position"]}) < overlap_threshold:
                    is_new = False
                    break
            if is_new:
                classified["dynamic_objects"].append({
                    "position": change["center"],
                    "size": change["area"],
                    "type": "dynamic"
                })

        # バトル画像との差分をキャラクターとして分類
        for change in battle_changes:
            is_new = True
            for existing in (classified["static_objects"] + classified["dynamic_objects"]):
                if calculate_overlap(change, {"center": existing["position"]}) < overlap_threshold:
                    is_new = False
                    break
            if is_new:
                classified["characters"].append({
                    "position": change["center"],
                    "size": change["area"],
                    "direction": change["angle"],
                    "type": "character"
                })

        return classified 

    def _create_object_mask(self):
        """
        固定オブジェクトのマスクを作成
        """
        if self.base_minimap is None:
            return None
            
        # グレースケールに変換
        gray = cv2.cvtColor(self.base_minimap, cv2.COLOR_BGR2GRAY)
        
        # 二値化
        _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # オブジェクト領域を検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # マスク画像を作成（3チャンネル）
        mask = np.zeros_like(self.base_minimap)
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                # オブジェクト領域を少し拡大してマージン追加
                cv2.drawContours(mask, [contour], -1, (255, 255, 255), 3)
        
        return mask

    def analyze_character_movement(self, current_minimap, prev_minimap=None):
        """
        キャラクターの動きを分析する
        
        Args:
            current_minimap: 現在のフレームのミニマップ
            prev_minimap: 前のフレームのミニマップ（オプション）
            
        Returns:
            dict: 分析結果
        """
        if self.object_mask is None:
            return {"error": "Object mask not created"}
            
        current_minimap = cv2.resize(current_minimap, self.target_size)
        
        # オブジェクトマスクを使用して固定オブジェクトを除外
        masked_current = cv2.bitwise_and(current_minimap, current_minimap, 
                                       mask=cv2.bitwise_not(self.object_mask))
        
        # キャラクター検出
        characters = self._detect_characters(masked_current)
        
        # 前フレームが提供された場合は移動を分析
        movement_analysis = None
        if prev_minimap is not None:
            prev_minimap = cv2.resize(prev_minimap, self.target_size)
            masked_prev = cv2.bitwise_and(prev_minimap, prev_minimap, 
                                        mask=cv2.bitwise_not(self.object_mask))
            movement_analysis = self._analyze_movement(masked_prev, masked_current)
        
        return {
            'characters': characters,
            'movement': movement_analysis,
            'visualization': {
                'masked_map': masked_current,
                'character_positions': self._visualize_characters(current_minimap, characters)
            }
        }

    def _detect_characters(self, masked_image):
        """
        マスク済み画像からキャラクターを検出
        """
        # グレースケールに変換
        gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
        
        # 二値化
        _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        
        # ノイズ除去
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # キャラクター領域を検出
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        characters = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # 向きを計算
                    rect = cv2.minAreaRect(contour)
                    
                    characters.append({
                        "position": (cx, cy),
                        "area": area,
                        "direction": rect[-1],
                        "contour": contour
                    })
        
        return characters

    def _analyze_movement(self, prev_image, current_image):
        """
        2フレーム間のキャラクターの移動を分析
        """
        prev_chars = self._detect_characters(prev_image)
        current_chars = self._detect_characters(current_image)
        
        movements = []
        
        # 各キャラクターの移動を追跡
        for prev_char in prev_chars:
            min_dist = float('inf')
            matched_char = None
            
            for curr_char in current_chars:
                dist = np.linalg.norm(
                    np.array(prev_char["position"]) - np.array(curr_char["position"])
                )
                if dist < min_dist and dist < 50:  # 50ピクセルを最大移動距離とする
                    min_dist = dist
                    matched_char = curr_char
            
            if matched_char:
                movement = {
                    "start": prev_char["position"],
                    "end": matched_char["position"],
                    "distance": min_dist,
                    "direction_change": matched_char["direction"] - prev_char["direction"],
                    "speed": min_dist  # フレーム間の距離を速度として使用
                }
                movements.append(movement)
        
        return {
            "total_movements": len(movements),
            "movements": movements,
            "average_speed": np.mean([m["speed"] for m in movements]) if movements else 0
        }

    def _visualize_characters(self, original_image, characters):
        """
        検出されたキャラクターを可視化
        """
        vis_image = original_image.copy()
        
        for char in characters:
            # キャラクターの位置を円で表示
            cv2.circle(vis_image, char["position"], 5, (0, 255, 0), -1)
            
            # 向きを矢印で表示
            angle_rad = np.deg2rad(char["direction"])
            arrow_length = 20
            end_point = (
                int(char["position"][0] + arrow_length * np.cos(angle_rad)),
                int(char["position"][1] + arrow_length * np.sin(angle_rad))
            )
            cv2.arrowedLine(vis_image, char["position"], end_point, (255, 0, 0), 2)
            
            # キャラクター領域を表示
            cv2.drawContours(vis_image, [char["contour"]], -1, (0, 255, 0), 2)
        
        return vis_image 

    def analyze_character_movement_sequence(self, current_frame_number, cap, fps=15, window_size=15):
        """
        現在のフレームの前後のフレームを使用してキャラクターの動きを分析する
        
        Args:
            current_frame_number: 現在のフレーム番号
            cap: VideoCapture オブジェクト
            fps: 1秒あたりのフレーム数
            window_size: 前後何フレームまで見るか
            
        Returns:
            dict: 分析結果
        """
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 前後のフレーム番号を計算
        start_frame = max(0, current_frame_number - window_size)
        end_frame = min(total_frames - 1, current_frame_number + window_size)
        
        # フレームシーケンスを取得
        frames = []
        minimaps = []
        frame_positions = []
        
        for frame_num in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                minimap = self.extract_minimap(frame)
                if minimap is not None:
                    frames.append(frame)
                    minimaps.append(minimap)
                    frame_positions.append(frame_num)
        
        if not minimaps:
            return {"error": "No valid frames found"}
        
        # 動きの分析
        movement_analysis = self._analyze_sequence_movement(minimaps, frame_positions)
        
        # 現在のフレームに戻す
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
        
        return movement_analysis

    def _analyze_sequence_movement(self, minimaps, frame_positions):
        """
        フレームシーケンスから動きを分析する
        """
        if not minimaps:
            return {"error": "No minimaps provided"}
            
        # 各フレームのキャラクター位置を追跡
        character_tracks = []
        movement_heatmap = np.zeros(self.target_size, dtype=np.float32)
        
        # 各フレームでキャラクターを検出
        frame_characters = []
        for minimap in minimaps:
            # オブジェクトマスクを適用
            if self.object_mask is not None:
                # マスクを現在のミニマップと同じサイズにリサイズ
                resized_mask = cv2.resize(self.object_mask, (minimap.shape[1], minimap.shape[0]))
                # マスクを反転（オブジェクト以外の領域を取得）
                inverted_mask = cv2.bitwise_not(resized_mask)
                # マスク処理
                masked_minimap = cv2.bitwise_and(minimap, inverted_mask)
            else:
                masked_minimap = minimap.copy()
            
            chars = self._detect_characters(masked_minimap)
            frame_characters.append(chars)
        
        # キャラクターの軌跡を追跡
        for i in range(len(frame_characters)):
            current_chars = frame_characters[i]
            
            for char in current_chars:
                track = {
                    "start_frame": frame_positions[i],
                    "positions": [(frame_positions[i], char["position"])],
                    "directions": [(frame_positions[i], char["direction"])]
                }
                
                # 後続フレームでの位置を追跡
                for j in range(i + 1, len(frame_characters)):
                    next_chars = frame_characters[j]
                    best_match = None
                    min_dist = float('inf')
                    
                    for next_char in next_chars:
                        dist = np.linalg.norm(
                            np.array(char["position"]) - np.array(next_char["position"])
                        )
                        if dist < min_dist and dist < 50:  # 50ピクセルを最大移動距離とする
                            min_dist = dist
                            best_match = next_char
                    
                    if best_match:
                        track["positions"].append((frame_positions[j], best_match["position"]))
                        track["directions"].append((frame_positions[j], best_match["direction"]))
                        
                        # ヒートマップを更新
                        pt1 = np.array(char["position"])
                        pt2 = np.array(best_match["position"])
                        cv2.line(movement_heatmap, 
                                tuple(pt1.astype(int)), 
                                tuple(pt2.astype(int)), 
                                1.0, 
                                2)
                
                if len(track["positions"]) > 1:  # 2フレーム以上で追跡できた場合のみ追加
                    character_tracks.append(track)
        
        # ヒートマップの正規化と可視化
        if movement_heatmap.max() > 0:
            movement_heatmap = (movement_heatmap / movement_heatmap.max() * 255).astype(np.uint8)
            movement_heatmap = cv2.applyColorMap(movement_heatmap, cv2.COLORMAP_JET)
        
        # 軌跡の可視化
        trajectory_map = minimaps[len(minimaps)//2].copy()  # 中央のフレームを使用
        for track in character_tracks:
            positions = [pos[1] for pos in track["positions"]]
            for i in range(len(positions) - 1):
                pt1 = positions[i]
                pt2 = positions[i + 1]
                # 移動の軌跡を描画（時間経過で色を変える）
                color_factor = i / (len(positions) - 1)
                color = (
                    int(255 * (1 - color_factor)),  # Blue
                    int(255 * color_factor),        # Green
                    0                               # Red
                )
                cv2.line(trajectory_map, pt1, pt2, color, 2)
                # 各位置にポイントを描画
                cv2.circle(trajectory_map, pt1, 3, color, -1)
            if positions:
                cv2.circle(trajectory_map, positions[-1], 3, (0, 255, 0), -1)
        
        return {
            "tracks": character_tracks,
            "total_tracks": len(character_tracks),
            "frame_range": (frame_positions[0], frame_positions[-1]),
            "visualization": {
                "heatmap": movement_heatmap,
                "trajectory": trajectory_map
            },
            "movement_statistics": self._calculate_movement_statistics(character_tracks)
        }

    def _calculate_movement_statistics(self, tracks):
        """
        軌跡から移動統計を計算
        """
        stats = {
            "average_speed": [],
            "total_distance": [],
            "direction_changes": [],
            "movement_patterns": []
        }
        
        for track in tracks:
            positions = [pos[1] for pos in track["positions"]]
            frames = [pos[0] for pos in track["positions"]]
            directions = [dir[1] for dir in track["directions"]]
            
            # 総移動距離と平均速度
            total_dist = 0
            speeds = []
            for i in range(len(positions) - 1):
                pt1 = np.array(positions[i])
                pt2 = np.array(positions[i + 1])
                dist = np.linalg.norm(pt2 - pt1)
                frame_diff = frames[i + 1] - frames[i]
                if frame_diff > 0:
                    speed = dist / frame_diff
                    speeds.append(speed)
                total_dist += dist
            
            # 方向変化
            direction_changes = []
            for i in range(len(directions) - 1):
                change = directions[i + 1] - directions[i]
                # 角度の正規化（-180から180の範囲に）
                if change > 180:
                    change -= 360
                elif change < -180:
                    change += 360
                direction_changes.append(change)
            
            stats["average_speed"].append(np.mean(speeds) if speeds else 0)
            stats["total_distance"].append(total_dist)
            stats["direction_changes"].append(direction_changes)
            
            # 移動パターンの分類
            if total_dist < 10:
                pattern = "stationary"
            elif len(set(direction_changes)) <= 2:
                pattern = "linear"
            else:
                pattern = "complex"
            stats["movement_patterns"].append(pattern)
        
        return {
            "average_speed": np.mean(stats["average_speed"]),
            "max_speed": np.max(stats["average_speed"]) if stats["average_speed"] else 0,
            "average_distance": np.mean(stats["total_distance"]) if stats["total_distance"] else 0,
            "movement_patterns": {
                pattern: stats["movement_patterns"].count(pattern)
                for pattern in set(stats["movement_patterns"])
            }
        } 

    def analyze_time_series(self, video_capture, start_frame, duration_seconds, fps=30):
        """
        指定された時間範囲内でのミニマップ上の動きを分析する
        
        Args:
            video_capture: cv2.VideoCapture オブジェクト
            start_frame: 開始フレーム番号
            duration_seconds: 分析する時間（秒）
            fps: ビデオのFPS（デフォルト30）
            
        Returns:
            movement_map: 時系列での動きを可視化した画像
            time_stats: 時間ごとの統計情報
        """
        total_frames = int(duration_seconds * fps)
        movement_map = np.zeros(self.target_size + (3,), dtype=np.uint8)
        time_stats = {
            "total_movements": 0,
            "time_segments": [],
            "hot_zones": []
        }
        
        # 色相の範囲を計算（時間経過を色で表現）
        hue_step = 180 / total_frames
        
        # 開始フレームに移動
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(total_frames):
            ret, frame = video_capture.read()
            if not ret:
                break
                
            # 現在のフレームのミニマップを分析
            current_minimap = self.extract_minimap(frame)
            diff_map, diff_stats = self.compare_with_base(current_minimap)
            
            if diff_stats['total_differences'] > 0:
                # 時間に基づいて色を生成（HSVカラースペース）
                hue = int(frame_idx * hue_step)
                color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                
                # 動きを検出した領域を描画
                for area in diff_stats['diff_areas']:
                    center = area['center']
                    # 中心点を描画（時間経過で色が変化）
                    cv2.circle(movement_map, center, 2, color.tolist(), -1)
                    
                    # 統計情報を記録
                    time_segment = {
                        "frame": start_frame + frame_idx,
                        "time": frame_idx / fps,
                        "position": center,
                        "area": area['area']
                    }
                    time_stats["time_segments"].append(time_segment)
                    time_stats["total_movements"] += 1
        
        # ホットゾーンの検出（よく動きが検出される領域）
        if time_stats["time_segments"]:
            positions = np.array([segment["position"] for segment in time_stats["time_segments"]])
            hot_zones = self._detect_hot_zones(positions)
            time_stats["hot_zones"] = hot_zones
        
        return movement_map, time_stats
        
    def _detect_hot_zones(self, positions, min_points=5, eps=10):
        """
        DBSCANを使用してホットゾーン（頻繁に動きが検出される領域）を検出
        
        Args:
            positions: 動きが検出された位置の配列
            min_points: クラスタを形成するための最小ポイント数
            eps: クラスタ半径
            
        Returns:
            hot_zones: 検出されたホットゾーンのリスト
        """
        if len(positions) < min_points:
            return []
            
        # DBSCANでクラスタリング
        clustering = DBSCAN(eps=eps, min_samples=min_points).fit(positions)
        
        hot_zones = []
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # ノイズポイントをスキップ
                continue
                
            # クラスタに属する点を抽出
            cluster_points = positions[clustering.labels_ == label]
            
            # クラスタの中心と範囲を計算
            center = np.mean(cluster_points, axis=0)
            radius = np.max(np.linalg.norm(cluster_points - center, axis=1))
            
            hot_zones.append({
                "center": tuple(map(int, center)),
                "radius": int(radius),
                "point_count": len(cluster_points)
            })
            
        return hot_zones 