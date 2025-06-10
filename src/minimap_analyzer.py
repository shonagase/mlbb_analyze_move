import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
import pytesseract  # OCRのために追加
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.base_minimap = cv2.imread('/Users/shoyanagatomo/Documents/git/mlbb_analyze_move/base_picture/Minimap.webp')
        if self.base_minimap is not None:
            self.base_minimap = cv2.resize(self.base_minimap, (200, 200))

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
        # フレームの寸法を取得して、ROIの座標を計算する
        height, width = frame.shape[:2]
        # TIME_ROIに基づいて、時間表示領域のx, y, width, heightを計算
        x = int(width * self.time_roi['x'])
        y = int(height * self.time_roi['y'])
        w = int(width * self.time_roi['width'])
        h = int(height * self.time_roi['height'])
        
        # 指定されたROIから時間表示領域を抽出
        time_area = frame[y:y+h, x:x+w]
        
        # 画像をグレースケールに変換して二値化処理を行う
        gray = cv2.cvtColor(time_area, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # カーネルを使用してエロージョン、ディレーション、クロージングでノイズを除去
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
        binary = cv2.dilate(binary, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        try:
            # pytesseractでテキストを抽出（数字とコロンだけを対象し、OCRエンジンを最適化）
            text = pytesseract.image_to_string(binary, config='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789:')
            logger.info(f"Extracted text: {text}")  # 抽出されたテキストをログ出力
            import re
            match = re.search(r'\d{1,2}:\d{2}', text)
            if match:
                logger.info(f"Matched game time: {match.group(0)}")  # マッチした時間をログ出力
                return match.group(0)
            else:
                logger.warning("No time pattern matched")
                return ""
        except Exception as e:
            logger.error(f"Time detection error: {e}")  # エラーをログ出力
            return ""

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
        current_minimap = cv2.resize(current_minimap, (200, 200))

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