import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from configs.config import LANES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LaneMapper:
    def __init__(self, lanes: Dict = LANES):
        """
        レーンマッピングを行うクラス

        Args:
            lanes (Dict): レーン領域の定義
        """
        self.lanes = lanes
        self.lane_history = {}  # track_idごとのレーン履歴

    def get_relative_position(self, bbox: List[float], frame_size: Tuple[int, int]) -> Tuple[float, float]:
        """
        バウンディングボックスの相対位置を計算

        Args:
            bbox (List[float]): バウンディングボックス [x1, y1, x2, y2]
            frame_size (Tuple[int, int]): フレームサイズ (width, height)

        Returns:
            Tuple[float, float]: 相対位置 (x, y)
        """
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        rel_x = center_x / frame_size[0]
        rel_y = center_y / frame_size[1]
        
        return rel_x, rel_y

    def determine_lane(self, position: Tuple[float, float]) -> str:
        """
        位置からレーンを判定

        Args:
            position (Tuple[float, float]): 相対位置 (x, y)

        Returns:
            str: レーン名
        """
        x, y = position

        # 各レーンの領域をチェック
        for lane_name, lane_coords in self.lanes.items():
            if lane_name == "jungle":
                continue  # ジャングルは特別な判定ロジックを使用

            x1, y1 = lane_coords[0]
            x2, y2 = lane_coords[1]

            if (x1 <= x <= x2) and (y1 <= y <= y2):
                return lane_name

        # ジャングルの判定（レーン以外の領域）
        return "jungle"

    def update_lane_history(self, track_id: int, lane: str):
        """
        トラックIDごとのレーン履歴を更新

        Args:
            track_id (int): トラッキングID
            lane (str): レーン名
        """
        if track_id not in self.lane_history:
            self.lane_history[track_id] = []
        self.lane_history[track_id].append(lane)

    def get_dominant_lane(self, track_id: int, window_size: int = 10) -> Optional[str]:
        """
        直近の履歴から主要なレーンを判定

        Args:
            track_id (int): トラッキングID
            window_size (int): 履歴のウィンドウサイズ

        Returns:
            Optional[str]: 主要なレーン名
        """
        if track_id not in self.lane_history:
            return None

        history = self.lane_history[track_id][-window_size:]
        if not history:
            return None

        # 最も頻出するレーンを返す
        return max(set(history), key=history.count)

    def process_tracks(self, tracks: List[Dict], frame_size: Tuple[int, int]) -> Dict[int, Dict]:
        """
        トラッキング結果からレーン情報を抽出

        Args:
            tracks (List[Dict]): トラッキング結果
            frame_size (Tuple[int, int]): フレームサイズ

        Returns:
            Dict[int, Dict]: トラックIDごとのレーン情報
        """
        lane_info = {}

        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            position = self.get_relative_position(bbox, frame_size)
            current_lane = self.determine_lane(position)
            
            self.update_lane_history(track_id, current_lane)
            dominant_lane = self.get_dominant_lane(track_id)

            lane_info[track_id] = {
                'current_lane': current_lane,
                'dominant_lane': dominant_lane,
                'position': position,
                'class_name': track['class_name']
            }

        return lane_info

    def analyze_lane_transitions(self, track_id: int) -> List[Tuple[str, str, int]]:
        """
        レーン遷移を分析

        Args:
            track_id (int): トラッキングID

        Returns:
            List[Tuple[str, str, int]]: レーン遷移のリスト (from_lane, to_lane, frame_count)
        """
        if track_id not in self.lane_history:
            return []

        transitions = []
        current_lane = self.lane_history[track_id][0]
        count = 1

        for lane in self.lane_history[track_id][1:]:
            if lane != current_lane:
                transitions.append((current_lane, lane, count))
                current_lane = lane
                count = 1
            else:
                count += 1

        # 最後のレーンの状態を追加
        transitions.append((current_lane, current_lane, count))
        return transitions

if __name__ == "__main__":
    # 使用例
    mapper = LaneMapper()
    frame_size = (1920, 1080)
    
    # トラッキング結果のサンプル
    sample_tracks = [
        {
            'track_id': 1,
            'bbox': [100, 100, 200, 200],
            'class_name': 'ally_hero'
        }
    ]
    
    lane_info = mapper.process_tracks(sample_tracks, frame_size)
    print("レーン情報:", lane_info) 