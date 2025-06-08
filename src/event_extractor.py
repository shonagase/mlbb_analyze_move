import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
from configs.config import EVENT_THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventExtractor:
    def __init__(self, thresholds: Dict = EVENT_THRESHOLDS):
        """
        ゲーム内イベントを検出するクラス

        Args:
            thresholds (Dict): イベント検出の閾値設定
        """
        self.thresholds = thresholds
        self.event_history = defaultdict(list)
        self.hero_positions = defaultdict(list)

    def update_hero_positions(self, frame_number: int, tracks: List[Dict]):
        """
        ヒーローの位置情報を更新

        Args:
            frame_number (int): フレーム番号
            tracks (List[Dict]): トラッキング結果
        """
        for track in tracks:
            if 'hero' in track['class_name']:
                track_id = track['track_id']
                bbox = track['bbox']
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                self.hero_positions[track_id].append((frame_number, center))

    def detect_teamfight(self, frame_number: int, tracks: List[Dict], radius: float = 300) -> Optional[Dict]:
        """
        チームファイトを検出

        Args:
            frame_number (int): フレーム番号
            tracks (List[Dict]): トラッキング結果
            radius (float): チームファイト判定の範囲

        Returns:
            Optional[Dict]: チームファイトイベント情報
        """
        hero_positions = []
        for track in tracks:
            if 'hero' in track['class_name']:
                bbox = track['bbox']
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                hero_positions.append({
                    'position': center,
                    'team': 'ally' if 'ally' in track['class_name'] else 'enemy'
                })

        if len(hero_positions) >= self.thresholds['teamfight']:
            # チームごとのヒーロー数をカウント
            teams = {'ally': 0, 'enemy': 0}
            for hero in hero_positions:
                teams[hero['team']] += 1

            # 両チームが参加している場合のみチームファイトと判定
            if teams['ally'] > 0 and teams['enemy'] > 0:
                # チームファイトの中心位置を計算
                center_x = np.mean([h['position'][0] for h in hero_positions])
                center_y = np.mean([h['position'][1] for h in hero_positions])

                return {
                    'type': 'teamfight',
                    'frame': frame_number,
                    'position': (center_x, center_y),
                    'participants': len(hero_positions),
                    'teams': teams
                }

        return None

    def detect_kill(self, frame_number: int, tracks: List[Dict], prev_tracks: List[Dict]) -> Optional[Dict]:
        """
        キルイベントを検出

        Args:
            frame_number (int): フレーム番号
            tracks (List[Dict]): 現在のトラッキング結果
            prev_tracks (List[Dict]): 前フレームのトラッキング結果

        Returns:
            Optional[Dict]: キルイベント情報
        """
        # 前フレームと現在のフレームのトラックIDを比較
        prev_ids = {t['track_id'] for t in prev_tracks if 'hero' in t['class_name']}
        current_ids = {t['track_id'] for t in tracks if 'hero' in t['class_name']}

        # 消失したヒーローを検出
        killed_ids = prev_ids - current_ids
        if killed_ids:
            # キルされたヒーローの情報を取得
            for killed_id in killed_ids:
                killed_hero = next((t for t in prev_tracks if t['track_id'] == killed_id), None)
                if killed_hero:
                    return {
                        'type': 'kill',
                        'frame': frame_number,
                        'position': ((killed_hero['bbox'][0] + killed_hero['bbox'][2]) / 2,
                                   (killed_hero['bbox'][1] + killed_hero['bbox'][3]) / 2),
                        'victim_team': 'ally' if 'ally' in killed_hero['class_name'] else 'enemy'
                    }

        return None

    def detect_tower_destruction(self, frame_number: int, tracks: List[Dict], prev_tracks: List[Dict]) -> Optional[Dict]:
        """
        タワー破壊を検出

        Args:
            frame_number (int): フレーム番号
            tracks (List[Dict]): 現在のトラッキング結果
            prev_tracks (List[Dict]): 前フレームのトラッキング結果

        Returns:
            Optional[Dict]: タワー破壊イベント情報
        """
        # タワーのトラックIDを比較
        prev_tower_ids = {t['track_id'] for t in prev_tracks if t['class_name'] == 'tower'}
        current_tower_ids = {t['track_id'] for t in tracks if t['class_name'] == 'tower'}

        # 消失したタワーを検出
        destroyed_ids = prev_tower_ids - current_tower_ids
        if destroyed_ids:
            # 破壊されたタワーの情報を取得
            for tower_id in destroyed_ids:
                tower = next((t for t in prev_tracks if t['track_id'] == tower_id), None)
                if tower:
                    return {
                        'type': 'tower_destruction',
                        'frame': frame_number,
                        'position': ((tower['bbox'][0] + tower['bbox'][2]) / 2,
                                   (tower['bbox'][1] + tower['bbox'][3]) / 2)
                    }

        return None

    def process_frame(self, frame_number: int, tracks: List[Dict], prev_tracks: List[Dict]) -> List[Dict]:
        """
        フレームからイベントを検出

        Args:
            frame_number (int): フレーム番号
            tracks (List[Dict]): 現在のトラッキング結果
            prev_tracks (List[Dict]): 前フレームのトラッキング結果

        Returns:
            List[Dict]: 検出されたイベントのリスト
        """
        events = []

        # ヒーローの位置を更新
        self.update_hero_positions(frame_number, tracks)

        # チームファイトの検出
        teamfight = self.detect_teamfight(frame_number, tracks)
        if teamfight:
            events.append(teamfight)

        # キルの検出
        kill = self.detect_kill(frame_number, tracks, prev_tracks)
        if kill:
            events.append(kill)

        # タワー破壊の検出
        tower = self.detect_tower_destruction(frame_number, tracks, prev_tracks)
        if tower:
            events.append(tower)

        # 検出されたイベントを履歴に追加
        for event in events:
            self.event_history[event['type']].append(event)

        return events

    def get_event_summary(self) -> Dict:
        """
        イベント履歴のサマリーを取得

        Returns:
            Dict: イベントの統計情報
        """
        summary = {}
        for event_type, events in self.event_history.items():
            summary[event_type] = {
                'count': len(events),
                'frames': [e['frame'] for e in events],
                'positions': [e['position'] for e in events]
            }
            
            if event_type == 'kill':
                # キルイベントの詳細な分析
                team_kills = {'ally': 0, 'enemy': 0}
                for event in events:
                    if event['victim_team'] == 'ally':
                        team_kills['enemy'] += 1
                    else:
                        team_kills['ally'] += 1
                summary[event_type]['team_kills'] = team_kills

        return summary

if __name__ == "__main__":
    # 使用例
    extractor = EventExtractor()
    
    # サンプルのトラッキングデータ
    current_tracks = [
        {'track_id': 1, 'class_name': 'ally_hero', 'bbox': [100, 100, 200, 200]},
        {'track_id': 2, 'class_name': 'enemy_hero', 'bbox': [150, 150, 250, 250]}
    ]
    
    prev_tracks = [
        {'track_id': 1, 'class_name': 'ally_hero', 'bbox': [100, 100, 200, 200]},
        {'track_id': 2, 'class_name': 'enemy_hero', 'bbox': [150, 150, 250, 250]},
        {'track_id': 3, 'class_name': 'tower', 'bbox': [300, 300, 400, 400]}
    ]
    
    events = extractor.process_frame(0, current_tracks, prev_tracks)
    print("検出されたイベント:", events) 