import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameAnalyzer:
    def __init__(self):
        """
        ゲーム分析を行うクラス
        """
        self.lane_data = defaultdict(list)
        self.event_data = defaultdict(list)
        self.hero_stats = defaultdict(lambda: defaultdict(int))
        self.team_stats = {'ally': defaultdict(int), 'enemy': defaultdict(int)}

    def add_lane_data(self, frame_number: int, lane_info: Dict[int, Dict]):
        """
        レーン情報を追加

        Args:
            frame_number (int): フレーム番号
            lane_info (Dict[int, Dict]): レーン情報
        """
        for track_id, info in lane_info.items():
            self.lane_data[track_id].append({
                'frame': frame_number,
                'lane': info['current_lane'],
                'position': info['position'],
                'class_name': info['class_name']
            })

    def add_event_data(self, frame_number: int, events: List[Dict]):
        """
        イベント情報を追加

        Args:
            frame_number (int): フレーム番号
            events (List[Dict]): イベント情報
        """
        for event in events:
            event['frame_number'] = frame_number
            self.event_data[event['type']].append(event)

    def calculate_lane_statistics(self) -> Dict:
        """
        レーン統計を計算

        Returns:
            Dict: レーン統計情報
        """
        stats = defaultdict(lambda: defaultdict(int))
        
        for track_id, data in self.lane_data.items():
            hero_class = data[0]['class_name']
            team = 'ally' if 'ally' in hero_class else 'enemy'
            
            # レーンごとの滞在時間を計算
            lane_counts = defaultdict(int)
            for entry in data:
                lane_counts[entry['lane']] += 1
            
            total_frames = len(data)
            for lane, count in lane_counts.items():
                percentage = (count / total_frames) * 100
                stats[team][lane] += percentage
                
                # ヒーロー個別の統計も記録
                self.hero_stats[track_id]['lane_preference'][lane] = percentage

        return dict(stats)

    def analyze_rotations(self) -> Dict:
        """
        ローテーション分析を実行

        Returns:
            Dict: ローテーション分析結果
        """
        rotations = defaultdict(list)
        
        for track_id, data in self.lane_data.items():
            current_lane = data[0]['lane']
            rotation_count = 0
            
            for entry in data[1:]:
                if entry['lane'] != current_lane:
                    rotation_count += 1
                    rotations[track_id].append({
                        'from_lane': current_lane,
                        'to_lane': entry['lane'],
                        'frame': entry['frame']
                    })
                current_lane = entry['lane']
            
            # ヒーローの統計を更新
            self.hero_stats[track_id]['rotation_count'] = rotation_count

        return dict(rotations)

    def analyze_teamfights(self) -> Dict:
        """
        チームファイト分析を実行

        Returns:
            Dict: チームファイト分析結果
        """
        teamfights = self.event_data.get('teamfight', [])
        analysis = {
            'total_count': len(teamfights),
            'average_participants': np.mean([tf['participants'] for tf in teamfights]) if teamfights else 0,
            'locations': defaultdict(int),
            'win_rate': defaultdict(float)
        }
        
        for tf in teamfights:
            # 場所の分類（マップを4分割して分類）
            x, y = tf['position']
            location = f"{'top' if y < 0.5 else 'bottom'}_{'left' if x < 0.5 else 'right'}"
            analysis['locations'][location] += 1
            
            # チームごとの勝率（キル数で判定）
            if tf.get('teams', {}).get('ally', 0) > tf.get('teams', {}).get('enemy', 0):
                analysis['win_rate'][location] = (analysis['win_rate'][location] + 1) / 2

        return dict(analysis)

    def analyze_objectives(self) -> Dict:
        """
        目標物（タワーなど）の分析を実行

        Returns:
            Dict: 目標物の分析結果
        """
        tower_destructions = self.event_data.get('tower_destruction', [])
        analysis = {
            'tower_destruction_timeline': [],
            'tower_destruction_by_lane': defaultdict(int)
        }
        
        for td in tower_destructions:
            x, y = td['position']
            # タワーの位置からレーンを推定
            lane = 'top' if y < 0.33 else 'mid' if y < 0.66 else 'bottom'
            
            analysis['tower_destruction_timeline'].append({
                'frame': td['frame_number'],
                'lane': lane,
                'position': (x, y)
            })
            analysis['tower_destruction_by_lane'][lane] += 1

        return analysis

    def calculate_team_performance(self) -> Dict:
        """
        チームのパフォーマンス指標を計算

        Returns:
            Dict: チームパフォーマンス指標
        """
        kills = self.event_data.get('kill', [])
        teamfights = self.event_data.get('teamfight', [])
        
        performance = {
            'ally': {
                'kills': sum(1 for k in kills if k.get('victim_team') == 'enemy'),
                'deaths': sum(1 for k in kills if k.get('victim_team') == 'ally'),
                'teamfight_participation': sum(1 for tf in teamfights if tf.get('teams', {}).get('ally', 0) > 0),
                'objective_control': sum(1 for td in self.event_data.get('tower_destruction', []))
            },
            'enemy': {
                'kills': sum(1 for k in kills if k.get('victim_team') == 'ally'),
                'deaths': sum(1 for k in kills if k.get('victim_team') == 'enemy'),
                'teamfight_participation': sum(1 for tf in teamfights if tf.get('teams', {}).get('enemy', 0) > 0),
                'objective_control': 0  # タワー破壊の帰属が不明な場合は0とする
            }
        }
        
        return performance

    def export_analysis(self, output_path: Path) -> None:
        """
        分析結果をエクスポート

        Args:
            output_path (Path): 出力先のパス
        """
        analysis_results = {
            'lane_statistics': self.calculate_lane_statistics(),
            'rotations': self.analyze_rotations(),
            'teamfights': self.analyze_teamfights(),
            'objectives': self.analyze_objectives(),
            'team_performance': self.calculate_team_performance(),
            'hero_stats': dict(self.hero_stats)
        }
        
        # JSON形式で保存
        with open(output_path / 'analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        # CSVフォーマットでも保存
        df = pd.DataFrame(analysis_results['team_performance']).T
        df.to_csv(output_path / 'team_performance.csv')
        
        logger.info(f"分析結果を保存しました: {output_path}")

if __name__ == "__main__":
    # 使用例
    analyzer = GameAnalyzer()
    
    # サンプルデータの追加
    lane_info = {
        1: {'current_lane': 'top', 'position': (0.2, 0.1), 'class_name': 'ally_hero'},
        2: {'current_lane': 'mid', 'position': (0.5, 0.5), 'class_name': 'enemy_hero'}
    }
    analyzer.add_lane_data(0, lane_info)
    
    events = [{
        'type': 'teamfight',
        'frame': 0,
        'position': (0.5, 0.5),
        'participants': 4,
        'teams': {'ally': 2, 'enemy': 2}
    }]
    analyzer.add_event_data(0, events)
    
    # 分析の実行
    print("レーン統計:", analyzer.calculate_lane_statistics())
    print("チームファイト分析:", analyzer.analyze_teamfights()) 