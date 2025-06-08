import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimapAnalyzer:
    def __init__(self):
        """
        MLBBのミニマップを分析するクラス
        """
        # ミニマップのROI（画面の左上）
        self.minimap_roi = {
            'x': 0.0,  # 左端から0%（完全に左端から）
            'y': 0.0,  # 上端から0%（完全に上端から）
            'width': 0.18,  # 画面幅の18%
            'height': 0.30  # 画面高さの30%
        }
        
        # チームの色定義（HSV形式）
        self.team_colors = {
            'ally': {
                'lower': np.array([90, 100, 100]),  # 青色系
                'upper': np.array([130, 255, 255])
            },
            'enemy': {
                'lower': np.array([0, 100, 100]),   # 赤色系
                'upper': np.array([10, 255, 255])
            }
        }
        
        # 移動履歴の保存用
        self.movement_history = {
            'ally': [],
            'enemy': []
        }
        
        # ミニマップのサイズ（初期値）
        self.minimap_width = 0
        self.minimap_height = 0

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
        
        # HSVに変換
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        
        for team in ['ally', 'enemy']:
            # チームの色でマスク作成
            mask = cv2.inRange(hsv, self.team_colors[team]['lower'], self.team_colors[team]['upper'])
            
            # ノイズ除去
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # 輪郭検出
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 各輪郭の中心を計算
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    positions[team].append((cx, cy))
        
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

    def visualize_minimap(self, minimap: np.ndarray, positions: Dict[str, List[Tuple[int, int]]]) -> np.ndarray:
        """
        ミニマップ上の検出結果を可視化

        Args:
            minimap (np.ndarray): ミニマップ画像
            positions (Dict[str, List[Tuple[int, int]]]): チームごとの位置座標

        Returns:
            np.ndarray: 可視化されたミニマップ
        """
        vis_map = minimap.copy()
        
        # チームごとに異なる色で描画
        colors = {
            'ally': (0, 255, 0),  # 緑
            'enemy': (0, 0, 255)  # 赤
        }
        
        for team, pos_list in positions.items():
            for pos in pos_list:
                cv2.circle(vis_map, pos, 3, colors[team], -1)
                cv2.circle(vis_map, pos, 5, colors[team], 1)
        
        return vis_map

    def process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, Dict[str, List[Tuple[int, int]]]]:
        """
        フレームを処理してミニマップ分析を実行

        Args:
            frame (np.ndarray): 入力フレーム
            frame_number (int): フレーム番号

        Returns:
            Tuple[np.ndarray, Dict[str, List[Tuple[int, int]]]]: 
            可視化されたミニマップと検出された位置情報
        """
        # ミニマップ領域の抽出
        minimap = self.extract_minimap(frame)
        
        # チームの位置を検出
        positions = self.detect_team_positions(minimap)
        
        # 移動履歴の更新
        self.update_movement_history(positions, frame_number)
        
        # 結果の可視化
        vis_map = self.visualize_minimap(minimap, positions)
        
        return vis_map, positions 