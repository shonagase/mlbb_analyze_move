from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from configs.config import MODEL_DIR, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, TRACKING_CONFIG
from minimap_analyzer import MinimapAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLBBDetector:
    def __init__(self, model_path: Optional[Path] = None):
        """
        MLBBの物体検出とトラッキングを行うクラス

        Args:
            model_path (Optional[Path]): YOLOv8モデルのパス
        """
        from ultralytics import YOLO
        from deep_sort_realtime.deepsort_tracker import DeepSort
        
        if model_path is None:
            model_path = MODEL_DIR / "mlbb_yolov8.pt"
        
        self.model = YOLO(str(model_path))
        self.tracker = DeepSort(
            max_age=TRACKING_CONFIG['max_age'],
            n_init=TRACKING_CONFIG['min_hits']
        )
        
        # ヒーローのクラスIDとクラス名のマッピング
        self.class_names = {
            0: "ally_hero",
            1: "enemy_hero",
            2: "tower",
            3: "minion",
            4: "jungle_monster"
        }
        
        # ミニマップ分析機能を追加
        self.minimap_analyzer = MinimapAnalyzer()

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        フレーム内のオブジェクトを検出

        Args:
            frame (np.ndarray): 入力フレーム

        Returns:
            List[Dict]: 検出されたオブジェクトのリスト
        """
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': score,
                'class_id': int(class_id),
                'class_name': self.class_names.get(int(class_id), "unknown")
            })

        return detections

    def track_objects(self, frame: np.ndarray, detections: List[Dict]) -> List[Dict]:
        """
        検出されたオブジェクトをトラッキング

        Args:
            frame (np.ndarray): 入力フレーム
            detections (List[Dict]): 検出されたオブジェクト

        Returns:
            List[Dict]: トラッキング結果
        """
        detection_list = []
        for det in detections:
            bbox = det['bbox']
            detection_list.append(([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                                 det['confidence'],
                                 det['class_name']))

        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        tracking_results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            try:
                # トラックの現在位置を取得
                ltrb = track.to_tlbr()  # [top, left, bottom, right]
                if isinstance(ltrb, (tuple, list, np.ndarray)) and len(ltrb) == 4:
                    tracking_results.append({
                        'track_id': track_id,
                        'bbox': [float(ltrb[0]), float(ltrb[1]), float(ltrb[2]), float(ltrb[3])],
                        'class_name': track.get_det_class()
                    })
                else:
                    logger.warning(f"Invalid bounding box format for track {track_id}: {ltrb}")
            except Exception as e:
                logger.error(f"Error processing track {track_id}: {str(e)}")
                continue

        return tracking_results

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], List[Dict], Tuple[np.ndarray, Dict]]:
        """
        フレームの処理（検出とトラッキング）

        Args:
            frame (np.ndarray): 入力フレーム

        Returns:
            Tuple[List[Dict], List[Dict], Tuple[np.ndarray, Dict]]: 
            検出結果、トラッキング結果、ミニマップ分析結果のタプル
        """
        detections = self.detect_objects(frame)
        tracks = self.track_objects(frame, detections)
        
        # ミニマップ分析を実行
        minimap_vis, minimap_positions = self.minimap_analyzer.process_frame(frame, len(self.minimap_analyzer.movement_history['ally']))
        
        return detections, tracks, (minimap_vis, minimap_positions)

    def visualize_results(self, frame: np.ndarray, detections: List[Dict], tracks: List[Dict], minimap_results: Tuple[np.ndarray, Dict]) -> np.ndarray:
        """
        検出とトラッキング結果を可視化

        Args:
            frame (np.ndarray): 入力フレーム
            detections (List[Dict]): 検出結果
            tracks (List[Dict]): トラッキング結果
            minimap_results (Tuple[np.ndarray, Dict]): ミニマップ分析結果

        Returns:
            np.ndarray: 可視化されたフレーム
        """
        vis_frame = frame.copy()

        # 検出結果の描画
        for det in detections:
            bbox = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = (0, 255, 0) if 'ally' in det['class_name'] else (0, 0, 255)
            
            cv2.rectangle(vis_frame, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         color, 2)
            cv2.putText(vis_frame, label, 
                       (int(bbox[0]), int(bbox[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # ミニマップ分析結果の描画
        minimap_vis = minimap_results[0]
        height, width = frame.shape[:2]
        x = int(width * self.minimap_analyzer.minimap_roi['x'])
        y = int(height * self.minimap_analyzer.minimap_roi['y'])
        vis_frame[y:y+minimap_vis.shape[0], x:x+minimap_vis.shape[1]] = minimap_vis
        
        return vis_frame

if __name__ == "__main__":
    # 使用例
    detector = MLBBDetector()
    frame = cv2.imread("path/to/your/frame.jpg")
    detections, tracks, minimap_results = detector.process_frame(frame)
    vis_frame = detector.visualize_results(frame, detections, tracks, minimap_results)
    cv2.imshow("Results", vis_frame)
    cv2.waitKey(0) 