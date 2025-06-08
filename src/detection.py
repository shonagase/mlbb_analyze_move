from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from configs.config import MODEL_DIR, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, TRACKING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLBBDetector:
    def __init__(self, model_path: Optional[Path] = None):
        """
        MLBBの物体検出とトラッキングを行うクラス

        Args:
            model_path (Optional[Path]): YOLOv8モデルのパス
        """
        if model_path is None:
            model_path = MODEL_DIR / "mlbb_yolov8.pt"
        
        self.model = YOLO(str(model_path))
        self.tracker = DeepSort(
            max_age=TRACKING_CONFIG['max_age'],
            n_init=TRACKING_CONFIG['min_hits'],
            iou_threshold=TRACKING_CONFIG['iou_threshold']
        )
        
        # ヒーローのクラスIDとクラス名のマッピング
        self.class_names = {
            0: "ally_hero",
            1: "enemy_hero",
            2: "tower",
            3: "minion",
            4: "jungle_monster"
        }

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
            ltrb = track.to_ltrb()
            tracking_results.append({
                'track_id': track_id,
                'bbox': [ltrb[0], ltrb[1], ltrb[2], ltrb[3]],
                'class_name': track.get_det_class()
            })

        return tracking_results

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """
        フレームの処理（検出とトラッキング）

        Args:
            frame (np.ndarray): 入力フレーム

        Returns:
            Tuple[List[Dict], List[Dict]]: 検出結果とトラッキング結果のタプル
        """
        detections = self.detect_objects(frame)
        tracks = self.track_objects(frame, detections)
        return detections, tracks

    def visualize_results(self, frame: np.ndarray, detections: List[Dict], tracks: List[Dict]) -> np.ndarray:
        """
        検出とトラッキング結果を可視化

        Args:
            frame (np.ndarray): 入力フレーム
            detections (List[Dict]): 検出結果
            tracks (List[Dict]): トラッキング結果

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

        # トラッキング結果の描画
        for track in tracks:
            bbox = track['bbox']
            label = f"ID: {track['track_id']}"
            color = (255, 0, 0)
            
            cv2.rectangle(vis_frame,
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])),
                         color, 2)
            cv2.putText(vis_frame, label,
                       (int(bbox[0]), int(bbox[1] - 25)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis_frame

if __name__ == "__main__":
    # 使用例
    detector = MLBBDetector()
    frame = cv2.imread("path/to/your/frame.jpg")
    detections, tracks = detector.process_frame(frame)
    vis_frame = detector.visualize_results(frame, detections, tracks)
    cv2.imshow("Results", vis_frame)
    cv2.waitKey(0) 