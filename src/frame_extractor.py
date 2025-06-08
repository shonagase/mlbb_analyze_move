import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
from configs.config import FRAME_DIR, FRAME_EXTRACTION_FPS, FRAME_QUALITY
from .image_enhancer import ImageEnhancer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameExtractor:
    def __init__(self, output_dir: Path = FRAME_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = FRAME_EXTRACTION_FPS
        self.quality = FRAME_QUALITY
        self.enhancer = ImageEnhancer()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        フレームの前処理を行う

        Args:
            frame (np.ndarray): 入力フレーム

        Returns:
            np.ndarray: 処理済みフレーム
        """
        # 解像度を1080pに統一
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        
        # 画質分析に基づいて必要な処理を決定
        apply_sr, apply_clahe, adjust_brightness = self.enhancer.get_enhancement_params(frame)
        
        # 画質改善処理を適用
        frame = self.enhancer.enhance_image(
            frame,
            apply_sr=apply_sr,
            apply_clahe=apply_clahe,
            adjust_brightness=adjust_brightness
        )
        
        # ノイズ除去
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        # シャープネス強調
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        # コントラストと明るさの調整
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        
        return frame

    def extract_frames(self, video_path: Path) -> Optional[Path]:
        """
        動画からフレームを抽出

        Args:
            video_path (Path): 動画ファイルのパス

        Returns:
            Optional[Path]: フレームが保存されたディレクトリのパス
        """
        try:
            # 動画の読み込み
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error("動画ファイルを開けませんでした")
                return None

            # 動画の情報を取得
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(video_fps / self.fps)

            # フレーム保存用のディレクトリを作成
            frames_dir = self.output_dir / video_path.stem
            frames_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"フレーム抽出を開始: {video_path.name}")
            logger.info(f"総フレーム数: {total_frames}, FPS: {video_fps}, 抽出間隔: {frame_interval}")

            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # フレームの処理
                    processed_frame = self.process_frame(frame)
                    
                    frame_path = frames_dir / f"frame_{saved_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), processed_frame, 
                              [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                    saved_count += 1

                frame_count += 1

                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"進捗: {progress:.1f}%")

            cap.release()
            logger.info(f"フレーム抽出完了: {saved_count}フレームを保存")
            return frames_dir

        except Exception as e:
            logger.error(f"フレーム抽出中にエラーが発生: {str(e)}")
            return None

    def get_frame_metadata(self, video_path: Path) -> Optional[dict]:
        """
        動画のメタデータを取得

        Args:
            video_path (Path): 動画ファイルのパス

        Returns:
            Optional[dict]: メタデータ情報
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None

            metadata = {
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
            }

            cap.release()
            return metadata

        except Exception as e:
            logger.error(f"メタデータ取得中にエラーが発生: {str(e)}")
            return None

if __name__ == "__main__":
    # 使用例
    extractor = FrameExtractor()
    video_path = Path("path/to/your/video.mp4")
    frames_dir = extractor.extract_frames(video_path)
    if frames_dir:
        print(f"フレームの保存先: {frames_dir}") 