import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor
from configs.config import FRAME_DIR, FRAME_EXTRACTION_FPS, FRAME_QUALITY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FrameExtractor:
    def __init__(self, output_dir: Path = FRAME_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = FRAME_EXTRACTION_FPS
        self.quality = FRAME_QUALITY

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
                    frame_path = frames_dir / f"frame_{saved_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame, 
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

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        フレームの前処理を行う

        Args:
            frame (np.ndarray): 入力フレーム

        Returns:
            np.ndarray: 処理済みフレーム
        """
        # リサイズ（必要に応じて）
        # frame = cv2.resize(frame, (1280, 720))
        
        # 色空間の変換（必要に応じて）
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame

if __name__ == "__main__":
    # 使用例
    extractor = FrameExtractor()
    video_path = Path("path/to/your/video.mp4")
    frames_dir = extractor.extract_frames(video_path)
    if frames_dir:
        print(f"フレームの保存先: {frames_dir}") 