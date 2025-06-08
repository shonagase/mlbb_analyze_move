import yt_dlp
from pathlib import Path
from typing import Optional
import logging
from configs.config import VIDEO_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoDownloader:
    def __init__(self, output_dir: Path = VIDEO_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.ydl_opts = {
            'format': 'bestvideo[height=1080][ext=mp4]+bestaudio[ext=m4a]/best[height=1080]',  # 1080p固定
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

    def download_video(self, url: str) -> Optional[Path]:
        """
        YouTubeのURLから動画をダウンロード

        Args:
            url (str): YouTubeのURL

        Returns:
            Optional[Path]: ダウンロードした動画のパス。失敗した場合はNone
        """
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # 動画情報の取得
                info = ydl.extract_info(url, download=False)
                video_title = info['title']
                video_ext = info['ext']
                
                # ダウンロード実行
                logger.info(f"動画のダウンロードを開始: {video_title}")
                ydl.download([url])
                
                # ダウンロードされたファイルのパスを取得
                video_path = self.output_dir / f"{video_title}.{video_ext}"
                
                if video_path.exists():
                    logger.info(f"ダウンロード完了: {video_path}")
                    return video_path
                else:
                    logger.error("ダウンロードされたファイルが見つかりません")
                    return None
                
        except Exception as e:
            logger.error(f"動画のダウンロードに失敗: {str(e)}")
            return None

    def get_video_info(self, url: str) -> Optional[dict]:
        """
        動画の情報を取得

        Args:
            url (str): YouTubeのURL

        Returns:
            Optional[dict]: 動画の情報。失敗した場合はNone
        """
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    'title': info['title'],
                    'duration': info['duration'],
                    'view_count': info.get('view_count', 0),
                    'like_count': info.get('like_count', 0),
                    'upload_date': info.get('upload_date', ''),
                    'channel': info.get('channel', ''),
                    'description': info.get('description', '')
                }
        except Exception as e:
            logger.error(f"動画情報の取得に失敗: {str(e)}")
            return None

if __name__ == "__main__":
    # 使用例
    downloader = VideoDownloader()
    url = "https://www.youtube.com/watch?v=example"
    video_path = downloader.download_video(url)
    if video_path:
        print(f"ダウンロードした動画: {video_path}") 