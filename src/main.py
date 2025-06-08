import streamlit as st
from pathlib import Path
import logging
from typing import Optional
import cv2
import numpy as np

# グローバル変数として必要なモジュールを保持
_modules = None

def load_modules():
    global _modules
    if _modules is None:
        from downloader import VideoDownloader
        from frame_extractor import FrameExtractor
        from detection import MLBBDetector
        from lane_mapper import LaneMapper
        from event_extractor import EventExtractor
        from analyzer import GameAnalyzer
        from configs.config import OUTPUT_DIR, MODEL_DIR
        _modules = (VideoDownloader, FrameExtractor, MLBBDetector, LaneMapper, 
                   EventExtractor, GameAnalyzer, OUTPUT_DIR, MODEL_DIR)
    return _modules

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLBBAnalyzer:
    def __init__(self):
        """
        MLBBの試合分析アプリケーション
        """
        (VideoDownloader, FrameExtractor, MLBBDetector, LaneMapper, 
         EventExtractor, GameAnalyzer, self.OUTPUT_DIR, self.MODEL_DIR) = load_modules()
         
        self.downloader = VideoDownloader()
        self.frame_extractor = FrameExtractor()
        self.detector = MLBBDetector()
        self.lane_mapper = LaneMapper()
        self.event_extractor = EventExtractor()
        self.analyzer = GameAnalyzer()

    def process_video(self, video_path: Path) -> None:
        """
        動画を処理して分析を実行

        Args:
            video_path (Path): 動画ファイルのパス
        """
        # フレームの抽出
        frames_dir = self.frame_extractor.extract_frames(video_path)
        if not frames_dir:
            logger.error("フレームの抽出に失敗しました")
            return

        # フレーム処理
        frame_files = sorted(frames_dir.glob("*.jpg"))
        prev_tracks = []
        frame_size = None

        for i, frame_file in enumerate(frame_files):
            # フレームの読み込み
            frame = cv2.imread(str(frame_file))
            if frame_size is None:
                frame_size = (frame.shape[1], frame.shape[0])

            # 物体検出とトラッキング
            detections, tracks = self.detector.process_frame(frame)

            # レーンマッピング
            lane_info = self.lane_mapper.process_tracks(tracks, frame_size)

            # イベント検出
            events = self.event_extractor.process_frame(i, tracks, prev_tracks)

            # データの蓄積
            self.analyzer.add_lane_data(i, lane_info)
            self.analyzer.add_event_data(i, events)

            prev_tracks = tracks

            if i % 10 == 0:
                logger.info(f"進捗: {i}/{len(frame_files)} フレーム処理完了")

        # 分析結果のエクスポート
        self.analyzer.export_analysis(self.OUTPUT_DIR)

def main():
    st.title("Mobile Legends: Bang Bang 試合分析")

    # 必要なモジュールをロード
    (VideoDownloader, FrameExtractor, MLBBDetector, LaneMapper, 
     EventExtractor, GameAnalyzer, OUTPUT_DIR, MODEL_DIR) = load_modules()

    # サイドバー
    st.sidebar.header("設定")
    analysis_mode = st.sidebar.selectbox(
        "分析モード",
        ["YouTube URL", "ローカルファイル"]
    )

    if analysis_mode == "YouTube URL":
        url = st.sidebar.text_input("YouTube URL")
        if st.sidebar.button("分析開始") and url:
            with st.spinner("動画をダウンロード中..."):
                downloader = VideoDownloader()
                video_path = downloader.download_video(url)
                if video_path:
                    st.success("動画のダウンロードが完了しました")
                    process_video(video_path)
                else:
                    st.error("動画のダウンロードに失敗しました")

    else:
        uploaded_file = st.sidebar.file_uploader("動画ファイルをアップロード", type=['mp4'])
        if uploaded_file is not None:
            # 一時ファイルとして保存
            temp_path = Path("temp.mp4")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.sidebar.button("分析開始"):
                process_video(temp_path)

    # メイン画面
    st.header("分析結果")
    
    # 分析結果の表示（analysis_results.jsonが存在する場合）
    results_path = OUTPUT_DIR / "analysis_results.json"
    if results_path.exists():
        import json
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # レーン統計
        st.subheader("レーン統計")
        st.write(results["lane_statistics"])
        
        # チームファイト分析
        st.subheader("チームファイト分析")
        st.write(results["teamfights"])
        
        # 目標物の分析
        st.subheader("目標物の分析")
        st.write(results["objectives"])
        
        # チームパフォーマンス
        st.subheader("チームパフォーマンス")
        st.write(results["team_performance"])

def process_video(video_path: Path):
    """
    動画を処理して分析を実行（プログレスバー付き）

    Args:
        video_path (Path): 動画ファイルのパス
    """
    analyzer = MLBBAnalyzer()
    
    with st.spinner("分析を実行中..."):
        try:
            # 動画の読み込み
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # プログレスバーの作成
            progress_bar = st.progress(0)
            frame_placeholder = st.empty()
            minimap_placeholder = st.empty()
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # フレームの処理
                detections, tracks, minimap_results = analyzer.detector.process_frame(frame)
                
                # 結果の可視化
                vis_frame = analyzer.detector.visualize_results(frame, detections, tracks, minimap_results)
                
                # フレームとミニマップの表示
                frame_placeholder.image(vis_frame, channels="BGR", use_column_width=True)
                minimap_placeholder.image(minimap_results[0], channels="BGR", caption="ミニマップ分析", use_column_width=True)
                
                # プログレスバーの更新
                progress = int((frame_count + 1) / total_frames * 100)
                progress_bar.progress(progress)
                
                frame_count += 1
            
            cap.release()
            
            # 移動パターンの分析結果を表示
            st.subheader("移動パターン分析")
            patterns = analyzer.detector.minimap_analyzer.analyze_movement_patterns()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("味方チームの移動パターン")
                st.write(patterns['ally'])
            
            with col2:
                st.write("敵チームの移動パターン")
                st.write(patterns['enemy'])
            
            st.success("分析が完了しました")
            
        except Exception as e:
            st.error(f"分析中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main() 