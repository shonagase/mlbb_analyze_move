import streamlit as st
import cv2
import numpy as np
import os
from pathlib import Path
import yt_dlp
from src.minimap_analyzer import MinimapAnalyzer
import warnings

# 警告メッセージを非表示にする
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# tempディレクトリの作成
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)
TEMP_VIDEO_PATH = os.path.join(TEMP_DIR, "temp_video.mp4")

# 初期化
if 'current_frame' not in st.session_state:
    st.session_state.current_frame = 0

st.title("MLBB Minimap Analyzer")

# 入力方法の選択
input_method = st.sidebar.radio("Select Input Method", ["File Upload", "YouTube URL"])

video_path = None

if input_method == "File Upload":
    # ファイルアップロード
    video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi'])
    if video_file is not None:
        # 一時ファイルとして保存
        with open(TEMP_VIDEO_PATH, "wb") as f:
            f.write(video_file.read())
        video_path = TEMP_VIDEO_PATH

else:
    # YouTube URLの入力
    youtube_url = st.sidebar.text_input("Enter YouTube URL")
    if youtube_url:
        try:
            with st.spinner('Downloading video...'):
                # 一時ファイルが存在する場合は削除
                if os.path.exists(TEMP_VIDEO_PATH):
                    os.remove(TEMP_VIDEO_PATH)
                if os.path.exists(TEMP_VIDEO_PATH + '.part'):
                    os.remove(TEMP_VIDEO_PATH + '.part')
                
                ydl_opts = {
                    'format': 'best',
                    'outtmpl': TEMP_VIDEO_PATH,
                    'cookiesfrombrowser': ('chrome',),
                    'quiet': True,
                    'no_warnings': True,
                    'noprogress': True,
                }
                
                # ダウンロードを試行
                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(youtube_url, download=True)
                        if os.path.exists(TEMP_VIDEO_PATH):
                            video_path = TEMP_VIDEO_PATH
                        else:
                            st.error("Failed to download video: Output file not found")
                            video_path = None
                except Exception as e:
                    st.error(f"Error during download: {str(e)}")
                    video_path = None
        except Exception as e:
            st.error(f"Error initializing download: {str(e)}")
            video_path = None

if video_path and os.path.exists(video_path):
    try:
        # ビデオの読み込み
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # アナライザーの初期化
        analyzer = MinimapAnalyzer()
        
        # サイドバーのコントロール
        st.sidebar.subheader("Frame Control")
        frame_slider = st.sidebar.slider("Frame", 0, total_frames-1, st.session_state.current_frame)
        st.session_state.current_frame = frame_slider
        
        # 分析オプション
        st.sidebar.subheader("Analysis Options")
        analyze_differences = st.sidebar.checkbox("Analyze Frame Differences", value=False)
        compare_with_base = st.sidebar.checkbox("Compare with Base Minimap", value=False)
        analyze_time_series = st.sidebar.checkbox("Analyze Time Series", value=False)
        
        if analyze_time_series:
            duration = st.sidebar.slider("Analysis Duration (seconds)", 1, 60, 30)
            
            if st.sidebar.button("Analyze Time Series"):
                # 現在のフレームから時間表示領域を取得
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
                ret, current_frame = cap.read()
                if ret:
                    # 時間表示領域を抽出
                    height, width = current_frame.shape[:2]
                    x = int(width * analyzer.time_roi['x'])
                    y = int(height * analyzer.time_roi['y'])
                    w = int(width * analyzer.time_roi['width'])
                    h = int(height * analyzer.time_roi['height'])
                    time_area = current_frame[y:y+h, x:x+w]
                    
                    # タイトルと時間表示領域を横に並べて表示
                    title_col, time_col = st.columns([3, 1])
                    with title_col:
                        st.subheader("Time Series Analysis")
                    with time_col:
                        # 元の時間表示領域
                        st.image(cv2.cvtColor(time_area, cv2.COLOR_BGR2RGB),
                                caption="Original Time Display",
                                use_column_width=True)
                        
                        # 処理後の時間表示領域
                        gray = cv2.cvtColor(time_area, cv2.COLOR_BGR2GRAY)
                        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                        kernel = np.ones((2,2), np.uint8)
                        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                        
                        st.image(processed,
                                caption="Processed Time Display",
                                use_column_width=True)
                
                # 時系列分析を実行
                movement_map, time_stats = analyzer.analyze_time_series(
                    cap,
                    st.session_state.current_frame,
                    duration,
                    fps
                )
                
                # 結果の表示
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("### Movement Map")
                    st.image(cv2.cvtColor(movement_map, cv2.COLOR_BGR2RGB),
                            caption="Movement Over Time",
                            use_column_width=True)
                    
                    # カラーバーの説明
                    st.write("Color indicates time progression:")
                    st.write("- Blue → Cyan → Green → Yellow → Red")
                    st.write("(Earlier → Later)")
                    
                    # 現在のフレームの経過時間を表示
                    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
                    ret, current_frame = cap.read()
                    if ret:
                        game_time = analyzer.extract_game_time(current_frame)
                        if game_time:
                            st.write(f"### Current Game Time: {game_time}")
                        
                        # 分析範囲の終了時間も表示
                        end_frame = min(st.session_state.current_frame + int(duration * fps), total_frames - 1)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
                        ret, end_frame_img = cap.read()
                        if ret:
                            end_time = analyzer.extract_game_time(end_frame_img)
                            if end_time:
                                st.write(f"Analysis End Time: {end_time}")
                
                with col2:
                    st.write("### Hot Zones")
                    if time_stats["hot_zones"]:
                        hot_zone_map = cv2.cvtColor(analyzer.base_minimap.copy(), cv2.COLOR_BGR2RGB)
                        
                        for zone in time_stats["hot_zones"]:
                            # ホットゾーンを半透明の円で表示
                            overlay = hot_zone_map.copy()
                            cv2.circle(overlay, zone["center"], zone["radius"],
                                     (255, 0, 0), -1)
                            # 透明度を設定（0.3）
                            hot_zone_map = cv2.addWeighted(overlay, 0.3, hot_zone_map, 0.7, 0)
                            # 中心点を表示
                            cv2.circle(hot_zone_map, zone["center"], 3, (0, 0, 255), -1)
                        
                        st.image(hot_zone_map,
                                caption="Activity Hot Zones",
                                use_column_width=True)
                        
                        # ホットゾーンの統計情報
                        st.write("### Movement Statistics")
                        st.write(f"Total movements detected: {time_stats['total_movements']}")
                        st.write("\nHot Zone Details:")
                        for i, zone in enumerate(time_stats["hot_zones"], 1):
                            st.write(f"Zone {i}:")
                            st.write(f"- Center: {zone['center']}")
                            st.write(f"- Radius: {zone['radius']} pixels")
                            st.write(f"- Activity: {zone['point_count']} movements")
                    else:
                        st.write("No significant hot zones detected")
        
        # 選択したフレームを表示
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
        ret, frame = cap.read()
        if ret:
            # フレームからミニマップを抽出
            current_minimap = analyzer.extract_minimap(frame)
            
            # ベースミニマップとの比較
            if compare_with_base:
                diff_map, diff_stats = analyzer.compare_with_base(current_minimap)
                if diff_map is not None:
                    st.subheader("Base Minimap Comparison")
                    col_base, col_current, col_diff = st.columns(3)
                    
                    with col_base:
                        st.image(cv2.cvtColor(analyzer.base_minimap, cv2.COLOR_BGR2RGB),
                                caption="Base Minimap",
                                use_column_width=True)
                    
                    with col_current:
                        st.image(cv2.cvtColor(current_minimap, cv2.COLOR_BGR2RGB),
                                caption="Current Minimap",
                                use_column_width=True)
                    
                    with col_diff:
                        st.image(cv2.cvtColor(diff_map, cv2.COLOR_BGR2RGB),
                                caption="Difference Map",
                                use_column_width=True)
                    
                    # 差分の統計情報を表示
                    st.write("Difference Statistics:")
                    st.write(f"Total differences detected: {diff_stats['total_differences']}")
                    st.write(f"Total difference area: {diff_stats['total_diff_area']:.2f} pixels²")
                    
                    if diff_stats['total_differences'] > 0:
                        st.write("\nDifference Areas:")
                        for i, area in enumerate(diff_stats['diff_areas']):
                            st.write(f"Area {i+1}:")
                            st.write(f"- Center: {area['center']}")
                            st.write(f"- Size: {area['area']:.2f} pixels²")
            
            if analyze_differences:
                # 既存の差分分析コード
                pass
    
    finally:
        # キャプチャを解放
        if 'cap' in locals():
            cap.release()
        
        # 一時ファイルの削除
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception as e:
            st.warning(f"Failed to delete temporary file: {str(e)}")

else:
    st.info("Please upload a video file or enter a valid YouTube URL to start analysis.") 