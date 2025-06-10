import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import torch
import yt_dlp
import tempfile
import os
from src.minimap_analyzer import MinimapAnalyzer
import matplotlib.pyplot as plt

st.set_page_config(page_title="MLBB Game Analyzer", layout="wide")

st.title("MLBB Game Analyzer")

# MinimapAnalyzerのインスタンスを作成
@st.cache_resource
def get_analyzer():
    return MinimapAnalyzer()

analyzer = get_analyzer()

# サイドバーの設定
st.sidebar.title("Settings")

# 入力方法の選択
input_method = st.sidebar.radio("Select Input Method", ["File Upload", "YouTube URL"])

# 動画の取得
video_path = None

if input_method == "File Upload":
    # ファイルアップロード
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi'])
    if uploaded_file is not None:
        # 一時ファイルとして保存
        temp_path = Path("temp_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        video_path = temp_path

else:
    # YouTube URLの入力
    youtube_url = st.sidebar.text_input("Enter YouTube URL")
    if youtube_url:
        try:
            with st.spinner("Downloading video..."):
                # yt-dlpの設定
                ydl_opts = {
                    'format': 'best',  # 最高品質の動画をダウンロード
                    'outtmpl': 'temp_video.%(ext)s',
                    'cookiesfrombrowser': ('chrome',),  # Chromeブラウザからクッキーを取得
                }
                
                # 動画のダウンロード
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=True)
                    video_path = Path(f"temp_video.{info['ext']}")
        except Exception as e:
            st.error(f"Error downloading video: {str(e)}")

# 分析設定
st.sidebar.subheader("Analysis Settings")
detect_minimap = st.sidebar.checkbox("Detect Minimap", value=True)
track_players = st.sidebar.checkbox("Track Players", value=True)
analyze_movement = st.sidebar.checkbox("Analyze Movement Patterns", value=True)

# 移動パターン分析の設定
if analyze_movement:
    time_window = st.sidebar.slider("Time Window (frames)", 10, 100, 30)

# メイン画面
if video_path is not None and video_path.exists():
    # ビデオの読み込み
    cap = cv2.VideoCapture(str(video_path))
    
    # ビデオ情報の表示
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps
    
    st.write(f"Video Duration: {duration:.2f} seconds")
    st.write(f"FPS: {fps}")
    st.write(f"Total Frames: {total_frames}")
    
    # フレーム選択のレイアウト
    col_prev, col_slider, col_next = st.columns([1, 10, 1])
    
    # セッション状態の初期化
    if 'current_frame' not in st.session_state:
        st.session_state.current_frame = 0
    
    with col_prev:
        if st.button("◀"):
            # 前のフレームへ（最小値以下にはならないように）
            st.session_state.current_frame = max(0, st.session_state.current_frame - 1)
    
    with col_slider:
        frame_number = st.slider("Select Frame", 
                               min_value=0, 
                               max_value=total_frames-1, 
                               value=st.session_state.current_frame)
        # スライダーの値が変更されたら、current_frameを更新
        st.session_state.current_frame = frame_number
    
    with col_next:
        if st.button("▶"):
            # 次のフレームへ（最大値以上にはならないように）
            st.session_state.current_frame = min(total_frames-1, st.session_state.current_frame + 1)
    
    # 現在のフレーム番号を表示
    st.write(f"Current Frame: {st.session_state.current_frame}")
    
    # 差分分析の設定
    st.sidebar.subheader("Analysis Options")
    analyze_differences = st.sidebar.checkbox("Analyze Frame Differences", value=False)
    compare_with_base = st.sidebar.checkbox("Compare with Base Minimap", value=False)
    
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
            # 3つの連続フレームのミニマップを取得
            minimaps = []
            for i in range(3):
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame + i)
                ret, frame = cap.read()
                if ret:
                    # フレームからミニマップを抽出
                    minimap = analyzer.extract_minimap(frame)
                    minimaps.append(minimap)
            
            if len(minimaps) == 3:
                # ミニマップに対して差分分析を実行
                prob_map, changes = analyzer.analyze_frame_differences(minimaps)
                
                # 結果の可視化
                heatmap = analyzer.visualize_changes(prob_map, changes)
                
                # 結果を表示
                st.subheader("Minimap Difference Analysis")
                col3, col4 = st.columns(2)
                
                with col3:
                    st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 
                            caption="Minimap Change Probability Heatmap", 
                            use_column_width=True)
                
                with col4:
                    st.write("Change Statistics:")
                    st.write(f"Total changes detected: {changes['total_changes']}")
                    st.write("Movement Vectors:")
                    for i, mv in enumerate(changes['movement_vectors']):
                        st.write(f"Vector {i+1}:")
                        st.write(f"- Start: {mv['start']}")
                        st.write(f"- End: {mv['end']}")
                        st.write(f"- Velocity: {mv['velocity']:.2f} pixels")
                    
                    # 変化領域の詳細情報を表示
                    st.write("\nChange Areas:")
                    for i, area in enumerate(changes['change_areas']):
                        st.write(f"Area {i+1}:")
                        st.write(f"- Center: {area['center']}")
                        st.write(f"- Size: {area['area']:.2f} pixels²")
                        st.write(f"- Intensity: {area['intensity']:.2f}")
    
    # 選択したフレームを表示
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
    ret, frame = cap.read()
    if ret:
        # フレームをRGBに変換
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 分析結果の表示
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Frame")
            # 経過時間の検出と表示
            game_time = analyzer.extract_game_time(frame)
            if game_time:
                st.write(f"Game Time: {game_time}")
            
            # 時間表示領域の可視化
            frame_with_roi = frame_rgb.copy()
            height, width = frame.shape[:2]
            x = int(width * analyzer.time_roi['x'])
            y = int(height * analyzer.time_roi['y'])
            w = int(width * analyzer.time_roi['width'])
            h = int(height * analyzer.time_roi['height'])
            
            # ROIを赤枠で表示
            cv2.rectangle(frame_with_roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # ROI部分を切り出して表示
            time_area = frame_rgb[y:y+h, x:x+w]
            
            # メインフレームとROI領域を表示
            st.image(frame_with_roi, use_container_width=True)
            st.write("Time Display Region (ROI):")
            st.image(time_area, use_container_width=False)
            
            # ROIの前処理結果を表示（デバッグ用）
            time_area_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            _, time_area_binary = cv2.threshold(time_area_gray, 200, 255, cv2.THRESH_BINARY)
            kernel = np.ones((2,2), np.uint8)
            time_area_processed = cv2.morphologyEx(time_area_binary, cv2.MORPH_CLOSE, kernel)
            
            st.write("Processed Time Display Region:")
            st.image(time_area_processed, use_container_width=False)
        
        with col2:
            st.subheader("Analysis Results")
            if detect_minimap:
                try:
                    # ミニマップの検出
                    minimap = analyzer.extract_minimap(frame)
                    if minimap is not None and minimap.size > 0:  # ミニマップが正しく抽出されたか確認
                        st.write("Minimap detected!")
                        
                        # デバッグ情報の表示
                        st.write(f"Minimap shape: {minimap.shape}")
                        st.write(f"Minimap size: {minimap.size}")
                        
                        minimap_rgb = cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB)
                        st.image(minimap_rgb, caption="Detected Minimap", use_container_width=True)
                        
                        # チーム位置の検出
                        positions = analyzer.detect_team_positions(minimap)
                        if positions and (len(positions['ally']) > 0 or len(positions['enemy']) > 0):
                            st.write(f"Detected {len(positions['ally']) + len(positions['enemy'])} team positions")
                            st.write(f"- Ally positions: {len(positions['ally'])}")
                            st.write(f"- Enemy positions: {len(positions['enemy'])}")
                            
                            # 位置の可視化（経過時間も含める）
                            visualized_minimap = analyzer.visualize_minimap(minimap, positions, game_time)
                            visualized_minimap_rgb = cv2.cvtColor(visualized_minimap, cv2.COLOR_BGR2RGB)
                            st.image(visualized_minimap_rgb, caption="Team Positions", use_container_width=True)
                            
                            # 移動パターンの分析
                            if analyze_movement:
                                st.subheader("Movement Analysis")
                                patterns = analyzer.analyze_movement_patterns()
                                
                                if patterns:
                                    # 移動パターンの表示
                                    st.write("Movement Patterns:")
                                    for team in ['ally', 'enemy']:
                                        st.write(f"\n{team.capitalize()} Team:")
                                        st.write("Zone Distribution:")
                                        for zone, count in patterns[team]['zones'].items():
                                            st.write(f"- {zone}: {count}")
                        else:
                            st.write("No team positions detected in the minimap")
                    else:
                        st.write("No minimap detected in this frame")
                        st.write("Debug info:")
                        st.write(f"Frame shape: {frame.shape}")
                        st.write(f"ROI settings: {analyzer.minimap_roi}")
                except Exception as e:
                    st.error(f"Error processing minimap: {str(e)}")
            
            if track_players:
                st.write("Player Tracking: Not implemented yet")
    
    # ビデオリソースの解放
    cap.release()
    
    # 一時ファイルの削除
    video_path.unlink()
else:
    if input_method == "File Upload":
        st.write("Please upload a video file to start analysis.")
    else:
        st.write("Please enter a valid YouTube URL to start analysis.") 