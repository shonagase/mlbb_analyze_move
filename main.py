import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import yt_dlp
import os
import tempfile
import torch
import traceback
import json
from datetime import datetime

# ページ設定
st.set_page_config(
    page_title="MLBB Move Analyzer",
    page_icon="🎮",
    layout="wide"
)

# タイトルとサブタイトル
st.title("MLBB Move Analyzer")
st.subheader("Mobile Legends: Bang Bangの試合動画を分析するツール")

# 結果保存用のディレクトリを作成
RESULTS_DIR = Path("/Users/shoyanagatomo/Documents/git/mlbb_analyze_move/data/output")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# サイドバー
with st.sidebar:
    st.header("設定")
    confidence_threshold = st.slider(
        "検出信頼度閾値",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    tracking_enabled = st.checkbox("トラッキングを有効にする", value=True)
    save_results = st.checkbox("分析結果を保存する", value=True)

# 入力方法の選択
input_method = st.radio(
    "動画の入力方法を選択",
    ["YouTubeリンク", "ファイルアップロード"]
)

video_path = None
video_info = {}

if input_method == "YouTubeリンク":
    youtube_url = st.text_input("YouTubeのURLを入力してください")
    if youtube_url:
        try:
            with st.spinner("動画をダウンロード中..."):
                # yt-dlpの設定
                temp_dir = tempfile.mkdtemp()
                video_path = os.path.join(temp_dir, "youtube_video.mp4")
                
                ydl_opts = {
                    'format': 'best[ext=mp4]',
                    'outtmpl': video_path,
                    'quiet': True,
                    'no_warnings': True,
                }
                
                # 動画情報の取得
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(youtube_url, download=False)
                    video_info = {
                        "title": info.get('title', 'Unknown'),
                        "duration": info.get('duration', 0),
                        "view_count": info.get('view_count', 0),
                        "url": youtube_url
                    }
                    st.write(f"タイトル: {video_info['title']}")
                    st.write(f"長さ: {video_info['duration']} 秒")
                    st.write(f"視聴回数: {video_info['view_count']:,} 回")
                    
                    # 動画のダウンロード
                    ydl.download([youtube_url])
                
                st.success("動画のダウンロードが完了しました！")
        
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")
            video_path = None

else:
    uploaded_file = st.file_uploader("試合の動画をアップロード", type=['mp4', 'mov', 'avi'])
    if uploaded_file is not None:
        # 一時ファイルとして保存
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        video_info = {
            "title": uploaded_file.name,
            "file_size": uploaded_file.size,
            "type": uploaded_file.type
        }

# 分析開始ボタン
if video_path and st.button("分析開始"):
    # 分析結果を保存するための辞書
    analysis_data = {
        "video_info": video_info,
        "analysis_settings": {
            "confidence_threshold": confidence_threshold,
            "tracking_enabled": tracking_enabled
        },
        "frames": []
    }
    
    # ビデオの読み込み
    video = cv2.VideoCapture(video_path)
    
    # YOLOモデルの読み込み
    with st.spinner("YOLOモデルを読み込み中..."):
        model = YOLO('yolov8n.pt')
    
    # DeepSORTトラッカーの初期化（トラッキングが有効な場合）
    tracker = None
    if tracking_enabled:
        tracker = DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )
    
    # プログレスバーの表示
    progress_text = "動画を分析中..."
    progress_bar = st.progress(0)
    
    # フレーム数の取得
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 結果を表示するプレースホルダー
    result_placeholder = st.empty()
    
    frame_count = 0
    detections = []
    
    try:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
                
            # YOLOで検出
            results = model(frame)[0]
            
            # 検出結果の処理
            frame_detections = []
            frame_data = {"frame_number": frame_count, "detections": [], "tracks": []}
            
            # boxes属性が存在し、空でない場合のみ処理
            if hasattr(results, 'boxes') and len(results.boxes) > 0:
                # テンソルをnumpy配列に変換
                boxes = results.boxes.data.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2, score, class_id = box
                    if score > confidence_threshold:
                        # DeepSORTが期待する形式に変換
                        detection = [
                            [float(x1), float(y1), float(x2), float(y2)],
                            float(score),
                            int(class_id)
                        ]
                        frame_detections.append(detection)
                        
                        # 分析結果に検出情報を追加
                        frame_data["detections"].append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "score": float(score),
                            "class_id": int(class_id)
                        })
            
            # トラッキング（有効な場合）
            if tracking_enabled and tracker is not None and frame_detections:
                tracks = tracker.update_tracks(frame_detections, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    
                    # トラッキング結果を描画
                    cv2.rectangle(frame, 
                                (int(ltrb[0]), int(ltrb[1])), 
                                (int(ltrb[2]), int(ltrb[3])), 
                                (0, 255, 0), 2)
                    cv2.putText(frame, 
                              f"ID: {track_id}", 
                              (int(ltrb[0]), int(ltrb[1])-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.9, 
                              (0, 255, 0), 
                              2)
                    
                    # 分析結果にトラッキング情報を追加
                    frame_data["tracks"].append({
                        "track_id": track_id,
                        "bbox": [float(x) for x in ltrb]
                    })
            
            # 分析結果に現在のフレームのデータを追加
            if save_results:
                analysis_data["frames"].append(frame_data)
            
            # プログレスバーの更新
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            # 結果を表示
            result_placeholder.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption="分析結果",
                use_container_width=True
            )
            
            frame_count += 1
    
    except Exception as e:
        st.error(f"分析中にエラーが発生しました: {str(e)}")
        st.error(f"エラーの詳細:\n{traceback.format_exc()}")
    
    finally:
        # クリーンアップ
        video.release()
        
        # 分析結果の保存
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = RESULTS_DIR / f"analysis_{timestamp}.json"
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=2)
            st.success(f"分析結果を保存しました: {result_file}")
        
        # 一時ファイルとディレクトリの削除
        try:
            os.remove(video_path)
            os.rmdir(os.path.dirname(video_path))
        except:
            pass
    
    st.success("分析が完了しました！")

# 保存された分析結果の表示
if st.sidebar.checkbox("保存された分析結果を表示"):
    saved_files = list(RESULTS_DIR.glob("*.json"))
    if saved_files:
        selected_file = st.sidebar.selectbox(
            "分析結果を選択",
            saved_files,
            format_func=lambda x: x.stem
        )
        
        if selected_file:
            with open(selected_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            
            st.subheader("動画情報")
            st.json(saved_data["video_info"])
            
            st.subheader("分析設定")
            st.json(saved_data["analysis_settings"])
            
            st.subheader("フレーム分析結果")
            frame_number = st.slider("フレーム番号", 0, len(saved_data["frames"])-1)
            st.json(saved_data["frames"][frame_number])
    else:
        st.sidebar.info("保存された分析結果はありません") 