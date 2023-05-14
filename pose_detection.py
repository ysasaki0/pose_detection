import streamlit as st
import numpy as np
import tempfile
import mediapipe as mp
import cv2
import imageio
import os

def reencode_video(input_path, output_path, speed_factor):
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(output_path, fps=fps * speed_factor)
    for frame in reader:
        writer.append_data(frame)
    writer.close()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=1, color=(0, 255, 0))
mark_drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(0, 0, 255))

upload_file = st.sidebar.file_uploader("動画アップロード", type='mp4')

speed_factor = st.sidebar.slider("再生速度の調整", min_value=0.25, max_value=2.0, value=1.0, step=0.25)

if upload_file is not None:
    # tempファイルに保存
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(upload_file.read())
        temp_file_path = temp_file.name

    with open(temp_file_path, 'rb') as file:
        cap_file = cv2.VideoCapture(temp_file.name)

        # 動画のFPSと解像度を取得
        fps = int(cap_file.get(cv2.CAP_PROP_FPS))
        width  = int(cap_file.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
        macro_block_size = 16
        size = (int((width * 0.5) // macro_block_size) * macro_block_size, int((height * 0.5) // macro_block_size) * macro_block_size)

        # 結果を保存するための一時的な動画ファイル
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
            temp_video_path = temp_video_file.name

        # 動画書き込み用のオブジェクトを作成
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, size)

        with mp_holistic.Holistic(static_image_mode=False) as holistic_detection:
            while cap_file.isOpened():
                success, image = cap_file.read()
                if not success:
                    break

                # 画像サイズを調整
                image = cv2.resize(image, dsize=size)

                results = holistic_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=results.pose_landmarks,
                    connections=mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mark_drawing_spec,
                    connection_drawing_spec=mesh_drawing_spec
                )
                # 画像をBGR形式に変換
                # image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # 動画にフレームを書き込む
                out.write(image)
         # リソースを解放
        cap_file.release()
        out.release()

    # 一時的に保存された動画ファイルを再エンコード
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_reencoded_video:
        reencoded_video_path = temp_reencoded_video.name
        reencode_video(temp_video_path, reencoded_video_path, speed_factor)


    #再エンコードされた動画ファイルをストリーミング
    with open(reencoded_video_path, "rb") as f:
        st.video(f.read(), format="video/mp4", start_time=0)

# アプリケーションが終了した後に一時ファイルを削除
    os.remove(temp_file_path)
    os.remove(temp_video_path)
    os.remove(reencoded_video_path)


