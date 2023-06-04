import streamlit as st
import sounddevice as sd
import tensorflow as tf
import numpy as np
import threading
import os
import emoji
import pickle
import tempfile
import time
from scipy.io.wavfile import write
from page2 import extract_features

def page3():
    st.title("Emotion Recognition from Speech")
    st.write("이 페이지에서는 AI가 당신의 목소리에서 감정을 판단하도록 합니다.")

    st.write("테스트하려는 오디오 파일을 업로드하세요:")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

    audio_file_path = None
    if "audio_file_path" not in st.session_state:
        st.session_state.audio_file_path = None
        
    if uploaded_file is None:
        st.write("오디오 파일을 업로드하거나 아래 버튼을 눌러 녹음을 시작하세요.")
        st.markdown('<span style="background-color: yellow">**오늘 하루를 떠올려보며 생각나는 일을 5초이상 10초이하로 말해주세요.**</span>', unsafe_allow_html=True)

        # Record audio from the microphone
        fs = 44100  # Sample rate
        seconds = 10  # Duration of recording

        if st.button('🔴 Record'):
            myrecording = np.zeros((fs * seconds, 2))  # Pre-allocate array for the recording

            def record_audio():
                nonlocal myrecording  # Use the array from the outer scope
                recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
                sd.wait()  # Wait until recording is finished
                myrecording[:] = recording[:]  # Copy the recording data into the pre-allocated array

            # Start the recording in a separate thread
            record_thread = threading.Thread(target=record_audio)
            record_thread.start()

            # Show a progress bar while recording
            progress_bar = st.progress(0)
            for i in range(seconds):
                # Wait for one second
                time.sleep(1)

                # Update the progress bar
                progress = (i + 1) / seconds
                progress_bar.progress(progress)

            # Wait for the recording thread to finish
            record_thread.join()

            myrecording_int16 = np.clip((myrecording * 32767), -32768, 32767).astype(np.int16)  # Convert to 16-bit data with clipping
            # Save to temporary file and use this as our audio file path
            st.session_state.audio_file_path = 'temp.wav'
            write(st.session_state.audio_file_path, fs, myrecording_int16)
            st.audio(st.session_state.audio_file_path)  # Pass the file path directly to st.audio
    else:
        st.session_state.audio_file_path = os.path.join(tempfile.gettempdir(), uploaded_file.name)
        with open(st.session_state.audio_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # 모델 경로 입력
    st.write("모델이 저장된 폴더의 경로를 입력하세요:")
    model_folder_path = st.text_input("Model folder path", "")

    if not os.path.isdir(model_folder_path):
        st.error("입력된 경로에 폴더가 없습니다. 경로를 확인하고 다시 시도해주세요.")
        return

    model_path = os.path.join(model_folder_path, 'saved_model')

    if not os.path.isdir(model_path):
        st.error("입력된 폴더에 모델이 없습니다. 경로를 확인하고 다시 시도해주세요.")
        return

    # 모델 로드
    model = tf.keras.models.load_model(model_path)

    if st.session_state.audio_file_path is not None:
    # 오디오 파일의 특징을 추출
        test_features = extract_features([st.session_state.audio_file_path])
        with open(os.path.join(model_folder_path, 'min_timesteps.pkl'), 'rb') as f:
            min_timesteps = pickle.load(f)
        test_features = test_features[:, :min_timesteps, :]

        # 모델로 예측 수행
        prediction = model.predict(test_features)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # 레이블 복원
        with open(os.path.join(model_folder_path, 'le.pkl'), 'rb') as f: 
            le = pickle.load(f)
        predicted_label = le.inverse_transform([predicted_class])[0]

        # 이모지 매핑
        emotion_to_emoji = {
            "기쁨": ":joy:",
            "상처": ":cry:",
            "슬픔": ":disappointed:",
            "분노": ":rage:",
            "당황": ":flushed:",
            "불안": ":worried:"
        }

        if predicted_label not in emotion_to_emoji:
            st.error("예측된 레이블에 해당하는 이모지가 없습니다.")
            return

        predicted_emoji = emoji.emojize(emotion_to_emoji[predicted_label])

        # 결과 출력
        st.markdown(f"### AI가 당신의 목소리에서 {predicted_label} {predicted_emoji} 감정을 인식하였습니다.")
    else:
        st.write("오디오 파일이 업로드되지 않았습니다. 오디오 파일을 업로드하거나 녹음해주세요.")
