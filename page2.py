import streamlit as st
import pandas as pd
import os
from io import StringIO
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from tensorflow.keras.utils import plot_model
import torch
import torchaudio
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM, SimpleRNN, Bidirectional
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# wav2vec 2.0 모델 및 프로세서 불러오기
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=16000)

def pad_sequences(sequences, max_len=None, padding_value=0.0):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    
    padded_sequences = np.full((len(sequences), max_len, sequences[0].shape[1]), padding_value)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    
    return padded_sequences

def extract_features(file_paths):
    features = []
    
    # Add a placeholder for the estimated time
    latest_iteration = st.empty()
    progress_bar = st.progress(0)
    
    start_time = time.time()
    
    for i, file_path in enumerate(file_paths):
        # Load the audio file and resample
        speech, _ = torchaudio.load(file_path)
        speech = resampler(speech)
        
        # If there are two channels, average them to get a single-channel signal
        if speech.shape[0] == 2:
            speech = speech.mean(dim=0, keepdim=True)
        
        speech = speech.squeeze()

        # Process the audio data with the Wav2Vec2.0 processor and model
        inputs = processor(speech, return_tensors="pt", padding=True, sampling_rate=16000)
        with torch.no_grad():
            outputs = wav_model(**inputs)

        # Extract the features from the model output
        features.append(outputs.logits.squeeze().cpu().numpy())

        # Update the progress bar
        progress_percent = (i + 1) / len(file_paths)
        progress_bar.progress(progress_percent)

        # Calculate elapsed time and estimated time remaining
        elapsed_time = time.time() - start_time
        estimated_time_remaining = (elapsed_time / (i + 1)) * (len(file_paths) - (i + 1))
        latest_iteration.text(f"Progress: {progress_percent*100:.2f}% Estimated time remaining: {estimated_time_remaining:.2f} seconds")

    return pad_sequences(features)

def create_model(model_type, input_shape, num_classes, dropout_rate=0.5, hidden_units=128, optimizer='adam'):
    model = Sequential()
    
    if model_type == 'gru':
        model.add(GRU(hidden_units, input_shape=input_shape, return_sequences=True))
        model.add(GRU(hidden_units))
    elif model_type == 'lstm':
        model.add(LSTM(hidden_units, input_shape=input_shape, return_sequences=True))
        model.add(LSTM(hidden_units))
    elif model_type == 'rnn':
        model.add(SimpleRNN(hidden_units, input_shape=input_shape, return_sequences=True))
        model.add(SimpleRNN(hidden_units))
    elif model_type == 'bi-lstm':
        model.add(Bidirectional(LSTM(hidden_units, return_sequences=True), input_shape=input_shape))
        model.add(Bidirectional(LSTM(hidden_units)))
    
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

def visualize_training(history):
    # Plot the training and validation loss
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot the training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    st.pyplot()

class StreamlitCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Print the logs to streamlit
        st.text(f"Epoch: {epoch + 1}")
        st.text(f"Loss: {logs['loss']:.4f}")
        st.text(f"Accuracy: {logs['accuracy']:.4f}")
        st.text(f"Validation Loss: {logs['val_loss']:.4f}")
        st.text(f"Validation Accuracy: {logs['val_accuracy']:.4f}")

def page2():
    st.title("Data Preparation")
    st.write("""
    먼저, AI 학습에 사용할 데이터를 준비합니다. 다음과 같은 단계를 따르세요:

    1. wav 파일이 저장된 폴더의 경로를 입력하세요.
    2. 'train_labels.csv', 'val_labels.csv', 'test_labels.csv' 파일을 업로드하세요.
        - 이 CSV 파일들은 각각 학습, 검증, 테스트에 사용할 데이터의 레이블을 포함해야 합니다.
        - 이 파일들은 'file_name' 컬럼을 포함해야 하며, 이 컬럼은 wav 파일의 이름과 일치해야 합니다.
        - 또한, 'class' 컬럼을 포함해야 하며, 이 컬럼은 해당 오디오 데이터의 클래스 레이블을 포함해야 합니다.
    3. 모든 파일이 업로드되면, 데이터가 자동으로 로드되며 클래스 레이블은 필요한 경우 라벨 인코딩이 적용됩니다.
    """)

    # 폴더 경로 입력
    st.write("Enter the path to the folder containing the wav files:")
    wav_folder = st.text_input("Wav folder path", "")

    if os.path.isdir(wav_folder):
        st.success("Folder path is valid.")
    else:
        st.error("Folder path is not valid. Please check your input.")

    st.write("Enter the path for download files:")
    download_folder = st.text_input("Download folder path", "")

    if os.path.isdir(download_folder):
        st.success("Folder path is valid.")
    else:
        st.error("Folder path is not valid. Please check your input.")
        
    # label 파일 업로드
    st.write("Upload the CSV files:")
    train_labels_file = st.file_uploader("Upload train_labels.csv", type=["csv"])
    val_labels_file = st.file_uploader("Upload val_labels.csv", type=["csv"])
    test_labels_file = st.file_uploader("Upload test_labels.csv", type=["csv"])

     # 파일 불러오기
    if wav_folder and train_labels_file and val_labels_file and test_labels_file:
        train_labels = pd.read_csv(train_labels_file)
        val_labels = pd.read_csv(val_labels_file)
        test_labels = pd.read_csv(test_labels_file)

        # Train 데이터 및 클래스 불러오기
        train_data = []
        for _, row in train_labels.iterrows():
            file_path = os.path.join(wav_folder, f"{row['file_name']}.wav").replace('\\', '/')
            train_data.append(file_path)
        val_data = []
        for _, row in val_labels.iterrows():
            file_path = os.path.join(wav_folder, f"{row['file_name']}.wav").replace('\\', '/')
            val_data.append(file_path)
        test_data = []
        for _, row in test_labels.iterrows():
            file_path = os.path.join(wav_folder, f"{row['file_name']}.wav").replace('\\', '/')
            test_data.append(file_path)

        train_classes = train_labels["class"]
        val_classes = val_labels["class"]
        test_classes = test_labels["class"]

        # 클래스 라벨 인코딩
        if train_classes.dtype == np.dtype("object"):
            le = LabelEncoder()
            train_classes_encoded = le.fit_transform(train_classes)
            val_classes_encoded = le.transform(val_classes)
            test_classes_encoded = le.transform(test_classes)

            with open(os.path.join(download_folder, 'le.pkl'), 'wb') as f:
                pickle.dump(le, f)

            st.write("Encoded class labels:")
            st.write(pd.DataFrame({"Train_Original": train_classes, "Train_Encoded": train_classes_encoded}))
            st.write(pd.DataFrame({"Val_Original": val_classes, "Val_Encoded": val_classes_encoded}))
            st.write(pd.DataFrame({"Test_Original": test_classes, "Test_Encoded": test_classes_encoded}))
        else:
            train_classes_encoded = train_classes
            val_classes_encoded = val_classes
            test_classes_encoded = test_classes

        st.write("Sample train data:")
        st.write(train_data[:5])

    # wav2vec 2.0 설명
    st.write("""
    **wav2vec 2.0**은 페이스북 AI 연구팀에 의해 개발된 오디오 데이터의 특징을 추출하는 딥러닝 모델입니다. 
    이 모델은 비지도 학습 방식으로 훈련되며, 기존의 오디오 특징 추출 방법보다 더 높은 성능을 보입니다. 
    wav2vec 2.0은 음성 인식, 감정 분석 등 다양한 오디오 관련 태스크에서 사용될 수 있습니다.
    """)

    # wav2vec 2.0 특징 추출 버튼
    if st.button('Extract features with wav2vec 2.0'):
        train_features = extract_features(train_data)
        val_features = extract_features(val_data)
        test_features = extract_features(test_data)

        min_timesteps = min(train_features.shape[1], val_features.shape[1], test_features.shape[1])
        with open(os.path.join(download_folder, 'min_timesteps.pkl'), 'wb') as f:
            pickle.dump(min_timesteps, f)

        train_features = train_features[:, :min_timesteps, :]
        val_features = val_features[:, :min_timesteps, :]
        test_features = test_features[:, :min_timesteps, :]
        
        # Save the features as numpy files
        np.save(os.path.join(download_folder, 'train_features.npy'), train_features)
        np.save(os.path.join(download_folder, 'val_features.npy'), val_features)
        np.save(os.path.join(download_folder, 'test_features.npy'), test_features)

        st.write(train_features.shape, val_features.shape, test_features.shape)
        st.success("Features have been successfully extracted and saved as numpy files in the provided directory.")

    if os.path.isfile(os.path.join(download_folder, 'train_features.npy')) and \
        os.path.isfile(os.path.join(download_folder, 'val_features.npy')) and \
        os.path.isfile(os.path.join(download_folder, 'test_features.npy')):

        train_features = np.load(os.path.join(download_folder, 'train_features.npy'))
        val_features = np.load(os.path.join(download_folder, 'val_features.npy'))
        test_features = np.load(os.path.join(download_folder, 'test_features.npy'))

    model = None
    # Model selection
    model_options = ['gru', 'lstm', 'rnn', 'bi-lstm']
    selected_model = st.selectbox("Select the model to use for training:", model_options)

    # Hyperparameters selection
    dropout_rate_options = [i * 0.1 for i in range(11)]
    hidden_units_options = [2**i for i in range(1, 11)]

    st.subheader("Hyperparameters")
    epochs = st.number_input("Enter the number of epochs:", min_value=1, value=10)
    dropout_rate = st.select_slider('Select the dropout rate:', options=dropout_rate_options, value=0.5, format_func=lambda x: f'{x:.1f}')
    hidden_units = st.select_slider('Select the number of hidden units:', options=hidden_units_options, value=128)
    optimizer_options = ['adam', 'sgd', 'rmsprop']
    optimizer = st.selectbox("Select the optimizer:", optimizer_options)

    # Create the model and visualize its architecture
    if st.button('Create model'):
        model = create_model(selected_model, (train_features.shape[1], train_features.shape[2]), len(np.unique(train_classes_encoded)), dropout_rate, hidden_units, optimizer)
        st.session_state['model'] = model

        # Textual summary
        str_io = StringIO()
        sys.stdout = str_io
        model.summary()
        sys.stdout = sys.__stdout__
        
        summary_str = str_io.getvalue()
        st.text(summary_str)
        
        # Visual summary
        plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True, dpi=96)
        st.image('model.png')

    # Training the model
    if st.button('Train model'):
        if 'model' in st.session_state:
            model = st.session_state['model']
            callback = StreamlitCallback()
            history = model.fit(train_features, train_classes_encoded, epochs=epochs, validation_data=(val_features, val_classes_encoded), callbacks=[callback])
            st.session_state['history'] = history 

            st.markdown('<span style="color:green; font-weight:bold; font-size:20px">Training completed.</span>', unsafe_allow_html=True)
        else:
            st.error('Please create the model before training.')
    if 'history' in st.session_state:  
        visualize_training(st.session_state['history'])

    # Evaluating the model
    if st.button('Evaluate model'):
        if 'model' in st.session_state:
            model = st.session_state['model']

            # Predict the test set results
            y_pred = model.predict(test_features)
            y_pred_classes = np.argmax(y_pred, axis = 1)

            # Inverse transform the encoded labels
            y_pred_classes_labels = le.inverse_transform(y_pred_classes)
            test_classes_labels = le.inverse_transform(test_classes_encoded)

           # Create and show the confusion matrix
            cm = confusion_matrix(test_classes_labels, y_pred_classes_labels)
            st.write('Confusion Matrix:')
            st.write(cm)
            cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

            # Visualize confusion matrix
            plt.figure(figsize=(10,7))
            sn.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Truth')
            st.pyplot(plt)

            # Show the classification report
            cr = classification_report(test_classes_labels, y_pred_classes_labels, output_dict=True)
            st.write('Classification Report:')
            st.write(pd.DataFrame(cr).transpose())
        else:
            st.error('Please train the model before evaluation.')


    # Save the model
    if st.button('Save model'):
        if 'model' in st.session_state:
            model = st.session_state['model']
            model_path = os.path.join(download_folder, 'saved_model')
            model.save(model_path)
            st.write(f'Model saved at {model_path}')
        else:
            st.error('Please create the model before saving.')