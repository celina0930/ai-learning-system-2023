import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import warnings
import matplotlib

matplotlib.rc('font', family='NanumBarunGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib import font_manager, rc
font_name= font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)


def data_analysis():
    st.header("Data Analysis")
    st.subheader("Upload Audio File (.wav)")
    audio_file = st.file_uploader("Upload Audio File", type=["wav"])
    if audio_file is not None:
        st.audio(audio_file, format='audio/wav')
        audio_data, sr = librosa.load(audio_file, sr=None)
        st.write("Sampling Rate:", sr, "Hz")
        st.write("Audio Length:", len(audio_data) / sr, "seconds")
        plot_waveform(audio_data, sr)
        plot_spectrogram(audio_data, sr)
        plot_chromagram(audio_data, sr)
        plot_mel_spectrogram(audio_data, sr)
        plot_mfcc(audio_data, sr)

    st.subheader("Upload Train Labels (.csv)")
    csv_file = st.file_uploader("Upload Train Labels", type=["csv"])
    if csv_file is not None:
        train_labels = pd.read_csv(csv_file)
        st.write(train_labels.head())

        st.subheader("Class Distribution")
        plot_combined_class_distribution_with_counts(train_labels)

def plot_waveform(audio_data, sr):
    st.subheader("Waveform")
    st.write("웨이브폼은 오디오의 진폭을 시간에 따라 표시한 것입니다. 이 표현은 오디오의 전체적인 레벨과 모양을 확인하는 데 도움이 됩니다.")
    st.write("#### Waveform 해석방법")
    st.write("웨이브폼을 통해 소리의 크기와 지속 시간, 그리고 소리의 모양을 살펴볼 수 있습니다. 웨이브폼에서 높은 진폭의 부분은 소리가 큰 부분을 나타내며, 낮은 진폭의 부분은 소리가 작은 부분을 나타냅니다.")
    plt.figure(figsize=(10, 4))
    plt.plot(audio_data)
    st.pyplot()

def plot_spectrogram(audio_data, sr):
    st.subheader("Spectrogram")
    st.write("스펙트로그램은 오디오의 시간-주파수 표현으로, 주파수의 강도를 시간에 따라 표시합니다. 이 표현은 주파수 성분의 변화를 관찰하고, 음의 특성을 파악하는 데 도움이 됩니다.")
    st.write("#### Spectrogram 해석 방법")
    st.write("스펙트로그램을 통해 소리의 다양한 주파수 성분과 그 변화를 관찰할 수 있습니다. 밝은 색상의 영역은 높은 에너지를 가진 주파수 성분을 나타내며, 어두운 색상의 영역은 낮은 에너지를 가진 주파수 성분을 나타냅니다.")
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    plt.imshow(D, aspect='auto', cmap='inferno', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    st.pyplot()

def plot_chromagram(audio_data, sr):
    st.subheader("Chromagram")
    st.write("크로마그램은 음악의 피치 클래스(12개의 반음)에 대한 에너지 분포를 시간에 따라 표시합니다. 이 표현은 화성 정보와 같은 음악의 구조적 특성을 파악하는 데 도움이 됩니다.")
    st.write("#### Chromagram 해석방법")
    st.write("크로마그램을 통해 시간에 따른 음악의 피치 클래스별 에너지 분포를 관찰할 수 있습니다. 밝은 색상의 영역은 각 피치 클래스에 대해 높은 에너지를 나타내며, 어두운 색상의 영역은 낮은 에너지를 나타냅니다.")
    chroma = librosa.feature.chroma_cqt(y=audio_data, sr=sr)
    plt.figure(figsize=(10, 4))
    plt.imshow(chroma, aspect='auto', cmap='inferno', origin='lower')
    plt.colorbar()
    plt.title("Chromagram")
    plt.tight_layout()
    st.pyplot()

def plot_mel_spectrogram(audio_data, sr):
    st.subheader("Mel Spectrogram")
    st.write("멜 스펙트로그램은 주파수 축을 멜 스케일로 변환한  스펙트로그램입니다. 멜 스케일은 인간 청각이 주파수의 차이를 인식하는 방식에 기반하여 설계되어 있습니다. 이 표현은 음성 및 오디오 인식 작업에 주로 사용됩니다.")
    st.write("#### Mel Spectrogram 해석방법")
    st.write("멜 스펙트로그램을 통해 소리의 멜 주파수 성분과 그 변화를 관찰할 수 있습니다. 밝은 색상의 영역은 높은 에너지를 가진 멜 주파수 성분을 나타내며, 어두운 색상의 영역은 낮은 에너지를 가진 멜 주파수 성분을 나타냅니다.")
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    plt.figure(figsize=(10, 4))
    plt.imshow(mel_spec_db, aspect='auto', cmap='inferno', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    st.pyplot()

def plot_mfcc(audio_data, sr):
    st.subheader("MFCC(Mel-Frequency Cepstral Coefficients)")
    st.write("MFCC는 오디오 신호의 스펙트럼에서 추출한 특성으로, 음성 인식 및 음악 장르 분류와 같은 작업에 널리 사용됩니다. MFCC는 오디오 신호의 전체적인 특성을 요약하여 나타냅니다.")
    st.write("#### MFCC 해석방법")
    st.write("MFCC 플롯을 통해 시간에 따른 오디오 신호의 주요 특성을 관찰할 수 있습니다. 이 플롯에서 색상의 변화와 패턴은 오디오 신호의 텍스처와 구조를 나타냅니다. 이러한 정보는 음성 인식, 음악 장르 분류 등의 작업에서 중요한 역할을 합니다.")
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    plt.imshow(mfccs, aspect='auto', cmap='inferno', origin='lower')
    plt.colorbar()
    plt.title("MFCC")
    plt.tight_layout()
    st.pyplot()

def plot_combined_class_distribution_with_counts(df):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(data=df, x='class', palette='Set2')
    
    # 막대 위에 개수 표시
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')
    
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    st.pyplot()