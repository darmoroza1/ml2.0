import pandas as pd
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


def load_data(csv_path):
    return pd.read_csv(csv_path)


def read_wav_files(data, directory):
    audio_data = []
    for index, row in data.iterrows():
        filepath = f"{directory}/{row['Chord']}.wav"
        sample_rate, audio = wavfile.read(filepath)
        audio_data.append(audio)
    return np.array(audio_data)


def prepare_data(X, y, n_components=50):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Нормалізація даних
    X_scaled = []
    for x in X:
        scaler = StandardScaler()
        X_scaled.append(scaler.fit_transform(x.reshape(-1, 1)).flatten())
    X_scaled = np.array(X_scaled)

    # Зменшення розмірності за допомогою PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

    # Додаткові перевірки
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Unique labels: {np.unique(y_encoded)}")

    return X_train, X_test, y_train, y_test, label_encoder
