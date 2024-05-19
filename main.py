from preprocess import load_data, read_wav_files, prepare_data
from hmm_model import train_hmm, evaluate_hmm
from neural_network import train_nn, evaluate_nn
from fourier_analysis import fourier_transform, evaluate_fourier_hmm
import numpy as np

# Завантаження даних
data = load_data('data/triads.csv')
audio_data = read_wav_files(data, 'data/')

# Підготовка даних з PCA
X_train, X_test, y_train, y_test, label_encoder = prepare_data(audio_data, data['Chord'], n_components=50)

# Визначення кількості класів
num_classes = len(label_encoder.classes_)

# Марківська модель
hmm_model = train_hmm(X_train, y_train, n_components=4)
hmm_accuracy = evaluate_hmm(hmm_model, X_test, y_test)

# Нейронна мережа
nn_model = train_nn(X_train, y_train, num_classes)
nn_accuracy = evaluate_nn(nn_model, X_test, y_test, num_classes)

# Аналіз Фур'є з Марківською моделлю
X_train_fft = np.array([fourier_transform(x) for x in X_train])
X_test_fft = np.array([fourier_transform(x) for x in X_test])
fourier_hmm_model = train_hmm(X_train_fft, y_train, n_components=4)
fourier_hmm_accuracy = evaluate_fourier_hmm(fourier_hmm_model, X_test_fft, y_test)

# Порівняння результатів
print(f"HMM Accuracy: {hmm_accuracy}")
print(f"Neural Network Accuracy: {nn_accuracy}")
print(f"Fourier HMM Accuracy: {fourier_hmm_accuracy}")
