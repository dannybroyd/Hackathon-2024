import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

# Parameters
UAV_DIR = 'UAV_sounds'
NON_UAV_DIR = 'non_UAV_sounds'
AUDIO_LENGTH = 2  # seconds
SAMPLE_RATE = 22050  # Hz
NUM_AUGMENT = 10  # Number of augmentations per UAV sample

# Function to load audio files and extract spectrogram
def load_and_preprocess(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=AUDIO_LENGTH)
    S = librosa.stft(y)
    S_db = librosa.amplitude_to_db(abs(S))
    return S_db

# Data augmentation function
def augment_data(y, sr):
    augmented_data = []
    for _ in range(NUM_AUGMENT):
        y_aug = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=np.random.uniform(-2, 2))
        S = librosa.stft(y_aug)
        S_db = librosa.amplitude_to_db(abs(S))
        augmented_data.append(S_db)
    return augmented_data

# Load and preprocess data
def load_data():
    X = []
    y = []
    # Load UAV sounds
    for filename in os.listdir(UAV_DIR):
        if filename.endswith('.wav'):
            audio_path = os.path.join(UAV_DIR, filename)
            S_db = load_and_preprocess(audio_path)
            X.append(S_db)
            y.append('UAV')
            # Augment UAV sounds
            y_raw, sr = librosa.load(audio_path, sr=SAMPLE_RATE, duration=AUDIO_LENGTH)
            augmented_spectrograms = augment_data(y_raw, sr)
            X.extend(augmented_spectrograms)
            y.extend(['UAV'] * NUM_AUGMENT)
    # Load non-UAV sounds
    for filename in os.listdir(NON_UAV_DIR):
        if filename.endswith('.wav'):
            audio_path = os.path.join(NON_UAV_DIR, filename)
            S_db = load_and_preprocess(audio_path)
            X.append(S_db)
            y.append('Non-UAV')
    return X, y

# Prepare data
X, y = load_data()

# Ensure all spectrograms have the same shape
max_shape = np.array([x.shape for x in X]).max(axis=0)
X = np.array([np.pad(x, [(0, max_shape[0] - x.shape[0]), (0, max_shape[1] - x.shape[1])], mode='constant') for x in X])

X = np.expand_dims(X, axis=-1)  # Add channel dimension
le = LabelEncoder()
y = le.fit_transform(y)
y = tf.keras.utils.to_categorical(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y_train.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}')

# Save the model
model.save('uav_sound_classifier.h5')
