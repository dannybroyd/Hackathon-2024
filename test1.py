import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

names = range(1, 19)
for name in names:
    # Step 1: Load the audio file
    audio_path = f'UAV_sounds_wav/0{name}Label.wav'
    audio, sample_rate = librosa.load(audio_path, sr=None)

    # Step 2: Perform STFT
    n_fft = 2**14
    hop_length = 512
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    print(f"S_db = {S_db}")

    # Step 3: Visualize the spectrogram
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(S_db, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig(f'UAV_sounds_wav/0{name}Label.png')