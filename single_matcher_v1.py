import os
import librosa
import numpy as np
from scipy.spatial import distance

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    return tempo, np.mean(chroma), np.mean(tonnetz)

def calculate_distance(ref_features, target_features):
    ref_tempo, ref_chroma, ref_tonnetz = ref_features
    target_tempo, target_chroma, target_tonnetz = target_features

    # Convertir las características de tonalidad cromática y aspectos cromáticos a vectores 1-D
    ref_chroma_flat = ref_chroma.flatten()
    target_chroma_flat = target_chroma.flatten()
    ref_tonnetz_flat = ref_tonnetz.flatten()
    target_tonnetz_flat = target_tonnetz.flatten()

    # Calcular distancias
    tempo_distance = abs(ref_tempo - target_tempo)
    chroma_distance = distance.euclidean(ref_chroma_flat, target_chroma_flat)
    tonnetz_distance = distance.euclidean(ref_tonnetz_flat, target_tonnetz_flat)
    return tempo_distance + chroma_distance + tonnetz_distance


def find_most_similar_song(reference_song_path, folder_path):
    ref_features = extract_features(reference_song_path)
    min_distance = float('inf')
    most_similar_song = None

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mp3'):
            target_song_path = os.path.join(folder_path, file_name)
            target_features = extract_features(target_song_path)
            distance = calculate_distance(ref_features, target_features)
            if distance < min_distance:
                min_distance = distance
                most_similar_song = target_song_path

    return most_similar_song

# Example usage:
reference_song_path = "/Users/ariklau/Desktop/demos/AQUI MASTER 1.mp3"
folder_path = "/Users/ariklau/Desktop/demos"
most_similar_song = find_most_similar_song(reference_song_path, folder_path)
print("The most similar song to the reference is:", most_similar_song)
