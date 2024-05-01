import os
import librosa
import numpy as np
from scipy.spatial.distance import cdist

def extract_features(audio_file, duration=20):
    y, sr = librosa.load(audio_file, duration=duration)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    return np.mean(tonnetz, axis=1)

def find_most_similar_songs(query_features, feature_list, num_songs):
    distances = cdist(query_features.reshape(1, -1), feature_list, metric='euclidean')
    most_similar_indices = np.argsort(distances)[0][:num_songs]
    return most_similar_indices

def main():
    # Carpeta donde se encuentran los archivos de audio
    audio_folder = "/Users/ariklau/Desktop/demos"

    # Solicitar al usuario el nombre del archivo de audio para analizar
    audio_file_name = input("Ingrese el nombre del archivo de audio para analizar (sin la extensión .mp3 o .wav): ")

    # Verificar si el archivo especificado existe en la carpeta
    audio_file = os.path.join(audio_folder, f"{audio_file_name}.mp3")
    if not os.path.isfile(audio_file):
        audio_file = os.path.join(audio_folder, f"{audio_file_name}.wav")
        if not os.path.isfile(audio_file):
            print("El archivo especificado no existe en la carpeta /Users/ariklau/Desktop/demos.")
            return

    # Extraer características del archivo de audio
    query_features = extract_features(audio_file)

    # Listar archivos de referencia en la carpeta
    reference_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(('.mp3', '.wav'))]

    # Extraer características de todas las canciones de referencia
    features_list = []
    for ref_audio_file in reference_files:
        features_list.append(extract_features(ref_audio_file))

    # Número de canciones similares a encontrar
    num_songs = 12

    # Buscar canciones similares
    similar_song_indices = find_most_similar_songs(query_features, features_list, num_songs)

    print("Canciones similares encontradas:")
    for idx in similar_song_indices:
        print(reference_files[idx])

if __name__ == "__main__":
    main()
