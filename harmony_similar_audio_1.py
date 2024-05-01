import os
import librosa
import numpy as np
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor

def extract_features(audio_file, duration=30):
    y, sr = librosa.load(audio_file, duration=duration)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_feature = np.array([tempo])
    return np.concatenate((np.mean(tonnetz, axis=1), tempo_feature))

def find_most_similar_songs(query_features, feature_lists, num_songs):
    # Calcular el promedio de las características de las canciones de consulta
    query_avg_features = np.mean(query_features, axis=0)
    # Calcular la distancia entre el promedio de las características de las canciones de consulta y las características de las canciones de referencia
    distances = cdist(query_avg_features.reshape(1, -1), np.array(feature_lists), metric='euclidean')
    # Encontrar las canciones más similares al promedio de las características de las canciones de consulta
    most_similar_indices = np.argsort(distances)[0][:num_songs]
    return most_similar_indices

def create_playlist(playlist_name, similar_songs):
    with open(playlist_name + ".m3u", "w") as f:
        for song in similar_songs:
            f.write(song + "\n")

def main():
    # Carpeta donde se encuentran los archivos de audio
    audio_folder = "/Users/ariklau/Desktop/WAV2ANALIZE/ARIKLAU"

    # Solicitar al usuario los nombres de los archivos de audio para analizar
    query_audio_file_names = []
    for i in range(2):
        query_audio_file_name = input(f"Ingrese el nombre del archivo de audio {i+1} para analizar (sin la extensión .mp3 o .wav): ")
        query_audio_file_names.append(query_audio_file_name)

    # Verificar si los archivos especificados existen en la carpeta
    query_audio_files = []
    for query_audio_file_name in query_audio_file_names:
        query_audio_file = os.path.join(audio_folder, f"{query_audio_file_name}.mp3")
        if not os.path.isfile(query_audio_file):
            query_audio_file = os.path.join(audio_folder, f"{query_audio_file_name}.wav")
            if not os.path.isfile(query_audio_file):
                print(f"El archivo {query_audio_file_name} no existe en la carpeta /Users/ariklau/Desktop/demos.")
                return
        query_audio_files.append(query_audio_file)

    # Extraer características de los archivos de audio de consulta
    query_features = [extract_features(query_audio_file) for query_audio_file in query_audio_files]

    # Listar archivos de referencia en la carpeta
    reference_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if f.endswith(('.mp3', '.wav'))]

    # Extraer características de todas las canciones de referencia
    feature_lists = [extract_features(ref_audio_file) for ref_audio_file in reference_files]

    # Buscar las 12 canciones más similares al promedio de las características de las canciones de consulta
    num_songs = 6
    most_similar_indices = find_most_similar_songs(query_features, feature_lists, num_songs)

    # Mostrar las canciones similares encontradas
    print("Canciones similares encontradas:")
    similar_songs = []
    for idx in most_similar_indices:
        similar_song = reference_files[idx]
        similar_songs.append(similar_song)
        print(similar_song)

    # Crear la lista de reproducción con el nombre del archivo de audio de consulta
    playlist_name = "_".join(query_audio_file_names) + "_playlist"
    create_playlist(playlist_name, similar_songs)
    print(f"Lista de reproducción '{playlist_name}.m3u' creada con las canciones similares al promedio de las canciones de consulta.")

def send_notification(message):
    applescript = f'''
    display notification "{message}" with title "Resultados listos"
    '''
    os.system(f"osascript -e '{applescript}'")

# Después de obtener los resultados
send_notification("Los resultados están listos. ¡Echa un vistazo!")

if __name__ == "__main__":
    main()
