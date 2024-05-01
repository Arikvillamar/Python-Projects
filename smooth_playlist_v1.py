import os
import librosa
import numpy as np
from scipy.spatial.distance import cosine

# Función para extraer características de audio
def extract_features(audio_file, format):
  if format == "mp3":
    y, sr = librosa.load(audio_file)
  elif format == "wav":
    y, sr = librosa.load(audio_file, sr=None)
  else:
    raise ValueError("Formato de audio no compatible")

  # Tempo
  tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

  # Coeficientes cromáticos
  chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
  chroma_mean = np.mean(chroma_stft)

  # Análisis de acordes
  y_harmonic, _ = librosa.effects.hpss(y)
  chords = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
  chords_mean = np.mean(chords)

  # RMS
  rms = librosa.feature.rms(y=y)
  rms_mean = np.mean(rms)

  return tempo, chroma_mean, chords_mean, rms_mean

# Función para calcular la similitud entre dos canciones
def calculate_similarity(features1, features2):
  return 1 - cosine(features1, features2)

# Ruta de la carpeta de archivos de canciones
folder_path = "/Users/ariklau/Desktop/demos"

# Canción de referencia
reference_song = "/Users/ariklau/Desktop/demos/para no pensar master.mp3"

# Calcular características de la canción de referencia
reference_features = extract_features(reference_song, format=os.path.splitext(reference_song)[1][1:])

# Umbral de similitud
umbral_de_similitud = 0.5

# Lista de canciones seleccionadas
playlist = []

# Lista de similitudes de inicio/final entre cada canción
similarity_list = []

# Diccionario de formatos de audio compatibles
supported_formats = {"mp3", "wav"}

# Iterar sobre los archivos en la carpeta
for filename in os.listdir(folder_path):
  if filename.endswith(tuple(supported_formats)):
    filepath = os.path.join(folder_path, filename)

    # Calcular características de la canción actual
    current_features = extract_features(filepath, format=os.path.splitext(filename)[1][1:])

    # Calcular la distancia coseno entre las características de la canción de referencia y la actual
    similarity = calculate_similarity(reference_features, current_features)

    # Si la similitud es suficiente, agregar la canción a la playlist
    if similarity < umbral_de_similitud:
      playlist.append((filename, similarity))
      similarity_list.append((filename, similarity))

# Ordenar la playlist basándose en la similitud
playlist.sort(key=lambda x: x[1])

# Exportar la playlist en formato .m3u
playlist_path = os.path.join(os.path.dirname(reference_song), "playlist.m3u")
with open(playlist_path, "w") as playlist_file:
  for song, _ in playlist:
    playlist_file.write(os.path.join(folder_path, song) + "\n")

print("Playlist generada y guardada en:", playlist_path)

# Mostrar las canciones en lista con el porcentaje de similitud del final/inicio entre cada una
print("\nCanciones en la lista con similitud del final/inicio:")
for i in range(len(similarity_list) - 1):
  song1 = similarity_list[i][0]
  song2 = similarity_list[i + 1][0]
  similarity = similarity_list[i + 1][1]# Similitud entre el final de song1 y el inicio de song2
  print(f"{song1} - {song2}: {similarity * 100:.2f}%")
