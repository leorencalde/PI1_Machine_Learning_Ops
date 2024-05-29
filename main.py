# Importar librerias
from fastapi import FastAPI, HTTPException
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Crear la aplicación FastAPI
app = FastAPI()

# Especificar las rutas absolutas a los archivos Parquet usando raw strings para evitar problemas con las barras invertidas
steam_games_path = 'Dataset/steam_games_transf.parque'
steam_games_recommend_path = 'Dataset/steam_games_recommend.parquet'

# Verificar que los archivos existen
if not os.path.exists(steam_games_path):
    raise FileNotFoundError(f"Archivo no encontrado: {steam_games_path}")
if not os.path.exists(steam_games_recommend_path):
    raise FileNotFoundError(f"Archivo no encontrado: {steam_games_recommend_path}")

# Cargar los datos desde los archivos Parquet
steam_games = pd.read_parquet(steam_games_path)
steam_games_recommend = pd.read_parquet(steam_games_recommend_path)

# Ruta raíz que devuelve un mensaje de bienvenida
@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de análisis de juegos de Steam"}

# Endpoint 1: devuelve información sobre el contenido desarrollado por una empresa en específico
@app.get("/developer/{desarrollador}")
def developer(desarrollador: str):

    # Filtra los datos para incluir solo los juegos desarrollados por el desarrollador especificado
    df = steam_games[steam_games['developer'] == desarrollador].copy()

    # Verifica si no se encontraron juegos para el desarrollador especificado
    if df.empty:
        return {"error": "Desarrollador no encontrado"}
    
    # Convertir la columna release_date a tipo datetime y manejar valores no convertibles
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    
    # Eliminar filas donde release_date no pudo ser convertido a datetime
    df = df.dropna(subset=['release_date'])

    # Extraer el año de la columna release_date
    df['release_year'] = df['release_date'].dt.year
    
    # Agrupar los datos por año de lanzamiento y realizar las agregaciones especificadas
    result = df.groupby('release_year').agg(
        cantidad_items=('id', 'count'), # Cuenta el número de juegos
        contenido_free=('price', lambda x: (x == 0).sum() / len(x) * 100) # Calcula el porcentaje de juegos gratuitos
    ).reset_index()
    
    # Convierte el porcentaje de juegos gratuitos a cadena de texto y añade el símbolo '%'
    result['contenido_free'] = result['contenido_free'].astype(str) + '%'
    
    return result.to_dict(orient='records')

# Sistema de recomendacion item-item: Se recomiendan 5 juegos similares a un juego dado por su ID utilizando la similitud del coseno
@app.get("/recomendacion_juego/{producto_id}")
def recomendacion_juego(producto_id: int, sample_size=1000):

    try:
        # Convertir sample_size a entero
        sample_size = int(sample_size)

         # Filtrar el juego por el ID proporcionado
        juego = steam_games_recommend[steam_games_recommend['id'] == producto_id]
        if juego.empty:
            raise HTTPException(status_code=404, detail="Juego no encontrado")
    
        # Crear una muestra más pequeña del conjunto de datos para evitar problemas de memoria
        steam_games_sample = steam_games_recommend.sample(min(sample_size, len(steam_games_recommend)), random_state=1)

        # Agregar el juego dado a la muestra para asegurar que esté presente
        steam_games_sample = pd.concat([steam_games_sample, juego], ignore_index=True)

        # Vectorizar los géneros
        vectorizer = CountVectorizer()
        genre_matrix = vectorizer.fit_transform(steam_games_sample['genres'])

        # Calcular la similitud del coseno
        cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

        # Obtener el índice del juego dado
        idx = steam_games_sample.index[steam_games_sample['id'] == producto_id].tolist()
        if not idx:
            raise HTTPException(status_code=404, detail="Juego no encontrado en la matriz de similitud")
    
        idx = idx[0]

        # Obtener las puntuaciones de similitud de coseno para el juego dado con todos los demás juegos
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Ordenar los juegos en función de las puntuaciones de similitud
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Seleccionar los 5 juegos más similares (excluyendo el propio juego)
        sim_scores = sim_scores[1:6]

        # Obtener los índices de los juegos más similares
        juego_indices = [i[0] for i in sim_scores]

        # Devolver los juegos más similares
        juegos_recomendados = steam_games_sample.iloc[juego_indices]
        return juegos_recomendados[['id', 'app_name', 'genres', 'developer', 'price']].to_dict(orient='records')

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
