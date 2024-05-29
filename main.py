# Importar librerias
from fastapi import FastAPI
import pandas as pd
import os

# Crear la aplicación FastAPI
app = FastAPI()

# Especificar las rutas absolutas a los archivos Parquet usando raw strings para evitar problemas con las barras invertidas
steam_games_path = 'Dataset/steam_games_transf.parquet'
user_items_path = 'Dataset/users_items_transf.parquet'
user_reviews_path = 'Dataset/user_reviews_sentiment_analysis.parquet'

# Verificar que los archivos existen
if not os.path.exists(steam_games_path):
    raise FileNotFoundError(f"Archivo no encontrado: {steam_games_path}")
if not os.path.exists(user_items_path):
    raise FileNotFoundError(f"Archivo no encontrado: {user_items_path}")
if not os.path.exists(user_reviews_path):
    raise FileNotFoundError(f"Archivo no encontrado: {user_reviews_path}")

# Cargar los datos desde los archivos Parquet
steam_games = pd.read_parquet(steam_games_path)
user_items = pd.read_parquet(user_items_path)
user_reviews = pd.read_parquet(user_reviews_path)

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

# Endpoint 2: devuelve información sobre el gasto y actividad de un usuario en específico
@app.get("/userdata/{user_id}")
def userdata(user_id: str):

    # Filtrar los ítems pertenecientes al usuario especificado
    df = user_items[user_items['steam_id'] == user_id]
    if df.empty:
        return {"error": "Usuario no encontrado"}
    
    # Calcular el total de dinero gastado por el usuario (aquí se usa el tiempo de juego como proxy)
    dinero_gastado = df['item_playtime_forever'].sum()  # No hay precio en user_items, se usará 'item_playtime_forever'

    # Contar la cantidad de ítems que tiene el usuario
    cantidad_items = df['item_item_id'].count()
    
    # Calcular el porcentaje de recomendaciones positivas del usuario
    recomendacion = user_reviews[user_reviews['user_id'] == user_id]['review_recommend'].mean() * 100
    
    return {
        "Usuario": user_id,
        "Dinero gastado": f"{dinero_gastado} USD",
        "% de recomendación": f"{recomendacion:.2f}%",
        "cantidad de items": cantidad_items
    }

# Endpoint 3: devuelve el usuario con más horas jugadas para un género dado y la acumulación de horas jugadas por año de lanzamiento
@app.get("/user_for_genre/{genero}")
def user_for_genre(genero: str):

    # Obtener los IDs de los juegos que pertenecen al género especificado
    game_ids = steam_games[steam_games['genres'].apply(lambda x: genero in x if isinstance(x, list) else False)]['id']

    # Filtrar los ítems jugados que corresponden a esos IDs de juegos
    df = user_items[user_items['item_item_id'].isin(game_ids)]
    if df.empty:
        return {"error": "Género no encontrado"}
    
    # Sumar las horas jugadas por cada usuario
    user_hours = df.groupby('steam_id')['item_playtime_forever'].sum()
    
    # Encontrar el usuario con más horas jugadas
    top_user = user_hours.idxmax()

    # Obtener los datos del usuario con más horas jugadas, agrupando por ítem
    top_user_data = df[df['steam_id'] == top_user].groupby('item_item_id')['item_playtime_forever'].sum().reset_index()
    
    result = {
        "Usuario con más horas jugadas para Género": top_user,
        "Horas jugadas": top_user_data.to_dict(orient='records')
    }
    return result

# Endpoint 4: devuelve el top 3 de desarrolladores con juegos más recomendados por año especificado
@app.get("/best_developer_year/{year}")
def best_developer_year(year: int):

    # Convertir la columna review_posted a tipo datetime
    user_reviews['review_posted'] = pd.to_datetime(user_reviews['review_posted'], errors='coerce')

    # Filtrar las reseñas que corresponden al año especificado
    df = user_reviews[user_reviews['review_posted'].dt.year == year]
    
    if df.empty:
        return {"error": "No hay datos para el año especificado"}
    
    # Contar las recomendaciones por cada ítem y obtener los top 3 más recomendados
    developer_recommendations = df[df['review_recommend']]['review_item_id'].value_counts().head(3)
    result = [{"Puesto 1": developer_recommendations.index[0]},
              {"Puesto 2": developer_recommendations.index[1]},
              {"Puesto 3": developer_recommendations.index[2]}]
    return result

# Endpoint 5: devuelve un análisis de sentimiento de las reseñas para un desarrollador especificado
@app.get("/developer_reviews_analysis/{desarrollador}")
def developer_reviews_analysis(desarrollador: str):

    # Obtener los IDs de los juegos desarrollados por la empresa especificada
    game_ids = steam_games[steam_games['developer'] == desarrollador]['id']

    # Filtrar las reseñas que corresponden a esos IDs de juegos
    df = user_reviews[user_reviews['review_item_id'].isin(game_ids)]
    
    if df.empty:
        return {"error": "Desarrollador no encontrado"}
    
    # Contar las reseñas positivas y negativas
    sentiment_analysis = df['sentiment_analysis'].value_counts()
    result = {desarrollador: {
        "Negative": sentiment_analysis.get(0, 0),
        "Positive": sentiment_analysis.get(2, 0)
    }}
    return result
     
# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
