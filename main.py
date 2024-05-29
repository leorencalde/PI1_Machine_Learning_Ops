# Importar librerias
from fastapi import FastAPI
import pandas as pd
import os

# Crear la aplicación FastAPI
app = FastAPI()

# Especificar las rutas absolutas a los archivos Parquet usando raw strings para evitar problemas con las barras invertidas
steam_games_path = 'Dataset/steam_games_transf.parquet'

# Verificar que los archivos existen
if not os.path.exists(steam_games_path):
    raise FileNotFoundError(f"Archivo no encontrado: {steam_games_path}")

# Cargar los datos desde los archivos Parquet
steam_games = pd.read_parquet(steam_games_path)

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
    
# Ejecutar la aplicación con Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
