# Machine Learning Ops

## Descripción

Este proyecto es una API para el análisis y recomendación de juegos de Steam utilizando FastAPI. La API proporciona múltiples funcionalidades, incluyendo información sobre juegos desarrollados por una empresa específica, datos de usuario, recomendaciones basadas en géneros y análisis de sentimiento de reseñas.

## Funcionalidades

1. **Información sobre juegos desarrollados por una empresa específica:**
   - Endpoint: `/developer/{desarrollador}`
   - Devuelve información sobre el contenido desarrollado por una empresa especificada.

2. **Información sobre el gasto y actividad de un usuario específico:**
   - Endpoint: `/userdata/{user_id}`
   - Devuelve información sobre el gasto y la actividad de un usuario de Steam.

3. **Usuario con más horas jugadas para un género específico:**
   - Endpoint: `/user_for_genre/{genero}`
   - Devuelve el usuario con más horas jugadas para un género dado y la acumulación de horas jugadas por año de lanzamiento.

4. **Top 3 desarrolladores con juegos más recomendados por año:**
   - Endpoint: `/best_developer_year/{year}`
   - Devuelve el top 3 de desarrolladores con juegos más recomendados para un año especificado.

5. **Análisis de sentimiento de las reseñas para un desarrollador especificado:**
   - Endpoint: `/developer_reviews_analysis/{desarrollador}`
   - Devuelve un análisis de sentimiento de las reseñas para un desarrollador especificado.

6. **Recomendación de juegos similares utilizando la similitud del coseno:**
   - Endpoint: `/recomendacion_juego/{producto_id}`
   - Devuelve una lista de 5 juegos recomendados similares al ingresado basado en la similitud del coseno de los géneros.

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/leorencalde/PI1_Machine_Learning_Ops.git
   cd PI1_Machine_Learning_Ops

2. Crea un entorno virtual:
   ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate

3. Instala las dependencias:
   ```bash
    pip install -r requirements.txt

## Uso

1. Corre la aplicación:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

2. Abre tu navegador y navega a http://127.0.0.1:8000 para ver el mensaje de bienvenida.

3. Para ver la documentación interactiva de la API, navega a http://127.0.0.1:8000/docs.

## Estructura del Proyecto

**main.py:** `Código principal de la aplicación FastAPI.` 

**requirements.txt:** `Lista de dependencias necesarias para el proyecto.`

**Dataset/:** `Carpeta que contiene los archivos Parquet utilizados para la carga de datos.`

## Contribuciones 

Las contribuciones son bienvenidas. Por favor, sigue los pasos a continuación para contribuir:

1. Haz un fork del repositorio.
   
2. Crea una nueva rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`).

3. Realiza tus cambios y haz commit (`git commit -am 'Añadir nueva funcionalidad'`).

4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`).

5. Abre un Pull Request.

## Contacto 

**LEONARDO RENTERIA**
   - Email: `leo921120@hotmail.com`
   - Telefono: `+573138228947`
   - Ubicación: `Bogotá DC, Colombia`
