# Autor: Romina Escudero - 
Contacto: https://www.linkedin.com/in/romina-escudero/

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>
# <p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center>**`Machine Learning Operations (MLOps) - Steam Games`**</h1>

<p align="center">

# Rol a desarrollar:
En este proyecto se nos propone, desarrollar el rol de un Data Scientist que comenzo a trabajar en la plataforma de distribución digital de videojuegos Steam.  
 
# Objetivos:
Realizar un MVP que cumpla con las siguientes etapas:

A partir de los dataset que se encuentran en la carpeta raw del siguiente drive https://drive.google.com/drive/folders/1pS38Xrc6Kt4QQ5mDY5i6nmbCINOxzWOT

Se solicita:

 1- Transformaciones: 
Para este MVP no se te pide transformaciones de datos especificas pero trabajaremos en leer el dataset con el formato correcto.

 2- Feature Engineering: 
En el dataset user_reviews se incluyen reseñas de juegos hechos por distintos usuarios. Crear la columna 'sentiment_analysis' aplicando análisis de sentimiento con NLP con la siguiente escala: debe tomar el valor '0' si es malo, '1' si es neutral y '2' si es positivo. Esta nueva columna debe reemplazar la de user_reviews.review para facilitar el trabajo de los modelos de machine learning y el análisis de datos. De no ser posible este análisis por estar ausente la reseña escrita, debe tomar el valor de 1. 

3- Desarrollo API: 

Desarrollar las siguientes consultas:

* def userdata( User_id : str ): Debe devolver cantidad de dinero gastado por el usuario, el porcentaje de recomendación en base a reviews.recommend y cantidad de items.

* def countreviews( YYYY-MM-DD y YYYY-MM-DD : str ): Cantidad de usuarios que realizaron reviews entre las fechas dadas y, el porcentaje de recomendación de los mismos en base a reviews.recommend.

* def genre( género : str ): Devuelve el puesto en el que se encuentra un género sobre el ranking de los mismos analizado bajo la columna PlayTimeForever.

* def userforgenre( género : str ): Top 5 de usuarios con más horas de juego en el género dado, con su URL (del user) y user_id.

* def developer( desarrollador : str ): Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora. 

* def sentiment_analysis( año : int ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento.
Ejemplo de retorno: {Negative = 182, Neutral = 120, Positive = 278}

4- Deployment: 

Utilizar algun servicio que permita que la API pueda ser consumida desde la web.

5- Análisis exploratorio de los datos: 

Investigar las relaciones que hay entre las variables del dataset, ver si hay outliers o anomalías, y ver si hay algún patrón interesante que valga la pena explorar en un análisis posterior.

6- Modelo de aprendizaje automático: 

Tenemos dos propuestas de trabajo: En la primera, el modelo deberá tener una relación ítem-ítem, esto es se toma un item, en base a que tan similar esa ese ítem al resto, se recomiendan similares. Aquí el input es un juego y el output es una lista de juegos recomendados, para ello recomendamos aplicar la similitud del coseno. La otra propuesta para el sistema de recomendación debe aplicar el filtro user-item, esto es tomar un usuario, se encuentran usuarios similares y se recomiendan ítems que a esos usuarios similares les gustaron. Se debe crear al menos uno de los dos sistemas de recomendación. El líder pide que el modelo derive obligatoriamente en un GET/POST en la API símil al siguiente formato:

Si es un sistema de recomendación item-item:


* def recomendacion_juego( id de producto ): Ingresando el id de producto, deberíamos recibir una lista con 5 juegos recomendados similares al ingresado.

Si es un sistema de recomendación user-item:

* def recomendacion_usuario( id de usuario ): Ingresando el id de un usuario, deberíamos recibir una lista con 5 juegos recomendados para dicho usuario.

7- Video: 
Hacer un video mostrando el resultado de las consultas propuestas y de tu modelo de ML entrenado.

----------------------

# Contenido de este repo:

1_extraccion.ipynb: En este notebook se extraen los datos de los archivos .gz que fueron proporcionados. 

2_tablas_principales.ipynb: En este notebook se crean las tablas: df_users, df_items, df_genres, df_reviews y las tablas intermedias user_items_df e item_genre

3_relaciones_ente_tablas.ipynb: En este notebook se analizan las relaciones entre las tablas y se soluciona la falta de congruencia entre las mismas.

4_tablas_para_consultas: En este notebook se crean tablas especificas para algunos de los endpoints requeridos.

5_consultas: En este notebook se crean y prueban las funciones solicitadas.

6_sistema_de_recomendacion.ipynb: En este notebook se realiza el modelo de aprendizaje automático solicitado en la primer propuesta de trabajo, es decir una relación ítem-ítem en la que ingresando el id de producto, recibimos una lista con 5 juegos recomendados similares al ingresado.

main.py: Archivo en el que se encuentran las funciones para los endpoints que se consumirán en la API, junto con sus correspondientes decoradores.

requirements.txt: Archivo que contiene las librerías utilizadas y permite automatizar la instalación de las mismas

-------------

# Material externo:

Las consultas a la API pueden realizarse en el siguiente link: https://rominaesc-steam-games-deploy.onrender.com/docs

El video que muestra el deploy exitoso se encuentra en https://drive.google.com/drive/folders/1pS38Xrc6Kt4QQ5mDY5i6nmbCINOxzWOT
