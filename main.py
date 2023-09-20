from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from pprint import pprint
from wordcloud import WordCloud, STOPWORDS


app = FastAPI()

# cargamos los datos
df_genres = pd.read_csv('data/df_genres.csv')
df_item_genre = pd.read_csv('data/df_item_genre.csv')
df_items = pd.read_csv('data/df_items.csv')
df_reviews = pd.read_csv('data/df_reviews.csv')
df_users_items = pd.read_csv('data/df_users_items.csv')
df_users = pd.read_csv('data/df_users.csv')
df_user_id_item_id_price = pd.read_csv("data/df_user_id_item_id_price.csv")
df_developer = pd.read_csv('data/df_developer.csv')
sist_recomendacion_df=pd.read_csv('data/sist_recomendacion_df_item.csv')



# ---------------------------------------------------------------------------------------

# def check(input,df,columna):
#     return any(part in input.lower() for part in df[columna])

# ---------------------------------------------------------------------------------------

# Función 1: función userdata

@app.get('/userdata/{user_id}')

def userdata(user_id: str):
    # Si el usuario no existe devuelve: {'dinero_gastado': 0.0, 'porcentaje_de_recomendaciones': nan, 'items_recomendados': 0, 'items_comprados': 0}
    
    user_id = user_id.strip()

    if any(part in user_id for part in df_user_id_item_id_price['user_id']):

        # Dinero gastado
        df_filtrado_dinero = df_user_id_item_id_price[df_user_id_item_id_price.user_id == user_id]
        dinero_gastado = round(df_filtrado_dinero.price.sum(),2)

        # Porcentaje de recomendaciones
        df_filtrado_recomendaciones = df_reviews[df_reviews.user_id == user_id]
        
        if df_filtrado_recomendaciones.shape[0] == 0:
            porcentaje_de_recomendacion = 0
        else:
            porcentaje_de_recomendacion = round(df_filtrado_recomendaciones.recommend.mean(),3)*100
        
        # items comprados
        items_comprados = int(df_filtrado_dinero.shape[0])
        
        # items recomendados
        items_recomendados = int(df_filtrado_recomendaciones.shape[0])

        out = {"dinero_gastado":dinero_gastado, "porcentaje_de_recomendaciones": porcentaje_de_recomendacion, 'items_recomendados': items_recomendados,'items_comprados':items_comprados}

    else:
         out = (F'El usuario {user_id}, no éxiste. Intente nuevamente. EJ: "76561197970982479"')

    return out

# ---------------------------------------------------------------------------------------
# 2. Función 2: countreviews

@app.get('/countreviews/{fecha_inicio}')

def countreviews (fecha_inicio:(str), fecha_fin):

    df_filtrado = df_reviews[(df_reviews['posted_date'] >= fecha_inicio) & (df_reviews['posted_date'] <= fecha_fin)]
    cant_usuarios = df_filtrado.shape[0]
    porcentaje_de_recomendacion = round(df_filtrado.recommend.mean(),3)*100
    return {"cantidad":cant_usuarios, "porcentaje":porcentaje_de_recomendacion}
    # return df_filtrado

# Función 3: función genre

@app.get('/genre/{genero}')

def genre (genero):
    genero_minusculas = genero.lower().strip()
    if genero_minusculas in list(df_genres.name):
        puesto = int(df_genres[df_genres.name == genero_minusculas].reset_index().at[0,'ranking'])
        salida = {"genero": genero_minusculas, "ranking":puesto}
    else:
        puesto = 'No se encuentra. Intente nuevamente. Ej: Action'
        salida = {"genero": genero_minusculas, "ranking":puesto}
    return salida


# Función 4: función genre

@app.get('/userforgenre/{genero}')


def userforgenre( genero : str ): 
    genero_min = genero.lower().strip()

    if genero_min in list(df_item_genre.genres):

        lista_de_items_del_genero = list(df_item_genre[df_item_genre.genres == genero_min].item_id)
        df = df_users_items[df_users_items.item_id.isin(lista_de_items_del_genero)]

        top5 = df.groupby('user_id')['playtime_forever'].sum().reset_index()
        top5 = top5.sort_values('playtime_forever',ascending=False).head() # aca ta el user id y las hs de juego
        top5 = top5.merge(df_users[['user_id','user_url']], on='user_id', how='left')
        users_top5 = list(top5.user_id)
        user_url_top5 = list(top5.user_url)

        # out = {"users_top5":users_top5}, 
        out ={"user_url_top5":user_url_top5}
        return out

    else:
        out = {f'el género {genero} no éxiste. Intente nuevamente. EJ: "Action"'}
        return out


# Función 5:

@app.get('/developer/{desarrollador}')

def developer(desarrollador : str ):

    desarrollador_minusculas = desarrollador.lower().strip()

    
    if any(part in desarrollador_minusculas for part in df_developer['developer']):
        
        df_filtrado = df_developer[df_developer.developer == desarrollador_minusculas]


        df_xanio = df_filtrado.groupby(['year']).size().reset_index(name=('cantidad_total'))
        df_filtrado_free = df_filtrado[df_filtrado.price == 0].groupby(['year']).size().reset_index(name=('cantidad_free'))
        df = df_xanio.merge(df_filtrado_free,how='left')
        cantidad_de_items = df.shape[0]
        df['porcentaje'] = round(df.cantidad_free /df.cantidad_total,3) *100
        df['porcentaje'] = df['porcentaje'].fillna(0)


        anios = df_xanio['year'].tolist()
        porcentaje_free_por_anio = df['porcentaje'].tolist()

        out = {
            "cantidad_de_items": cantidad_de_items,
            "anios": anios,
            "porcentaje_free_por_anio": porcentaje_free_por_anio
            }
        
    else:
         out = {F'El desarrollador {desarrollador}, no éxiste. Intente nuevamente. EJ: "kotoshiro"'}

    return out


# Función 6: sentiment_analysis

@app.get('/sentiment_analysis/{anio}')

def sentiment_analysis( anio : int ):

    df_filtrado = df_reviews[df_reviews.year == anio].groupby(['sentiment_analysis']).size().reset_index(name=('cantidad'))

    negative = int(df_filtrado.iloc[0,1])
    neutral = int(df_filtrado.iloc[1,1])
    positive = int(df_filtrado.iloc[2,1])

    return {"Negative":negative,"Neutral":neutral,"Positive":positive}


# --------

# Instanciamos el CV
vectorizer = CountVectorizer()
stopwords = STOPWORDS
# eliminamos las "stop words", palabras comunes no informativas
tf = TfidfVectorizer(stop_words='english')

# calculamos los features para cada ítem (texto)
tfidf_matrix = tf.fit_transform(sist_recomendacion_df['texto'])

# calculamos las similitudes entre todos los documentos
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
n = 5

results = {} 
for idx, row in sist_recomendacion_df.iterrows():
    # guardamos los indices similares basados en la similitud coseno. Los ordenamos en modo ascendente, siendo 0 nada de similitud y 1 total
    similar_indices = cosine_similarities[idx].argsort()[:-n-2:-1] 
    # guardamos los N más cercanos
    similar_items = [(f"{sist_recomendacion_df.loc[i, 'title']}") for i in similar_indices]
    results[f"{row['title']}"] = similar_items[1:]

@app.get('/recomendacion/{titulo}')

def recomendacion(titulo:str):
    '''Ingresas un nombre de pelicula y te recomienda las similares en una lista'''
    titulo = titulo.title().strip()

    if sist_recomendacion_df['title'].str.contains(titulo).any():
        titulo = titulo.title().strip()
        lista = (results[titulo])
        data = {'titulo':titulo , 'lista recomendada': lista}
    else:
        mensaje = "El item ingresado: {}, no se encuentra en la base de datos.".format(titulo)
        data = {mensaje}    
    return data