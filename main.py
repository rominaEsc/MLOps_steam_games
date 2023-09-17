from fastapi import FastAPI
import pandas as pd


app = FastAPI()

# cargamos los datos
df_genres = pd.read_csv('data/df_genres.csv')
df_item_genre = pd.read_csv('data/df_item_genre.csv')
df_items = pd.read_csv('data/df_items.csv')
df_reviews = pd.read_csv('data/df_reviews.csv')
df_users_items = pd.read_csv('data/df_users_items.csv')
df_users = pd.read_csv('data/df_users.csv')
df_user_id_item_id_price = pd.read_csv("data/df_user_id_item_id_price.csv")



# Función 1: función userdata

@app.get('/userdata/{user_id}')

def userdata(user_id: str):
    # si el usuario no existe devuelve: {'dinero_gastado': 0.0, 'porcentaje_de_recomendaciones': nan, 'items_recomendados': 0, 'items_comprados': 0}
    user_id = user_id.strip()
    df_filtrado_dinero = df_user_id_item_id_price[df_user_id_item_id_price.user_id == user_id]
    dinero_gastado = df_filtrado_dinero.price.sum()
    df_filtrado_recomendaciones = df_reviews[df_reviews.user_id == user_id]
    porcentaje_de_recomendacion = round(df_filtrado_recomendaciones.recommend.mean(),3)*100
    items_comprados = df_filtrado_dinero.shape[0]
    items_recomendados = df_filtrado_recomendaciones.shape[0]


    return {"dinero_gastado":dinero_gastado, "porcentaje_de_recomendaciones": porcentaje_de_recomendacion, 'items_recomendados': items_recomendados,'items_comprados':items_comprados}

# 2. falta

#@app.get('/userdata/{user_id}')


# Función 3: función genre

@app.get('/genre/{genero}')


def genre (genero):
    genero_minusculas = genero.lower().strip()
    if genero_minusculas in list(df_genres.name):
        puesto = df_genres[df_genres.name == genero_minusculas].reset_index().at[0,'ranking']
    else:
        puesto = 'No se encuentra. Intente nuevamente. Ej: Action'

    return {"genero": genero_minusculas, "ranking":puesto}

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

        return {"users_top5":users_top5, "user_url_top5":user_url_top5}

    else:
        mensaje = 'el género ', genero, ' no éxiste. Intente nuevamente. EJ: "Action"'
        return mensaje
