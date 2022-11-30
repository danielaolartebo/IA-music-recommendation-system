import dash
import dash_bootstrap_components as dbc
import numpy as np                                              #Matemática y manejo de arreglos.
from dash import html, dcc
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import cdist                        #Cálcula la distancia entre dos vectores.
from dash.dependencies import Input, Output
import joblib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import warnings
warnings.filterwarnings("ignore")

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='cd8951e567fc43b1b4d6e485749fd646', client_secret='148e45e4e0fa4358831887cffba9af24'))

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

data = joblib.load("data.joblib")

scaler = joblib.load("scaler.joblib")

kmeans = joblib.load("model.joblib")

nombre_de_las_variables_de_entrada = joblib.load("nombre_de_las_variables_de_entrada.joblib")


def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)

    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]
    song_data['name'] = name
    song_data['year'] = year
    song_data['explicit'] = int(results['explicit'])
    song_data['duration_ms'] = results['duration_ms']
    song_data['popularity'] = results['popularity']

    for key, value in audio_features.items():
        song_data[key] = value

    song_data = pd.DataFrame(song_data, index=[0])

    song_genres = []
    for artist in results['artists']:
        song_genres.extend(sp.artist(artist_id=artist['id'])['genres'])
    song_genres = list(set(song_genres))
    for genre in nombre_de_las_variables_de_entrada[15:]:
        if(genre in song_genres):
            song_data[genre] = [1]
        else:
            song_data[genre] = [0]
    return song_data


def get_song_data(song, data):
    try:
        #Se obtienen los datos de una canción con base en su nombre y año.
        song_data = data[(data['name'] == song['name']) & (data['year'] == song['year'])].iloc[0]
        return song_data

    except IndexError:
        return find_song(song['name'], song['year'])


def get_mean_vector(song_list, data):

    song_vectors = []                                                               #Se inicializa el vector media.

    for song in song_list:
        song_data = get_song_data(song, data)                                       #Se obtiene el vector (datos) de cada canción.

        if song_data is None:                                                       #En caso de que una canción no se encuente en el dataset principal
            raise ValueError('Error: {} does not exist in database or Spotify'.format(song['name'])) #se lanza un error

        song_vector = song_data[nombre_de_las_variables_de_entrada].values          #Se crea una copia del vector de datos de las canciones
        song_vectors.append(song_vector)                                            #sólo con las columnas que nos interesan.

    song_matrix = np.array(list(song_vectors))                                      #Se crea una matriz a partir de la información anterior.

    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()                  #Se inicializa el diccionario.

    for key in dict_list[0].keys():
        flattened_dict[key] = []                    #Se añaden las claves del diccionario.

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)       #Se añaden los valores del diccionario.

    return flattened_dict


def recommend_songs(song_list, data, n_songs=10):   #Se recomiendan 10 canciones a partir de una lista de canciones de entrada y el dataset principal.

    metadata_cols = ['name', 'url', 'track_url']       #Información que se retornará de cada canción recomendada.
    song_dict = flatten_dict_list(song_list)                    #Se crea un diccionario con la lista de canciones de entrada.

    song_center = get_mean_vector(song_list, data)              #Se obtiene el vector media de la lista de canciones de entrada.

    scaled_song_center = scaler.transform(song_center.reshape(1, -1))   #Se normaliza el vector media de la lista de canciones de entrada.

    cluster = kmeans.predict(scaled_song_center)    #Se predice el cluster al que pertenece el vector media de la lista de canciones de entrada.

    scaled_data = scaler.transform(data[data.cluster == cluster[0]][nombre_de_las_variables_de_entrada])    #Se normalizan los datos de las canciones que pertenecen al mismo cluster.

    distances = cdist(scaled_song_center, scaled_data, 'cosine')    #Se encuentra la distancia coseno se cada canción en el cluster con el vector media.

    index = list(np.argsort(distances)[:, :n_songs][0])     #Se eligen los 10 indices con menor distancia coseno.

    rec_songs = data[data.cluster == cluster[0]].reset_index().iloc[index]   #Se encuentra toda la información de las 10 cacniones a partir de su indice.
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]   #En caso de que una canción recomendada sea igual a una canción de entrada está recomendación se omite.
    img_url = []
    track_url = []
    for index, song in rec_songs.iterrows():
        results = sp.search(q= 'track: {} year: {}'.format(song['name'], song['year']), limit=1)
        img_url.append(results['tracks']['items'][0]['album']['images'][1]['url'])
        track_url.append(results['tracks']['items'][0]['external_urls']['spotify'])
    rec_songs['url'] = img_url
    rec_songs['track_url'] = track_url
    return rec_songs[metadata_cols].to_dict(orient='records')   #Se crea un diccionario con la información que se desea de cada canción recomendada.

def generate_table(dataframe, track_url, max_rows=10):
    return html.Table(
        html.Tbody([
            html.Tr([
                html.Td(render(col, i, dataframe, track_url)) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    , style={"margin": "auto"})

def render(col, i, dataframe, track_url):
    if(col == 'url'):
        return html.A(href=track_url[i], target="_blank", children=html.Img(src=dataframe.iloc[i][col]))
    else:
        return dataframe.iloc[i][col]


app.title = 'Music Recommendation System'
app.layout = html.Div([
    html.Br(),
    html.Div(html.H1('Music Recommendation System'), style={"textAlign": "center"}),
    html.Br(),
    html.Div([
        html.Table([
            html.Tbody(
                html.Tr([
                    html.Td([
                        html.Label("Song name:", htmlFor="input-song-name"),
                        dcc.Input(id="input-song-name", value='', type="text", className="form-control")
                    ]),
                    html.Td([
                        html.Label("Song year:", htmlFor="input-song-year"),
                        dcc.Input(id="input-song-year", type="number", className="form-control")
                    ])
                ])
            )
        ], style={"margin": "auto"}),
        html.Br(),
        html.Br(),
        html.Div(id="output-recommended-songs")
    ])
])


@app.callback(
    Output(component_id="output-recommended-songs", component_property="children"),
    Input(component_id="input-song-name", component_property="value"),
    Input(component_id="input-song-year", component_property="value")
)
def recommendation_output_div(song_name, song_year):
    if song_name == '' or str(song_year) == 'None':
        return html.Table(style={"margin": "auto"})
    try:
        songs = recommend_songs([{"name": song_name, "year": song_year}], data)
    except ValueError as e:
        return html.Table(html.Tbody(html.Tr(html.Td(str(e)))), style={"margin": "auto"})
    songs = pd.DataFrame(songs)
    return generate_table(songs[['name', 'url']], songs['track_url'])


if __name__ == '__main__':
    app.run_server(debug=True)
