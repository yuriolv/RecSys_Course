import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process



#baixando as bases de dados
movies_df = pd.read_csv('./movies.csv', usecols=['movieId', 'title'])
ratings_df = pd.read_csv('./Ratings.csv', usecols=['movieId', 'userId', 'rating'])
#print(movies_df.head())
#print(ratings_df.head())



#transformando o df de ratings em uma matriz de i = filmes, j = avaliações dos usuários 
movies_users = ratings_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
mat_movies = csr_matrix(movies_users.values) #matriz de vetores de filmes com atributos de avaliação



#treinando o modelo
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20)
model.fit(mat_movies)



#função que gera a recomendação para um filme 'selecionado' 

def recommender(movie_name, data , n):
    idx = process.extractOne( movie_name, movies_df['title'])[2]
    print('\n\nFilme Selecionado: ',movies_df[ 'title'][idx], 'Index : ',idx)
    print('Buscando recomendações baseadas na sua escolha...\n')
    distance, indices  = model.kneighbors(data[idx], n_neighbors=n+1)
    for i in indices:
        for j in i:
            if j != idx:
                print(movies_df['title'][j])
    


recommender('Jurassic Park', mat_movies,  10)