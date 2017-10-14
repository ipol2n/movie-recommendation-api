import numpy as np
import pandas as pd
from sklearn.externals import joblib
import sklearn
from sklearn.decomposition import TruncatedSVD

class Recommendation:
    def predict(movie_name):
        utility_matrix = pd.read_csv('movie-names.csv', header=0)
        print(utility_matrix.head())
        resultant_matrix = joblib.load('movie-recommendation.pkl')
        corr_matrix = np.corrcoef(resultant_matrix)
        movie_names = utility_matrix.columns
        movie_list = list(movie_names)
        if movie_name in movie_list:
            target_movie_index = movie_list.index(movie_name)
            corr_target_movie = corr_matrix[target_movie_index]
            result = list(movie_names[(corr_target_movie<1.0) & (corr_target_movie>0.9)])
        else:
            result = None
        return result

    def training():
        columns = ['user_id', 'item_id', 'rating', 'timestamp']
        frame = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)
        columns = ['item_id', 'movie title', 'release date', 'video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
          'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

        movies = pd.read_csv('ml-100k/u.item', sep='|', names=columns, encoding='latin-1')
        movie_names = movies[['item_id', 'movie title']]
        combined_movies_data = pd.merge(frame, movie_names, on='item_id')
        utility_matrix = pd.pivot_table(data=combined_movies_data, values='rating', index='user_id', columns='movie title', fill_value=0)
        X = utility_matrix.T
        SVD = TruncatedSVD(n_components=12, random_state=17)
        resultant_matrix = SVD.fit_transform(X)
        joblib.dump(resultant_matrix, 'movie-recommendation.pkl')
        utility_matrix.to_csv('movie-names.csv', index_label=False)
