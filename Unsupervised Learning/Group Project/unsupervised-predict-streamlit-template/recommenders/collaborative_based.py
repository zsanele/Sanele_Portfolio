"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from fuzzywuzzy import fuzz

# Importing data
df_movies = pd.read_csv('resources/data/movies.csv',sep = ',',delimiter=',')
df_ratings = pd.read_csv('resources/data/ratings.csv')
df_ratings.drop(['timestamp'], axis=1,inplace=True)

# We make use of an K-Nearest Neighbors algorithm model trained on a subset of the MovieLens 10k dataset.

#Preprocessing

df_movies_cnt = pd.DataFrame(df_ratings[:10000].groupby('movieId').size(), columns=['count'])

#now we need to take only movies that have been rated at least 5 times to get some idea of the reactions of users towards it
popularity_thres = 5
popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]

df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])

# filter data to come to an approximation of user likings.
ratings_thres = 2
active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]


# pivot and create movie-user matrix
movie_user_mat = df_ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)
#map movie titles to images

movie_to_idx = {
    movie: i for i, movie in 
    enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title))
}
# transform matrix to scipy sparse matrix
movie_user_mat_sparse = csr_matrix(movie_user_mat.values)



def fuzzy_matching(mapper, movie_list):

    """
    return the closest match via fuzzy ratio. 
    
    Parameters
    ----------    
    mapper: dict, map movie title name to index of the movie in data

    movie_list: list, name of user input movies
    

    Return
    ------
    index of the closest match
    """
    import random
 
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), movie_list[random.randint(0, 2)].lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    ''''if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))'''
    return match_tuple[0][1]





# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
        """
        return top n similar movie recommendations based on user's input movie


        Parameters
        ----------

        movie_list: list, name of user input movie

        top_n: int, top n recommendations

        Return
        ------
        list of top n similar movie recommendations
        """
        #fit
        model_knn=pickle.load(open('resources/models/model_knn.pkl', 'rb'))
        data=movie_user_mat_sparse
        model_knn.fit(data)
        # get input movie index
        #print('You have input movie:', movie_list)
        mapper=movie_to_idx
        idx = fuzzy_matching(mapper, movie_list)
        
        #print('Recommendation system start to make inference')
        #print('......\n')
        distances, indices = model_knn.kneighbors(data[idx], n_neighbors=top_n+1)

        # get reverse mapper
        reverse_mapper = {v: k for k, v in mapper.items()}

        raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

        final = []
        for i, (idx, dist) in enumerate(raw_recommends):
            final.append((' {1}'.format(i+1, reverse_mapper[idx])))    
        return final


        