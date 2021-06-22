from MatrixFactorizationModel import MatrixFactorizationModel;
import pandas as pd
import numpy as np

'''
@author Noah Teshima
Alg:
1. Get userId list list_user
2. Get movieId list list_movie
3. For each user:
    a. Get ratings by user
    b. For each index of ratings in users_ratings dataframe:
        i. Get movieid index (column index) in ratings['movieId']
        ii. Append (user, movie, rating) to output list
4. Returns 3-tuple (output_list, list_user, list_movie)
'''
def get_user_item_pairs(movie_database, users_ratings):
    list_userId = np.array(pd.unique(users_ratings['userId']))
    list_movieId = np.array(pd.unique(movie_database['movieId']))
    movie_index_list = pd.Index(movie_database['movieId'])
    output_list = []
    # index by user
    for user_index in np.arange(0, len(list_userId)):
        # associated user_id with user
        user_id = list_userId[user_index]
        # ratings by user
        ratings = users_ratings[users_ratings['userId'] == user_id]
        # for each row index associated with user
        for index in ratings.index:
            # get movie id for finding movie index
            movie_id = users_ratings['movieId'].iloc[index]
            # get movie index to place in list
            movie_index = movie_index_list.get_loc(movie_id)
            # get rating
            rating = users_ratings['rating'].iloc[index]
            output_list.append((user_index, movie_index, rating))
    return (np.array(output_list, dtype='i4,i4,i4'), list_userId, list_movieId)

'''
@author Luis Jibaja
'''
def display_rank(movie_database,k_top_movies_id, k_top_scores,measure):
    movies_rank = pd.DataFrame()
    measure_key = measure + ' score'
    for i in range(len(k_top_movies_id)):
        movie_id = k_top_movies_id[i]
        movie = movie_database[movie_database['movieId']==movie_id].copy()
        movie = movie.drop('genres',axis=1)
        movie[measure_key] = k_top_scores[i]
        movies_rank = movies_rank.append(movie)
    
    movies_rank = movies_rank.sort_values([measure_key],ascending=False)
    print(movies_rank)

'''
@author Luis Jibaja
'''
def user_base_rank(U, V, list_userId, list_movieId, movie_database, user_id, num_recomendations=5, measure='dot'):
    user_index = -1 
    user_index = np.where(list_userId==user_id)
    if user_index ==  -1:
        print("Error: User Id not valid, Not Found")
        return
    
    user_embedding = U[user_index]
    scores = compute_scores(user_embedding,V,similarity_mesure=measure)[-1]

    k_top_scores_index = np.argsort(scores)[-1*num_recomendations:]
    k_top_scores = scores[k_top_scores_index]

    k_top_moviesId = list_movieId[k_top_scores_index]
    display_rank(movie_database,k_top_moviesId,np.around(k_top_scores,2),measure)


'''
@author Luis Jibaja
'''
def movie_base_rank(U, V, list_userId, list_movieId, movie_database,movie_title,num_recomendations=5,measure='dot'):
    movies_index = movie_database[movie_database['title'].str.contains(movie_title)].index.values
    if len(movies_index) == 0:
        print("Error: " + movie_title)
        print("Movie Title invalid, Not Found")
        return

    movie_index = movies_index[0]
    movie_embedding = V[movie_index]

    scores = compute_scores(movie_embedding,V,similarity_mesure=measure)

    k_nearest_scores_index = np.argsort(scores)[-1*num_recomendations:]
    k_nearest_scores = scores[k_nearest_scores_index]

    k_nearest_moviesId = list_movieId[k_nearest_scores_index]

    display_rank(movie_database,k_nearest_moviesId,np.around(k_nearest_scores,2),measure)

'''
@author Luis Jibaja
'''
def compute_scores(query_embedding,item_embedding,similarity_mesure='dot'):
    """
        Similarity Mesure: Dot Product or Cosine
    """    
    if similarity_mesure == 'dot':
        score = np.dot(query_embedding,item_embedding.T)

    elif similarity_mesure == 'cos':
        query_norm = np.linalg.norm(query_embedding)
        item_norm = np.linalg.norm(item_embedding)
        score = np.dot(query_embedding,item_embedding.T) / (query_norm * item_norm)

    return score

def main():
    movie_database = pd.read_csv('./notebooks/ml-latest-small/movies.csv')
    users_ratings = pd.read_csv('./notebooks/ml-latest-small/ratings.csv')
    (user_item_rating, list_userId, list_movieId) = get_user_item_pairs(movie_database, users_ratings)
    matrix_factorization_model = MatrixFactorizationModel(user_item_rating)
    # matrix_factorization_model.train(list_userId, list_movieId, .00005, 50, 2)
    matrix_factorization_model.depickle('modelName')
    (U, V, cost) = matrix_factorization_model.get_model()
    movie_base_rank(U, V, list_userId, list_movieId, movie_database, 'Aladdin', 5, 'dot')
    
if __name__ == '__main__':
    main()


