'''
The IMDB data can be accessed at:
                                  https://www.kaggle.com/orgesleka/imdbmovies


query: the movie you need to received recommendations based on
k: number of recommendations with respect to euclidean distances between query and movies

'''


def knn(data, query, k):
    distances_and_indices = []

    for idx, sample in enumerate(data[:,1:]):
        distance = euc_distance(sample,query)
        distances_and_indices.append((distance, idx))

    sorted_dist_ind = sorted(distances_and_indices)
    movies = [data[i, 0] for d,i in sorted_dist_ind[:k]]

    return movies



def euc_distance(sample, query):
    summation = 0
    for i, j in zip(sample, query):
        summation += np.power(i - j, 2)
        
    return np.sqrt(summation)



def recommend_movies(data, query, k):
    return knn(data, query, k)


if __name__ == '__main__':
    data = pd.read_csv('path', error_bad_lines=False)
    df = pd.concat([data.loc[:, 'title'], data.loc[:, 'Action':]], axis=1).values

    imdb = [8]   # imdb point
    a_movie = np.concatenate((imdb, np.random.randint(0,2, size=27)))  # randomly created genres

    movies = recommend_movies(df, a_movie, 5)

    print(movies)
