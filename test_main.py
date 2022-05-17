from main import *
import unittest

links_df = pd.read_csv('D:/python/test2/links.csv')
movie_df = pd.read_csv('D:/python/test2/movies.csv')
df_ratings = pd.read_csv('D:/python/test2/ratings.csv')

user2user_encoded, userencoded2user, movie2movie_encoded, movie_encoded2movie, num_users, num_movies, min_rating, max_rating = pre_processing(
    df_ratings)
x_train, x_val, y_train, y_val = preparing_data(df_ratings, min_rating, max_rating)
model, history = train_model(num_users, num_movies, x_train, x_val, y_train, y_val)
rec_movies = find_movies(df_ratings, movie_df, model, movie2movie_encoded, user2user_encoded, movie_encoded2movie, 1,
                         'Drama')
rec_movies2 = find_movies(df_ratings, movie_df, model, movie2movie_encoded, user2user_encoded, movie_encoded2movie, 1,
                          'без выбора жанра')

model1, history1 = train_model(612, 9724, [[65, 2657], [287, 27], [569, 2419], [12, 1021]],
                               [[324, 1018], [0, 2], [103, 4229], [440, 7727]],
                               [0.77777778, 0.55555556, 0.66666667, 1.0], [0.55555556, 0.77777778, 0.55555556, 1.0])


class Test_main(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_pre_processing(self):
        self.assertEqual(type(pre_processing(df_ratings)), tuple)

    def test_preparing_data(self):
        self.assertEqual(type(preparing_data(df_ratings, 2.0, 5.0)), tuple)

    def test_create_model(self):
        self.assertEqual(type(create_model(50, 100)), RecommenderNet)

    def test_train_model(self):
        self.assertEqual(type(train_model(612, 9724, [[65, 2657], [287, 27], [569, 2419], [12, 1021]],
                                          [[324, 1018], [0, 2], [103, 4229], [440, 7727]],
                                          [0.77777778, 0.55555556, 0.66666667, 1.0],
                                          [0.55555556, 0.77777778, 0.55555556, 1.0])), tuple)

    def test_find_movies(self):
        self.assertEqual(type(
            find_movies(df_ratings, movie_df, model1, {0: 1, 3: 1, 6: 2, 47: 3}, {1: 0, 2: 1, 3: 2, 4: 3},
                        {1: 0, 1: 3, 2: 6, 3: 47}, 2, 'без выбора жанра', '')), str)

    def test_genre(self):
        arr = rec_movies.split('>')
        arr2 = []
        for i in range(1, len(arr) - 1, 3):
            arr2.append(arr[i].replace('</a', '')[1:])

        check = True

        for i in arr2:
            r = movie_df[movie_df['title'] == i]['genres'].values[0].split('|')

            if not 'Drama' in r:
                check = False
        self.assertEqual(check, True)

    def test_movies_not_watched(self):
        new_df = movie_df.merge(df_ratings)
        arr = rec_movies2.split('>')
        arr2 = []
        for i in range(1, len(arr) - 1, 3):
            arr2.append(arr[i].replace('</a', '')[1:])  # названия рекомендуемых фильмов

        check = True

        for i in arr2:
            user_df = new_df[(new_df['userId'] == 1) & (new_df['title'] == i)]
            if not user_df.empty:
                check = False
        self.assertEqual(check, True)


unittest.main()
