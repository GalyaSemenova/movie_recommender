# from PyQt6 import uic

from PyQt6.QtWidgets import QFileDialog
from PyQt6 import QtCore, QtGui, QtWidgets
from ui import Ui_MainWindow
from PyQt6.QtGui import QIcon
# Основа - совместная фильтрация с использованием набора данных MovieLens для рекомендаций фильмов пользователям
# Шаги в модели следующие:
# 1. Сопоставить id пользователя с «вектором пользователя» с помощью матрицы внедрения
# 2. Сопоставить id фильма с «вектором фильма» с помощью матрицы встраивания
# 3. Вычислить скалярное произведение между вектором пользователя и вектором фильма, чтобы получить оценку совпадения
#    между пользователем и фильмом (прогнозируемый рейтинг)
# 4. Обучить вложения с помощью градиентного спуска, используя все известные пары «пользователь-фильм».

# Item-based collaborative filtering recommendation algorithms
# Neural Collaborative Filtering

# Есть две функции с двумя переменными х и тета. надо одновременно их найти. можно по очереди подбирать параметры, но
# есть функция, которая оптимизированно находить оба параметра сразу. В формуле используется квадрат ошибки, который
# вычисляется как 1) сумма всех оценок пользователей по каждому фильму 2) сумма всех оценок фильмов для каждого
# пользователя
# Нам надо минимизировать х по отношению к тета и тета по отношению к х. Это мы и решаем.

import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from recommender import RecommenderNet
import sys


# предварительная обработка для кодирования пользователей и фильмов в виде целочисленных индексов.
def pre_processing(df):
    user_ids = df["userId"].unique().tolist()
    #присвоили по порядку id пользователям
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    movie_ids = df["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    df["user"] = df["userId"].map(user2user_encoded)
    df["movie"] = df["movieId"].map(movie2movie_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    df["rating"] = df["rating"].values.astype(np.float32)

    # минимальные и максимальные рейтинги будут использоваться для нормализации рейтингов позже
    min_rating = min(df["rating"])
    max_rating = max(df["rating"])

    return user2user_encoded, userencoded2user, movie2movie_encoded, movie_encoded2movie, num_users, num_movies, min_rating, max_rating


# Подготовка данных для обучения и проверки
def preparing_data(df, min_rating, max_rating):
    df = df.sample(frac=1, random_state=42)
    x = df[["user", "movie"]].values
    # Нормируем цели от 0 до 1. Облегчает обучение.
    y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

    # Предполагаем обучение на 90% данных и проверку на 10%.
    train_indices = int(0.9 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )
    return x_train, x_val, y_train, y_val


def create_model(num_users, num_movies):
    model = RecommenderNet(num_users, num_movies, RecommenderNet.EMBEDDING_SIZE)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001)
    )
    return model


# Обучение модели на основе разделения данных
def train_model(num_users, num_movies, x_train, x_val, y_train, y_val):
    model = create_model(num_users, num_movies)
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=5,
        verbose=1,
        validation_data=(x_val, y_val),
    )
    return model, history


def draw_plot(history):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()


def find_movies(df, movie_df, model, movie2movie_encoded, user2user_encoded, movie_encoded2movie, userId, user_genre,
                rec_movies=''):
    links_df = pd.read_csv('links.csv')
    res = movie_df.merge(links_df, on=["movieId"])

    genre = user_genre
    # получим пользователя и посмотрим лучшие рекомендации
    user_id = userId
    movies_watched_by_user = df[df.userId == user_id]
    movies_not_watched = movie_df[
        ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
    ]["movieId"]
    movies_not_watched = list(
        set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
    )
    movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_array = np.hstack(
        ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
    )
    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]
    recommended_movie_ids = [
        movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
    ]


    print("Movies with high ratings from user")
    print("----" * 8)
    top_movies_user = (
        movies_watched_by_user.sort_values(by="rating", ascending=False)
            .head(5)
            .movieId.values
    )
    movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]

    for row in movie_df_rows.itertuples():
        print(row.title, ":", row.genres)

    # print("----" * 8)
    # if genre == None:
    #     print("Top 10 movie recommendations")
    # else:
    #     print("Top movie recommendations")
    # print("----" * 8)
    recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]

    if genre == 'без выбора жанра':
        for row in recommended_movies.itertuples():
            row_link = res[res['title'] == row.title]['imdbId'].values[0]
            size = 7 - len(str(row_link))
            link = 'https://www.imdb.com/title/tt' + str('0' * size) + str(row_link)
            rec_movies += f'<a href={link}> {row.title}</a>' + '<br>'


    else:
        for row in recommended_movies.itertuples():
            list_genres = row.genres.split('|')
            if genre in list_genres:
                row_link = res[res['title'] == row.title]['imdbId'].values[0]
                size = 7 - len(str(row_link))
                link = 'https://www.imdb.com/title/tt' + str('0' * size) + str(row_link)
                rec_movies += f'<a href={link}> {row.title}</a>' + '<br>'

    return rec_movies


def main():
    app = QtWidgets.QApplication([])
    application = Recom()

    application.setFixedSize(560, 446)
    application.show()

    sys.exit(app.exec())


class Recom(QtWidgets.QMainWindow):
    def __init__(self):
        super(Recom, self).__init__()
        self.id = -1
        self.help = 0
        self.user2user_encoded = {}
        self.userencoded2user = {}
        self.movie2movie_encoded = {}
        self.movie_encoded2movie = {}
        self.num_users = 0
        self.num_movies = 0
        self.min_rating = 0
        self.max_rating = 0
        self.x_train = []
        self.x_val = []
        self.y_train = []
        self.y_val = []
        self.model = None
        self.history = None
        self.genre = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_UI()
        self.ui.movies_label_2.setOpenExternalLinks(True)

    def init_UI(self):
        self.setWindowIcon(QIcon('icon.png'))

        self.ui.pushButton.clicked.connect(self.get_params)
        self.ui.pushButton_2.clicked.connect(self.load_file)

    def get_params(self):
        id = self.ui.id_lineEdit.text()
        if id.isdigit():
            id = int(id)
        else:
            raise TypeError
            return

        genre = self.ui.comboBox.currentText()

        df = pd.read_csv('ratings.csv')

        if not any(df.userId == id):
            self.ui.movies_label_2.setText('- - - Такого пользователя не существует! - - -')
            return
        if self.help == 0:
            self.user2user_encoded, self.userencoded2user, self.movie2movie_encoded, self.movie_encoded2movie, self.num_users, self.num_movies, self.min_rating, self.max_rating = pre_processing(
                df)
            self.x_train, self.x_val, self.y_train, self.y_val = preparing_data(df, self.min_rating, self.max_rating)
            self.model, self.history = train_model(self.num_users, self.num_movies, self.x_train, self.x_val, self.y_train, self.y_val)
            self.help = 1
        # draw_plot(history)

        # Показать 10 лучших фильмов, рекомендуемых пользователю
        movie_df = pd.read_csv('movies.csv')
        rec_movies = find_movies(df, movie_df, self.model, self.movie2movie_encoded, self.user2user_encoded, self.movie_encoded2movie, id,
                                 genre)
        self.ui.movies_label_2.setText(rec_movies)

    def load_file(self):
        res = QFileDialog.getOpenFileName(self, 'Open File', 'D:/python/test2')

        df_ratings = pd.read_csv('ratings.csv')

        f = open(res[0], 'r')
        for line in f:
            line = line.replace('\n', '').split(',')
            help_df = df_ratings[df_ratings["userId"] == int(line[0])]
            if (help_df[help_df["movieId"] == int(line[1])].empty) == True:
                df_ratings.loc[len(df_ratings.index)] = line

        df_ratings['userId'] = df_ratings['userId'].astype('int64')
        df_ratings['movieId'] = df_ratings['movieId'].astype('int64')

        df_ratings.to_csv('ratings.csv', index=False)

        self.help = 0
        return res[0]


if __name__ == '__main__':
    main()
