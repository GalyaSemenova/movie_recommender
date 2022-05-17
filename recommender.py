# Мы встраиваем как пользователей, так и фильмы в 50-мерные векторы.
# Модель вычисляет оценку соответствия между встраиванием пользователя и фильма с помощью скалярного произведения
# и добавляет смещение для каждого фильма и для каждого пользователя.
# Оценка совпадения масштабируется до интервала [0, 1] с помощью сигмоиды (поскольку наши рейтинги нормализованы
# к этому диапазону).


from keras.applications.densenet import layers
from tensorflow import keras
import tensorflow as tf



class RecommenderNet(keras.Model):
    EMBEDDING_SIZE = 50

    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Добавить все компоненты (включая смещение)
        x = dot_user_movie + user_bias + movie_bias
        # Активация сигмовидной force приводит к тому, что рейтинг находится в диапазоне от 0 до 1
        return tf.nn.sigmoid(x)



