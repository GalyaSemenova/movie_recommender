# movie_recommender
 The movie recommendation system based on collaborative filtering using the MovieLens dataset

The basis is collaborative filtering using the MovieLens dataset to recommend movies to users

The steps in the model are as follows:
--------------------------------------

1. Match the user id with the "user vector" using the implementation matrix
2. Match the movie id to the "movie vector" using the embed matrix
3. Calculate the scalar product between the user vector and the movie vector to get a match score
between user and movie (predicted rating)
4. Train attachments using gradient descent using all known user-movie pairs.

The principle of operation
--------------------------

There are two functions with two variables x and theta. we need to find them at the same time. you can take turns selecting parameters, but
there is a function that optimizes finding both parameters at once. The formula uses the error square, which is
calculated as 
1) the sum of all user ratings for each movie 
2) the sum of all movie ratings for each of the user
We need to minimize x with respect to theta and theta with respect to X. That's what we decide.

Installation
------------

1. Download the application to your desktop
2. Run the application in the PyCharm IDE
3. Next, install all the necessary libraries that the system will offer
4. After that, click the "Run" button, which will launch our application

Licensing
---------

Please see the file called LICENSE.

