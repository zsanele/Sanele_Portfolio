"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

#collab_model = open("resources/make_recommendation1.pkl","rb")

#other imports
from PIL import Image

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Getting Started","Recommender System","Solution Overview","EDA"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it! **PLEASE CHOOSE A DIFFERENT SET OF MOVIES**")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Getting Started":
        st.title('Movie Recommender Engine')
        st.info('Working With The App')
        st.image(r'resources/imgs/intro.jpg',use_column_width=True)
        st.header('**Using The App**')
        st.markdown(''' i. Navigate and select a Movie Recommender System Algorithm of your choice. ''')
        st.markdown(''' ii. Collaborative Filtering algorithm will recommend a movie to a user based on what other users with similar movie preferences and interests have watched and recommend that to them.''')
        st.markdown(''' iii. Content_Based Filtering algorithm will recommend a movie to a user based on movies that are similar to what they have watched before and then recommend those movies to them.''')
        st.markdown(''' iv. After selecting an algorithm, select 3 options of your favorite movies that you would like to get recommendations from.''')
        st.markdown(''' v. Click 'Recommend' and get a list of movie recommendations for you.''')
        
        st.header('**Illustrations**')
        st.image(r'resources/imgs/intro.gif',use_column_width=True)


    if page_selection == "Solution Overview":
        #st.write("Describe your winning approach on this page")

        st.title('Movie Recommender Engine')
        st.info('General Information') 
        st.image(r'resources/imgs/unnamed.jpg',use_column_width=True)
        st.header('**About Recommendation Systems **')
        st.markdown('''Recommendation systems are becoming increasingly important in today’s extremely busy world. People are always short on time with the myriad tasks they need to accomplish in the limited 24 hours. Recommender systems are socially and economically critical for ensuring that individuals can make appropriate choices surrounding the content they engage with on a daily basis. Therefore, the recommendation systems are important as they help them make the right choices.''')
        st.markdown('''The purpose of a recommendation system is to search for content that would be interesting to an individual. Moreover, it involves a number of factors to create personalised lists of useful and interesting content specific to each user.Another objective of the recommendation system is to achieve customer loyalty by providing relevant content and maximising the time spent by a user on your website or channel.''')
                    
        st.header(' **Why Movie Recommendation Systems **')
        st.markdown(''' i. They help the user find movies of their interest.''')
        st.markdown('''  ii. Helps the item provider to deliver their items to the right user.''')
        st.markdown(''' iii.    Showcase personalised content to each user.''')
        st.markdown(''' iv. Websites can improve user-engagement to the website.''')
        st.header(' **Movie Recommendation System Used**')
        st.image(r'resources/imgs/Collaborative_filtering_vbujt7.gif',use_column_width=True)
        st.markdown('''Movie recommendation Systems are driven by Machine Learning Algorithms. This Recommendation Engine uses Collaborative Filtering and Content-Based Filtering to predict unseen movie ratings gathered from thousands of users based on their historical preference.''')
        st.markdown('''When a movie website misses a recommendation system, it results in users browsing through a long list of movies, with no suggestions about what to watch. This, in turn, reduces the propensity of a user to engage with the website and use its services. This recommendation system will help users get recommendations of movies based on similar movies they once watched or based on what other users who have similar interest in movies as them have loved watching. ''')
        st.image(r'resources/imgs/unnamed2.jpg',use_column_width=True)

        st.header('**Collaborative Filtering**')
        st.markdown('''i.   User-based Filtering: these systems recommend products to a user that similar users have liked. These systems are widely used, and they try to predict the rating or preference that a user would give an item-based on past ratings and preferences of other users.''')
        st.markdown('''ii.  Item-based Filtering: These systems identify similar items based on how people have rated it in the past.''')
        st.header('**Content-based Filtering**')
        st.markdown('''This system uses item metadata, such as genre, director, description, actors, for movies, to make these recommendations. The general idea behind these recommender systems is that if a person likes a particular item, he or she will also like an item that is similar to it. And to recommend that, it will make use of the user's past item metadata.''')

    if page_selection == "EDA":
        st.title("Movies Graphs and Relationships")
        st.markdown("Using the raw movie data , we plotted some graphs/dashboards that show relationships between variables like the budget of a movie and it's rating , the rapid rise in budget of movies over the years, cast and directors of movies amongst others")
        st.markdown("**NB: Please view reports in full page to see them clearly and interact with them nicely**")
        #data
        #Movie data has movie ID movie name and genre(s)
        Movies = pd.read_csv('resources/data/movies.csv')
        #Rating data has movie Id and ratings
        Ratings = pd.read_csv('resources/data/ratings.csv')
        #IMDB data has movie ID, cast , directors and movie budget 
        imdb = pd.read_csv('resources/data/imdb_data.csv')


        if st.checkbox('Actors'):
                st.markdown('''For Actors data we look at ..''')
                st.markdown("i. The cast for the top 10 rated movies and also how many of the 10 movies did the cast repeat if they did")
                st.markdown("ii. We also look at the average number of cast members over the years and we observe the leading characters for the top 10 rated movies")
                st.markdown('''<iframe width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiZjcyNGY1NWYtM2JmNS00N2QxLWJlMWMtNDZlMzZkYWI5MzZmIiwidCI6ImRmZTE1OTcyLTUzNmUtNDY0OS05ZTZhLWFhMWIxZGZiOGRlZSJ9" frameborder="0" allowFullScreen="true"></iframe>''', unsafe_allow_html=True)
        elif st.checkbox('Movies'):
                st.markdown('''For the movies data we look at ..''')
                st.markdown("i. The top 10 rated movies titles and the sum of the ratings")
                st.markdown("ii. we also look at the number of users that have rated movies and the change in ratings over the years")
                st.markdown("iii. Lastly we look at how each genrer average in terms of rating")
                st.markdown('''<iframe width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiZjlhMTMzOGItNjdkZS00ZmJlLWJhNzQtMTE0ODQyMzljNDI2IiwidCI6ImRmZTE1OTcyLTUzNmUtNDY0OS05ZTZhLWFhMWIxZGZiOGRlZSJ9" frameborder="0" allowFullScreen="true"></iframe>''', unsafe_allow_html=True)
        elif st.checkbox('Genre'):
                st.markdown('''For the genrer data we look at ..''')
                st.markdown("i. we look again at the top 10 rated movies but this time we alos check the genrers they belong to")
                st.markdown("ii. we also look at the overall number of movies per genrer with drama leading by over 9000 movies")
                st.markdown('''<iframe width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiNzM5NWFiYjYtMTRkYS00NGJmLTkxNGUtZjA2MzdmYjUwY2FlIiwidCI6ImRmZTE1OTcyLTUzNmUtNDY0OS05ZTZhLWFhMWIxZGZiOGRlZSJ9" frameborder="0" allowFullScreen="true"></iframe>''', unsafe_allow_html=True)
        elif st.checkbox('Directors'):
                image = Image.open(r'resources/reports/directors.jpg')
                st.markdown("i. we look at the top 10 rated movie titles and this time we also look at the directors who directed them")
                st.markdown('ii. We also look at the change in number of directors over time')
                st.markdown("iii. lastly we look at the top 10 rated directors and the movies they directed")
                st.markdown('''<iframe width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiMzBmYmFlMTAtZmY5ZC00OTRmLWJhZTItZmFlYzhhZDZmZGQ5IiwidCI6ImRmZTE1OTcyLTUzNmUtNDY0OS05ZTZhLWFhMWIxZGZiOGRlZSJ9" frameborder="0" allowFullScreen="true"></iframe>''',unsafe_allow_html=True)
        elif st.checkbox('Budget'):
                st.markdown('''For the Budget data we look at ..''')
                st.markdown("i. The top 10 movie titles and their budgets we also observe the sum ratings of the movies")
                st.markdown("ii. We then look at the change in budgets over the years")
                st.markdown("iii. lastly we look at the overall budget for different genrers, with comedy having over $10 Billion")
                st.markdown('''<iframe width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiMjY0OTUzYWMtNzY4Mi00YmM3LWE2NDMtZGRmNmI4MGIxZjUwIiwidCI6ImRmZTE1OTcyLTUzNmUtNDY0OS05ZTZhLWFhMWIxZGZiOGRlZSJ9" frameborder="0" allowFullScreen="true"></iframe>''',unsafe_allow_html=True)
        elif st.checkbox('Overall'):
                #st.markdown('''Overall Movies Data''')
                st.markdown("""<iframe width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiMDVlZDMxZGEtY2YwYy00Mzk1LWEyNGItYTAwMjJmOTk3NDRmIiwidCI6ImRmZTE1OTcyLTUzNmUtNDY0OS05ZTZhLWFhMWIxZGZiOGRlZSJ9" frameborder="0" allowFullScreen="true"></iframe>""",unsafe_allow_html=True)

               


    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
