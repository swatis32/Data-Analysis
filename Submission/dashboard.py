import streamlit as st
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import os
import warnings
warnings.filterwarnings("ignore")

wd="C:/LivePerson Project/Submission/"
os.chdir(wd)

st.markdown("""
<style>
body {
    color: #e50000;
    background-color: #e5000;
}
</style>
    """, unsafe_allow_html=True)

st.title("Welcome to Netflix! Unlimited Movies, TV shows and more... ")
st.write ("""
### Top Voted on Netflix
""")


#netflix data
netflix_data=pd.read_csv("netflix_titles.csv")
netflix_data['year_added'] = pd.DatetimeIndex(netflix_data['date_added']).year


#imdb data
imdb_titles=pd.read_csv('IMDb movies.csv', usecols=['imdb_title_id','title','year','genre','writer'])
imdb_ratings=pd.read_csv('IMDb ratings.csv',usecols=['imdb_title_id','weighted_average_vote','us_voters_rating','non_us_voters_rating'])

#data merge
imdb_merge=imdb_titles.merge(imdb_ratings,left_on='imdb_title_id',right_on='imdb_title_id',how='inner')

#data splits
netflix_all=netflix_data.merge(imdb_merge,left_on='title',right_on='title',how='inner')
netflix_all['year_added'] = pd.DatetimeIndex(netflix_all['date_added']).year

netflix_all.drop_duplicates(subset=['title'], inplace=True)

#reset index to 0:len(netflix_all)
netflix_all.reset_index(drop=True, inplace=True)

netflix_movies=netflix_all[netflix_all['type']=='Movie']
netflix_shows=netflix_all[netflix_all['type']=='TV Show']

# Display top 10 things to watch

ranked_data=netflix_all.sort_values(by='weighted_average_vote', ascending=False)

st.dataframe(ranked_data[['title','type','duration','cast','director','genre','description']][0:10])

# Recommender based on genre, director, cast

netflix_all['director'] = netflix_all['director'].fillna(' ')
netflix_all['director'] = netflix_all['director'].astype(str)
netflix_all['cast'] = netflix_all['cast'].fillna(' ')
netflix_all['cast'] = netflix_all['cast'].astype(str)
netflix_all['genre'] = netflix_all['genre'].fillna(' ')
netflix_all['genre'] = netflix_all['genre'].astype(str)
netflix_all['writer'] = netflix_all['writer'].fillna(' ')
netflix_all['writer'] = netflix_all['writer'].astype(str)


netflix_all['recommender'] = netflix_all['genre'] + ' ' + ' ' + netflix_all['director'] + ' ' + netflix_all['cast']+ ' ' + netflix_all['writer']

cv = CountVectorizer()
count_mat = cv.fit_transform(netflix_all['recommender'])
cosine_sim = cosine_similarity(count_mat,count_mat)
#print(cosine_sim)

indices = pd.Series(netflix_all['title'])

def recommend_a_movie(name):
    movies=[]
    idx = indices[indices == name].index[0]
    sort_index = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    recommended_list= sort_index.iloc[1:5]
    for i in recommended_list.index:
        movies.append(indices[i])
    return movies

st.write ("""
### Don't see anything you like?
""")

movie_name=st.selectbox("Select a Movie/TV Show", indices )

st.write ("""
### Top recommendations based on your selection
""")

st.write(recommend_a_movie(movie_name))

