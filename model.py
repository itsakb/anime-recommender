import pandas as pd
import numpy as np
import pickle

#read anime.csv file
anime = pd.read_csv('anime.csv')

#remove irrelevant columns for the model
#'MAL_ID','Name', 'Score', 'Genres', 'Type', 'Episodes', 'Studios', 'Ranked'
anime = anime[['MAL_ID','Name', 'Genres', 'Type', 'Episodes', 'Studios', 'Source']]

#remove all fields containing unknown
anime = anime[anime['Episodes'] != 'Unknown']
anime = anime[anime['Studios'] != 'Unknown']

#read synopsis.csv file(having overview/summary of each anime)
synopsis = pd.read_csv('anime_with_synopsis.csv')

#reduce synopsis to relevant columns only
synopsis = synopsis[['MAL_ID', 'sypnopsis']]

#merge both the table on ID
anime = anime.merge(synopsis, on = 'MAL_ID')

#convert list to string in order to convert it to tags later
anime['Genres'] = anime['Genres'].apply(lambda x : x.replace(", ", " "))

#create Tags using multiple columns
anime['tags'] = anime['sypnopsis']+" "+ anime['Genres']+" "+anime['Type']+" "+ anime['Studios']+" "+anime['Source']

#remove all columns used for tags
anime = anime[['MAL_ID', 'Name', 'Episodes', 'tags']]

#drop any null values
anime = anime.dropna()

#convert all tags to lowercase
anime['tags'] = anime['tags'].apply(lambda x : x.lower())

#use nltk to Stem tags
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(obj):
    l = []
    for i in obj.split():
        l.append(ps.stem(i))
    return " ".join(l)
anime['tags'] = anime['tags'].apply(stem)

#use sklearn to create a CountVectorizer object on top 6000 words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=6000, stop_words='english')

#convert the tags to vectors
vectors = cv.fit_transform(anime['tags']).toarray()

#use cosine_similarity to give angle to each vectorised tags
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

#recommend
def recommend(prompt):
    index = anime[anime['Name'] == prompt].index[0]
    cosine_angles = similarity[index]
    rec = sorted(list(enumerate(cosine_angles)), reverse=True, key=lambda x: x[1])[1:6]

    for i in rec:
        print(anime.iloc[i[0]].Name)

recommend('Dragon Ball')

pickle.dump(anime, open('anime.pkl', 'wb'))