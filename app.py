from flask import Flask, render_template, request
import pickle
import pandas as pd
import requests

def getPoster(id):
    try:
        response = requests.get('https://api.jikan.moe/v4/anime/{}/full'.format(id))
        data = response.json()
    except KeyError:
        return data['data']['images']['jpg']['image_url']

def recommend(prompt):
    index = anime_list[anime_list['Name'] == prompt].index[0]
    cosine_angles = similarity[index]
    rec = sorted(list(enumerate(cosine_angles)), reverse=True, key=lambda x: x[1])[1:11]

    l = []
    p = []
    for i in rec:
        print(i)
        l.append(anime_list.iloc[i[0]].Name)
        p.append(getPoster(i[0]))
    return l, p

app = Flask(__name__)
anime_list = pickle.load(open('anime.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    name = request.form.get('name')

    recommendations, posters = recommend(name)

    temp = "Recommendation for {} are\n".format(name)

    return render_template("index.html", prediction_text = "{}\n{}".format(temp, recommendations))

if __name__ == '__main__':
    app.run(debug = True)