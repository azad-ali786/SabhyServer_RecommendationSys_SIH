import enum
from operator import itemgetter
from unittest import result
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# df = pd.read_csv('data\dataset.csv')
# title_list = df['Title'].tolist()
# description_list = df['Description'].tolist()
# keywords_list = df['Keywords'].tolist()

f = open("data/dataset.json")
json_file = json.load(f)

def get_data(json_file):
    desc_list = []
    kw_list = []
    id_list = []
    int_list = []
    for i in range(len(json_file)):
        id = json_file[i]['_id']
        desc = json_file[i]['desc']
        kw = ' '.join(json_file[i]['keyWords'])
        interests = ' '.join(json_file[i]['interests'])
        desc_list.append(desc)
        id_list.append(id)
        kw_list.append(kw)
        int_list.append(interests)
    return kw_list, desc_list, id_list, int_list

kw_list, desc_list, id_list, int_list = get_data(json_file)     # Add this in flask function too

def get_tfidf(_id, num = 5):
    idx = id_list.index(_id)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(desc_list)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    res_indices = [i[0] for i in sim_scores]
    res_indices = list((itemgetter(*res_indices)(id_list)))
    return res_indices

def get_kw(_id, num = 5):
    idx = id_list.index(_id)
    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(kw_list)
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    sim_scores = list(enumerate(cosine_sim2[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    res_indices = [i[0] for i in sim_scores]
    res_indices = list((itemgetter(*res_indices)(id_list)))
    return res_indices

def get_interests(interests):
    results = ([i for i, e in enumerate(int_list) if e in interests])
    res_indices = list((itemgetter(*results)(id_list)))
    return res_indices

interests = ['Technology']

# print(get_tfidf('95skg5vx6m',5))
# print(get_kw('95skg5vx6m',5))
# print(get_interests(interests))