import enum
from operator import itemgetter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

df = pd.read_csv('data\dataset.csv')
title_list = df['Title'].tolist()
description_list = df['Description'].tolist()
keywords_list = df['Keywords'].tolist()

def get_tfidf(idx, num = 5):
    description = description_list[idx]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(description_list)
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    res_indices = [i[0] for i in sim_scores]
    return res_indices

def get_kw(idx, num = 5):
    keywords = keywords_list[idx]

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df['Keywords'])
    cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

    sim_scores = list(enumerate(cosine_sim2[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    res_indices = [i[0] for i in sim_scores]
    return res_indices

print(get_tfidf(342,3))
print(get_kw(342,3))