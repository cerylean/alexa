import json
from pandas.io.json import json_normalize

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg')

import pylab
pylab.close()

import nltk
from nltk import ngrams
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics

# def parse_json(json_file):
#     with open (json_file) as data_file:
#         data= json.load(data_file)
#         df = pd.io.json.json_normalize(data)
#     return df
#
# def parse_reviews(df,col):
#     new_df = pd.DataFrame(columns = list(df[col][0][0].keys()))
#     for i in range(df.shape[0]):
#         for review in df[col][i]:
#             new_df = new_df.append(review,ignore_index=True)
#     return new_df

def segment_stars(df,stars,stars1):
        df = df[(df['review_rating']==stars) | (df['review_rating']==stars1)]
        df = df.reset_index()
        return df

def clean_data(dataframe,col):
    punctuation = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    import_errors = ['_„Ž','_„ñ','_ã_','_„','Ã±','ñ']
    df2 = dataframe.copy()
    df2 = dataframe.apply(lambda x: x.astype(str).str.lower())
    for e in import_errors:
        df2[col] = df2[col].str.replace(e,'')
    for p in punctuation:
        df2[col] = df2[col].str.replace(p,'')
    new_array = df2[col].str.split()
    for i in range(len(new_array)):
        new_array[i] = [x for x in new_array[i] if not x in stop_words_adj]
        new_array[i] = [x.strip(' ') for x in new_array[i]]
        new_array[i] = ' '.join(new_array[i])
    return new_array

# def word_dictionary(array,pl):
#     word_dict = {}
#     for ix in range(len(array)):
#         for word in range(0,len(array[ix])-3):
#             if tuple(array[ix][word:word+pl]) in word_dict:
#             # if str(array[ix][word:word+pl]) in word_dict:
#                 # word_dict[str(array[ix][word:word+pl])] += 1
#                 word_dict[tuple(array[ix][word:word+pl])] += 1
#             else:
#                 # word_dict[str(array[ix][word:word+pl])] = 1
#                 word_dict[tuple(array[ix][word:word+pl])] = 1
#     return word_dict

def vectorisize(data,ngrams,max_features):
    vectorizer = TfidfVectorizer(ngram_range=ngrams, max_features=max_features)
    tfidf_model = vectorizer.fit_transform(data)
    return tfidf_model

def cluster_text(data,ngrams,clusters,max_features):
    vectorizer = TfidfVectorizer(ngram_range=ngrams,max_features=max_features)
    tfidf_model = vectorizer.fit_transform(data)

    kmeans = KMeans(n_clusters=clusters).fit(tfidf_model)
    centroids = kmeans.cluster_centers_
    pred_labels = kmeans.predict(tfidf_model)
    labels = kmeans.labels_
    names = []
    for cluster in centroids:
        sorted_cluster = sorted(cluster,reverse=True)
        top_fifteen = sorted_cluster[:15]
        indices = np.argsort(cluster)[::-1][:15]
        for idx in indices:
            names.append(vectorizer.get_feature_names()[idx])
    return names

def elbow_method(Kmax,tfidf_matrix):
    distortions = []
    K = range(1,Kmax)
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(tfidf_matrix)
        labels = kmeans.labels_
        distortions.append(kmeans.inertia_)
    return list(K),distortions

def plot_elbow(K_lst,distortions,file_name):
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig(file_name)
    plt.close()


# common_phrases = dict((k, v) for k, v in review_dict.items() if v >= 50)

df = pd.read_csv('amazon_reviews.csv', encoding='ISO-8859-1',usecols=[1,2,3,4,5,6])
# df['review_color'] = df['review_color'].map({'Color: Black Configuration: Echo Dot':'Black','Color: White Configuration: Echo Dot':'White','Color: Black Configuration: Echo Dot0':'Black'})
stop_words = stopwords.words('english')
stop_words_add = ['echo', 'generation', 'dots','alexa' ]
stop_words_remove =  ['more','not','against',"don't", "should've"]
stop_words_limit = stop_words[:145]
stop_words_adj = set([word for word in stop_words_limit if word not in stop_words_remove] +stop_words_add)


stars_5,stars_1, stars_5_and_1 = segment_stars(df,5,5), segment_stars(df,1,1), segment_stars(df,5,1)
# stars_5_and_1 = stars_5_and_1.sort_values(by='review_rating')
# stars_5_and_1 = stars_5_and_1[:2000]
# stars_5_and_1 = stars_5_and_1.reset_index()

# cleaned_all = clean_data(df,'review_text')
cleaned_5 = clean_data(stars_5,'review_text')
cleaned_1 = clean_data(stars_1,'review_text')
# cleaned_5_and_1 = clean_data(stars_5_and_1,'review_text')
# labelz = cluster_text(cleaned_5_and_1,(4,4),10)
#
# X_tfidf_5 = vectorisize(cleaned_5_and_1,(4,4),5000)
# K, distortions = elbow_method(15,X_tfidf_5)
# plot_elbow(K,distortions,'elbow_plot_5_and_1star.png')

# X_tfidf = vectorisize(cleaned_all,(4,4),5000)
# K, scores = elbow_method(20,X_tfidf)
# plot_elbow(K,scores,'elbow_plot_all_20.png')

#
# X_tfidf = vectorisize(cleaned_all,(4,4),5000)
# K, distortions = elbow_method(15,X_tfidf)

# count_vect = CountVectorizer(lowercase=False)
# X_train_counts = count_vect.fit_transform(cleaned)
# ngrams = (4,4)
# max_features = 2000

# tfidf = TfidfVectorizer(ngram_range=ngrams)
# def cluster_test(data, ngrams_lst):
#     for ng in ngrams_lst:
#         print (f"ngrams: {ng[0]}", cluster_text(data,10,ng,8000))
# def review_df(data,col_lst,cluster, ngram_lst,max_features):
#     new_df = pd.DataFrame(np.array([1,2,3,4,5,6,7,8,9,10]),index=None)
#     zip_ng_col = zip(ngram_lst,col_lst)
#     for ng,col in zip_ng_col:
#         df[col] = np.array(cluster_text(cleaned,cluster,ng,max_features))
#     return new_df
# # #
df_star = pd.DataFrame()
# df_1star = pd.DataFrame(np.array(list(range(30))),index=None)
# df_all   = pd.DataFrame(np.array(list(range(30))),index=None)
# #
def review_df(df, data,col,file,ngram, cluster=10,max_features=5000):
    # labels = cluster_text(data,ngram,cluster,max_features)
    df[col] = cluster_text(data,ngram,cluster,max_features)
    df.to_csv(file)


# cluster_text(data,ngrams,clusters,max_features):

# labelz = cluster_text(cleaned_5_and_1,(4,4),10,5000)

review_df(df_star,cleaned_5,'4-gram','5_star_wmax.csv',(4,4),1,5000)
review_df(df_star,cleaned_1,'4-gram','1_star_wmax.csv',(4,4),1,5000)

# # review_df(df_1star,cleaned_1,'2-gram',(2,2))
# # review_df(df_1star,cleaned_1,'3-gram',(3,3))
# # review_df(df_1star,cleaned_1,'4-gram',(4,4))
# label_1 = review_df(df_1star,cleaned_1,'5-gram','1_star.csv',(4,4))
# # review_df(df_1star,cleaned_1,'6-gram',(6,6))
#
# label_all = review_df(df_all,cleaned_all,'5-gram','all_star.csv',(4,4))
#
#
# # # df_5_star = review_df(cleaned,['2-gram','3-gram','4-gram','5-gram','6-gram'],10,[(2,2),(3,3),(4,4),(5,5),6,6)],8000)
# common_reviews = pd.ExcelWriter('common_amazon_reviews.xlsx',engine='openpyxl')
# df_5star.to_excel(common_reviews,'5_star_nostop_max1000')
# df_1star.to_excel(common_reviews,'1_star_nostop_max1000')
# common_reviews.save()

# X_train_tfidf = tfidf.fit_transform(cleaned)
# print(len(tfidf.vocabulary_))
# km = KMeans(n_clusters=4, random_state=1)
# km.fit(X_train_tfidf)
# y_kmeans = km.predict(X_train_tfidf)
# km.labels_

# idea: plot histogram of distribution
