import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn import metrics

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

df = pd.read_csv('amazon_reviews.csv', encoding='ISO-8859-1',usecols=[1,2,3,4,5,6])
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

df_star = pd.DataFrame()

def review_df(df, data,col,file,ngram, cluster=10,max_features=5000):
    # labels = cluster_text(data,ngram,cluster,max_features)
    df[col] = cluster_text(data,ngram,cluster,max_features)
    df.to_csv(file)

# labelz = cluster_text(cleaned_5_and_1,(4,4),10,5000)

review_df(df_star,cleaned_5,'4-gram','5_star_wmax.csv',(4,4),1,5000)
review_df(df_star,cleaned_1,'4-gram','1_star_wmax.csv',(4,4),1,5000)

# df_5star.to_excel(common_reviews,'5_star_nostop_max1000')
# df_1star.to_excel(common_reviews,'1_star_nostop_max1000')
# common_reviews.save()
