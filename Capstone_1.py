import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import pdb

def segment_stars(df,stars,stars1):
        df = df[(df['review_rating']==stars) | (df['review_rating']==stars1)]
        df = df.reset_index()
        return df

def clean_data(dataframe,col):
    punctuation = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    import_errors = ['_„Ž','_„ñ','_ã_','_„','Ã±','ñ']
    df2 = dataframe.copy()
    for e in import_errors:
        df2[col] = df2[col].str.replace(e,'')
    for p in punctuation:
        df2[col] = df2[col].str.replace(p,'')
    return df2[col]

def cluster_text(clean_data,ngrams,clusters,max_features,num_returned):
    vectorizer = TfidfVectorizer(ngram_range=ngrams,max_features=max_features,stop_words=stopwords)
    tfidf_model = vectorizer.fit_transform(clean_data)

    kmeans = KMeans(n_clusters=clusters).fit(tfidf_model)
    centroids = kmeans.cluster_centers_
    names = []
    for cluster in centroids:
        sorted_cluster = sorted(cluster,reverse=True)
        top_rows = sorted_cluster[:num_returned]
        indices = np.argsort(cluster)[::-1][:num_returned]
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

def define_stopwords(add_to_list,remove_from_list,limit=None):
    stop_words = stopwords.words('english')
    stop_words_limit = stop_words[:limit]
    stop_words_adj = set([word for word in stop_words_limit if word not in remove_from_list] + add_to_list)
    return stop_words_adj

def common_review_sentiments(clean_data_list,ngram,cluster=1,max_features=5000,sentiments_returned=15):
    sentiment_list = []
    col_name = []
    # pdb.set_trace()
    for lst in clean_data_list:
        sentiment_list.append(cluster_text(lst,ngram,cluster,max_features,sentiments_returned))
        var_name = [name for name in globals() if globals()[name] is lst]
        col_name.append(var_name[0])
    df = pd.DataFrame(sentiment_list,col_name).T
    df.to_csv('common_review_sentiments.csv')
    print(df)

df = pd.read_csv('amazon_reviews.csv', encoding='ISO-8859-1',usecols=[1,2,3,4,5,6])
stopwords = define_stopwords(['echo', 'generation','dot', 'dots','alexa' ],['more','not','against',"don't", "should've"],145)
stars_5,stars_1 = segment_stars(df,5,5), segment_stars(df,1,1)
cleaned_5, cleaned_1  = clean_data(stars_5,'review_text'), clean_data(stars_1,'review_text')
common_review_sentiments([cleaned_5,cleaned_1],(4,4),1,5000,15)
# df_star = pd.DataFrame()#
# review_df(df_star,cleaned_5,'4-gram','5_star_wmax.csv',(4,4),1,5000)
# review_df(df_star,cleaned_1,'4-gram','1_star_wmax.csv',(4,4),1,5000)
