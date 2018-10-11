import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import pdb


def segment_stars(df,col_name,stars,stars1):
    '''
    Create a new dataframe filtered by parameter

    INPUT: DataFrame, Column to filter by, filter parameter 1, filter parameter 2
    OUTPUT: DataFrame

    '''
    df = df[(df['review_rating']==stars) | (df['review_rating']==stars1)]
    df = df.reset_index()
    return df

def clean_data(dataframe,col):
    '''
    Removes punctuation and import errors from dataframe

    INPUT: DataFrame, Column name to clean (as string)
    OUTPUT: Cleaned DataFrame Series

    '''
    punctuation = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    import_errors = ['_„Ž','_„ñ','_ã_','_„','Ã±','ñ']
    df2 = dataframe.copy()
    for e in import_errors:
        df2[col] = df2[col].str.replace(e,'')
    for p in punctuation:
        df2[col] = df2[col].str.replace(p,'')
    return df2[col]

def vectorize_and_cluster(clean_data,ngrams,clusters,max_features,num_returned):
    '''
    Transforms DataFrame Series into tfidf matrix, clusters the matrix

    INPUT: Series, # of ngrams, # of cluster, # of max_features, # feature_names
    OUTPUT: top-ranked feature_names

    '''
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
    '''
    Loops through various cluster values and returns corresponding distortion sum

    INPUT: Max value of clusters (K) to loop through, tfidf matrix
    OUTPUT: lists of K values, list of distortions

    '''
    distortions = []
    K = range(1,Kmax)
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(tfidf_matrix)
        labels = kmeans.labels_
        distortions.append(kmeans.inertia_)
    return list(K),distortions

def plot_elbow(K_lst,distortions,file_name):
    '''
    Plots results from elbow_method()

    INPUT: lists of K values, list of distortions
    OUTPUT: saved line graph

    '''
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig(file_name)
    plt.close()

def define_stopwords(add_to_list,remove_from_list,limit=None):
    '''
    Defines the stopwords to be removed from dataframe

    INPUT: list, list, limit of nltk stopwords list
    OUTPUT: list of final stopwords

    '''
    stop_words = stopwords.words('english')
    stop_words_limit = stop_words[:limit]
    stop_words_adj = set([word for word in stop_words_limit if word not in remove_from_list] + add_to_list)
    return stop_words_adj

def common_review_sentiments(df_list,col_name,ngram,cluster=1,max_features=5000,sentiments_returned=15):
    '''
    Cleans and vectorizes data and returns a dataframe of top sentiments

    INPUT: list of dataframes, column name, ngram tuple)
    OUTPUT: DataFrame of top sentiments

    '''
    clean_data_list, sentiment_list = [],[]
    col_names = []
    for df in df_list:
        clean_data_list.append(clean_data(df,col_name))
    for lst in clean_data_list:
        sentiment_list.append(vectorize_and_cluster(lst,ngram,cluster,max_features,sentiments_returned))
        df_name = [name for name in globals() if globals()[name] is lst]
        # col_names.append(df_name[0])
    df = pd.DataFrame(sentiment_list,['5-star','1-star']).T
    df.to_csv('common_review_sentiments.csv')
    print(df)

if __name__ == '__main__':

    df = pd.read_csv('data/amazon_reviews.csv', encoding='ISO-8859-1',usecols=[1,2,3,4,5,6])
    stopwords = define_stopwords(['echo', 'generation','dot', 'dots','alexa' ],['more','not','against',"don't", "should've"],145)
    stars_5,stars_1 = segment_stars(df,'review_rating',5,5), segment_stars(df,'review_rating',1,1)
    common_review_sentiments([stars_5,stars_1],'review_text',(4,4))
