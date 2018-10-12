import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from tabulate import tabulate


def segment_stars(df,col_name,param1,param2):
    '''
    Create a new dataframe filtered by up to two chosen parameters

    INPUT:  DataFrame (df)
            Column containing parameters (str)
            Filter parameter 1 (str or int)
            Filter parameter 2 (str or int)

    OUTPUT: DataFrame

    '''
    df = df[(df[col_name]==param1) | (df[col_name]==param2)]
    df = df.reset_index()
    return df

def clean_data(dataframe,col):
    '''
    Removes punctuation and import errors from dataframe

    INPUT:  DataFrame (df)
            Column name to clean (string)

    OUTPUT: Cleaned DataFrame Column (series)

    '''
    punctuation = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'
    import_errors = ['_„Ž','_„ñ','_ã_','_„','Ã±','ñ','ð']
    df2 = dataframe.copy()
    for e in import_errors:
        df2[col] = df2[col].str.replace(e,'')
    for p in punctuation:
        df2[col] = df2[col].str.replace(p,'')
    return df2[col]

def define_stopwords(add_to_list,remove_from_list,limit=None):
    '''
    Defines the stopwords to be removed from dataframe using the NLTK stopwords
    as a base list

    INPUT:  Words to add (list)
            Words to remove (list)
            Limit index of nltk stopwords list (integer)

    OUTPUT: Adjusted stopwords (list)

    '''
    stop_words = stopwords.words('english')
    stop_words_limit = stop_words[:limit]
    stop_words_adj = set([word for word in stop_words_limit if word not in remove_from_list] + add_to_list)
    return stop_words_adj

def vectorize_tfidf(clean_data,ngrams,max_features):
    vectorizer = TfidfVectorizer(ngram_range=ngrams,max_features=max_features,stop_words=stopwords)
    tfidf_matrix = vectorizer.fit_transform(clean_data)
    feature_names = vectorizer.get_feature_names()
    return tfidf_matrix, feature_names

def vectorize_and_cluster(clean_data,ngrams,clusters,max_features,num_returned):
    '''
    Transforms DataFrame Series into tfidf matrix, clusters the matrix

    INPUT:  Cleaned series (series)
            # of ngrams (tuple of two integers)
            # of clusters (integer)
            # of max_features (integer)
            # of ranked feature_names (integer)

    OUTPUT: Top-ranked feature_names (list)

    '''

    tfidf_matrix, feature_names = vectorize_tfidf(clean_data,ngrams,max_features)
    kmeans = KMeans(n_clusters=clusters).fit(tfidf_matrix)
    centroids = kmeans.cluster_centers_
    names = []
    for cluster in centroids:
        sorted_cluster = sorted(cluster,reverse=True)
        top_rows = sorted_cluster[:num_returned]
        indices = np.argsort(cluster)[::-1][:num_returned]
        for idx in indices:
            names.append(feature_names[idx])
    return names




def generate_ranked_sentiments(df_list,col_name,ngram,cluster=1,max_features=5000,sentiments_returned=15):
    '''
    Cleans and vectorizes data and returns a dataframe of top sentiments

    INPUT:  List of dataframes (list)
            Column name to find sentiments from (string)
            Ngram (tuple of two integers)

    OUTPUT: DataFrame of top sentiments

    '''
    clean_data_list, sentiment_list = [],[]
    col_names = []
    for df in df_list:
        clean_data_list.append(clean_data(df,col_name))
    # pdb.set_trace()
    for lst in clean_data_list:
        sentiment_list.append(vectorize_and_cluster(lst,ngram,cluster,max_features,sentiments_returned))
        df_name = [name for name in globals() if globals()[name] is lst]
        # col_names.append(df_name[0])
    df = pd.DataFrame(sentiment_list,['5-gram_5_star_review','5-gram_1_star_review']).T
    # file_name = str(ngram[0]) + 'word_sentiments_from_' + col_name + '.csv'
    # df.to_csv(file_name)
    return (df)

def to_markdown(df):
    '''
    Returns a markdown, rounded representation of a dataframe
    '''
    print(tabulate(df, headers='keys', tablefmt='pipe'))

if __name__ == '__main__':

    df = pd.read_csv('data/amazon_reviews.csv', encoding='ISO-8859-1',usecols=[1,2,3,4,5,6])
    stopwords = define_stopwords(['echo','room', 'generation','dot', 'dots','alexa' ],['more','not','against',"don't", "should've"],145)
    stars_5,stars_1 = segment_stars(df,'review_rating',5,5), segment_stars(df,'review_rating',1,1)
    to_markdown(generate_ranked_sentiments([stars_5,stars_1],'review_text',(5,5)))
