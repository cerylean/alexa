import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Capstone_1 import vectorize_tfidf
from sklearn.cluster import KMeans

import matplotlib as mpl
mpl.rcParams.update({
    'font.size'           : 20.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'medium',
    'ytick.labelsize'     : 'medium',
    'legend.fontsize'     : 'large',
    'legend.loc'          : 'upper right'
})

def plot_histogram(df,col_name):
    '''
    INPUT:  DataFrame
            Column name (string)

    OUTPUT: Histogram figure (.png)

    '''
    plt.hist(df[col_name],bins=5)
    plt.title('Distribution of Star Ratings', fontweight="bold")
    plt.xticks([1,2,3,4,5])
    plt.xlabel('Stars')
    plt.tight_layout()
    plt.savefig('images/histogram_of_stars.png')
    plt.close()

def run_elbow_method(clean_data,ngrams,max_features,Kmax):
    '''
    Loops through cluster values (K) and return corresponding distortion sum

    INPUT:  Cleaned series (series)
            # of ngrams (tuple of two integers)
            # of max_features (integer)
            Max # of K (integer)

    OUTPUT: K values (list)
            Distortions (list)

    '''
    tfidf_matrix, feature_names = vectorize_tfidf(clean_data,ngrams,max_features)
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

    INPUT:  K values (list)
            Distortions (list)

    OUTPUT: Line graph figure (.png)

    '''
    plt.plot(K_lst, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k (5-star Reviews)')
    plt.xticks(np.arange(min(K_lst), max(K_lst)+1, 2.0))
    plt.savefig(file_name)
    plt.close()

df = pd.read_csv('data/amazon_reviews.csv', encoding='ISO-8859-1',usecols=[1,2,3,4,5,6])
plot_histogram(df,'review_rating')
