from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np

from yellowbrick.cluster import SilhouetteVisualizer, KElbowVisualizer

from bokeh.models.widgets import TableColumn

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')


def get_elbow_plot(X):

    output_text = ""
    try:
        model = KMeans(random_state=40,)
        elbow_score = KElbowVisualizer(model, k=(1, 30))
        elbow_score.fit(X)
        elbow_value = elbow_score.elbow_value_
        model = KMeans(elbow_value, random_state=42)
        silhoutte_score = SilhouetteVisualizer(model, colors='yellowbrick')
        silhoutte_score.fit(X)

        output_text = """The optimal number of clusters is """ + \
                      str(silhoutte_score.n_clusters_) + """ and the silhouette score is """ + \
                      str(np.round(silhoutte_score.silhouette_score_, 2))
    except ValueError as e:
        print(e)

    return output_text


def get_tsne(df, c_no, mapper):

    clust_norm_data_df = pd.DataFrame(df)
    source_clust_data = dict()

    try:
        kmeans = KMeans(n_clusters=c_no, random_state=40).fit(clust_norm_data_df)
        clust_norm_data_df['Cluster'] = kmeans.predict(df)
        tsne = TSNE(n_components=2, perplexity=40, n_iter=300, random_state=100)
        tsne_results = tsne.fit_transform(clust_norm_data_df.iloc[:, :-1])

        clust_norm_data_df['tsne-2d-one'] = tsne_results[:, 0]
        clust_norm_data_df['tsne-2d-two'] = tsne_results[:, 1]
        mapper['transform'].low = min(clust_norm_data_df['Cluster'].values)
        mapper['transform'].high = max(clust_norm_data_df['Cluster'].values)

        source_clust_data = dict(x=clust_norm_data_df['tsne-2d-one'], y=clust_norm_data_df['tsne-2d-two'],
                                 cluster=clust_norm_data_df['Cluster'])
    except ValueError as e:
        print (e)

    return source_clust_data


def clustering_data(clust_df, active_features, active_norm, active_clust_no,
                    clustering_data_source, mapper, clust_scat):

    if 'ALL' in active_features:
        clust_df = clust_df
    else:
        clust_df = clust_df.loc[:, active_features]

    clust_df = pd.get_dummies(clust_df, drop_first=True)
    if active_norm == 1:
        clust_norm_data = pd.DataFrame(StandardScaler().fit_transform(clust_df.values))
    else:
        clust_norm_data = clust_df

    if clust_norm_data.shape[1] == 1:
        clust_norm_data = clust_norm_data.values.reshape(-1, 1)

    output_text = get_elbow_plot(clust_norm_data)
    clust_scat.title.text = output_text

    source_clust_data = get_tsne(clust_norm_data, active_clust_no, mapper)
    return source_clust_data
