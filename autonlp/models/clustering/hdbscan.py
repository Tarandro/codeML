from ...models.clustering.trainer import Clustering
from ...utils.utils_clustering import clustering_sklearn_label_doc_topic
from sklearn.cluster import DBSCAN
#import hdbscan

import logging
from ...utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


class Hdbscan_sklearn(Clustering):
    name_clustering = 'Hdbscan'
    dimension_embedding = "doc_embedding"

    def __init__(self, flags_parameters, embedding, name_model_full, column_text):
        Clustering.__init__(self, flags_parameters, embedding, name_model_full, column_text)

    def hyper_params(self):
        parameters = dict()

        parameters["min_cluster_size"] = self.flags_parameters.min_cluster_size

        if 'acp' in self.name_model_full.lower():
            parameters["n_components"] = self.flags_parameters.acp_n_components
        elif 'umap' in self.name_model_full.lower():
            parameters["n_components"] = self.flags_parameters.umap_n_components
            parameters["n_neighbors"] = self.flags_parameters.umap_n_neighbors

        self.p = parameters

        return parameters

    def model_cluster(self):
        #hdb = hdbscan.HDBSCAN(min_cluster_size=self.p["min_cluster_size"], metric='euclidean', cluster_selection_method='eom')
        hdb = DBSCAN()
        return hdb

    def get_doc_topic(self, x_preprocessed, fit=False):
        return clustering_sklearn_label_doc_topic(self.pipeline)
