from ...models.clustering.trainer import Clustering
from ...utils.utils_clustering import clustering_sklearn_predict_doc_topic
from sklearn.cluster import KMeans
from ...utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


class Kmeans_sklearn(Clustering):
    name_clustering = 'KMeans'
    dimension_embedding = "doc_embedding"

    def __init__(self, flags_parameters, embedding, name_model_full, column_text):
        Clustering.__init__(self, flags_parameters, embedding, name_model_full, column_text)

    def hyper_params(self):
        parameters = dict()

        parameters["n_groups"] = self.flags_parameters.n_groups

        if 'acp' in self.name_model_full.lower():
            parameters["n_components"] = self.flags_parameters.acp_n_components
        elif 'umap' in self.name_model_full.lower():
            parameters["n_components"] = self.flags_parameters.umap_n_components
            parameters["n_neighbors"] = self.flags_parameters.umap_n_neighbors

        self.p = parameters

        return parameters

    def model_cluster(self):
        kmeans = KMeans(n_clusters=self.p["n_groups"], random_state=15)
        return kmeans

    def get_doc_topic(self, x_preprocessed, fit=False):
        return clustering_sklearn_predict_doc_topic(self.pipeline, x_preprocessed)