from ...models.clustering.trainer import Clustering
from ...utils.utils_clustering import clustering_similarity_voc
from sklearn.cluster import KMeans
import numpy as np
from ...utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


class Similarity_voc(Clustering):
    name_clustering = 'Similarity_voc'
    dimension_embedding = "doc_embedding"

    def __init__(self, flags_parameters, embedding, name_model_full, column_text):
        Clustering.__init__(self, flags_parameters, embedding, name_model_full, column_text)
        self.vect_vocabulary_labels = {}

    def hyper_params(self):
        parameters = dict()

        parameters["vocabulary_labels"] = self.flags_parameters.vocabulary_labels
        self.p = parameters

        return parameters

    def get_doc_topic(self, x_preprocessed, fit=False):

        if self.vect_vocabulary_labels == {}:
            for k, list_words in self.p["vocabulary_labels"].items():
                voc_preprocessed = self.embedding.transform(list_words, None)
                if self.name_embedding not in ['tf-idf', 'tf']:
                    voc_preprocessed = np.array([v for v in voc_preprocessed if np.sum(v) != 0])
                    if voc_preprocessed.shape[0] == 0:
                        voc_preprocessed = np.array([[0]])
                self.vect_vocabulary_labels[k] = voc_preprocessed

        return clustering_similarity_voc(x_preprocessed, self.vect_vocabulary_labels)