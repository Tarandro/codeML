from ...models.clustering.trainer import Clustering
from ...utils.utils_clustering import matrice_doc_topic
from sklearn.decomposition import LatentDirichletAllocation
import copy


class LDA_sklearn(Clustering):
    name_clustering = 'LDA'
    dimension_embedding = "doc_embedding"

    def __init__(self, flags_parameters, embedding, name_model_full, column_text):
        flags_parameters_ = copy.deepcopy(flags_parameters)
        flags_parameters_.tf_wde_vector_size = 10e10
        flags_parameters_.tfidf_wde_vector_size = 10e10
        Clustering.__init__(self, flags_parameters_, embedding, name_model_full, column_text)

    def hyper_params(self):
        parameters = dict()

        parameters["n_groups"] = self.flags_parameters.n_groups
        parameters["max_iter_lda"] = self.flags_parameters.max_iter_lda

        self.p = parameters

        return parameters

    def model_cluster(self):
        lda_model = LatentDirichletAllocation(n_components=self.p["n_groups"], max_iter=self.p["max_iter_lda"],
                                              learning_method='online', learning_offset=10., random_state=15)
        return lda_model

    def get_doc_topic(self, x_preprocessed, fit=False):
        return matrice_doc_topic(self.pipeline, x_preprocessed, fit)
