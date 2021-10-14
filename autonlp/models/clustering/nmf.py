from ...models.clustering.trainer import Clustering
from ...utils.utils_clustering import matrice_doc_topic
from sklearn.decomposition import NMF
import copy


class NMF_sklearn(Clustering):
    name_clustering = 'NMF'
    dimension_embedding = "doc_embedding"

    def __init__(self, flags_parameters, embedding, name_model_full, column_text):
        flags_parameters_ = copy.deepcopy(flags_parameters)
        flags_parameters_.tf_wde_vector_size = 10e10
        flags_parameters_.tfidf_wde_vector_size = 10e10
        Clustering.__init__(self, flags_parameters_, embedding, name_model_full, column_text)

    def hyper_params(self):
        parameters = dict()

        parameters["n_groups"] = self.flags_parameters.n_groups
        parameters["alpha_nmf"] = self.flags_parameters.alpha_nmf
        parameters["l1_ratio"] = self.flags_parameters.l1_ratio

        self.p = parameters

        return parameters

    def model_cluster(self):
        if "nmf_frobenius" in self.name_model_full.lower():
            nmf_model = NMF(n_components=self.p["n_groups"], random_state=15, alpha=self.p["alpha_nmf"],
                            l1_ratio=self.p["l1_ratio"])
        else:
            nmf_model = NMF(n_components=self.p["n_groups"], random_state=1, beta_loss='kullback-leibler', solver='mu',
                            max_iter=1000, alpha=self.p["alpha_nmf"], l1_ratio=self.p["l1_ratio"])
        return nmf_model

    def get_doc_topic(self, x_preprocessed, fit=False):
        return matrice_doc_topic(self.pipeline, x_preprocessed, fit)