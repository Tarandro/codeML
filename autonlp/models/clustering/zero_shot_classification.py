from ...models.clustering.trainer import Clustering
from ...utils.utils_clustering import clustering_similarity_voc
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
from ...utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


class Zero_shot_classification(Clustering):
    name_clustering = 'Zero_shot_classification'
    dimension_embedding = "doc_embedding"

    def __init__(self, flags_parameters, embedding, name_model_full, column_text):
        Clustering.__init__(self, flags_parameters, embedding, name_model_full, column_text)
        self.batch_size = self.flags_parameters.batch_size

    def hyper_params(self):
        parameters = dict()

        parameters["vocabulary_labels"] = self.flags_parameters.vocabulary_labels
        parameters["pattern_zero_shot"] = self.flags_parameters.pattern_zero_shot
        self.p = parameters

        return parameters

    def get_doc_topic(self, x_preprocessed, fit=False):

        sequences = x_preprocessed
        model_zero_shot = self.embedding.embedding.model_zero_shot()

        labels_for_classifier = []
        dict_map_labels = {}
        dict_score = {}
        for label, label_voc in self.p["vocabulary_labels"].items():
            dict_score[label] = []
            if isinstance(label_voc, list):
                labels_for_classifier += label_voc
                for it in label_voc:
                    dict_map_labels[it] = label
            else:
                labels_for_classifier.append(label_voc)
                dict_map_labels[label_voc] = label

        doc_topic = []
        size_batch = self.batch_size
        n_batch = len(sequences) // size_batch
        for i in tqdm(range(n_batch + 1)):

            if len(sequences[i * size_batch: (i + 1) * size_batch]) == 0:
                continue

            preds = model_zero_shot(sequences[i * size_batch: (i + 1) * size_batch], labels_for_classifier,
                                    hypothesis_template=self.p["pattern_zero_shot"])

            for p in preds:
                if isinstance(p, list):
                    labels = p[0]['labels']
                    scores = p[0]['scores']
                else:
                    labels = p['labels']
                    scores = p['scores']
                score_by_label = {label: [] for label in dict_score.keys()}
                for label_voc, score in zip(labels, scores):
                    score_by_label[dict_map_labels[label_voc]].append(score)
                best_score = 0
                best_label = None
                for label in dict_score.keys():
                    score = np.mean(score_by_label[label])
                    dict_score[label].append(score)
                    if score > best_score:
                        best_score = score
                        best_label = label
                doc_topic.append(best_label)

        return doc_topic, dict_score