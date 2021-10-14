from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from joblib import dump, load
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


from ...models.embeddings.trainer import Embedding
from ...utils.metrics import calcul_metric_classification
from ...utils.utils_clustering import clustering_similarity_voc, df_word_cluster, topic_terms_cluster
from ...features.cleaner import reduce_text_data

import logging
from ...utils.logging import get_logger

logger = get_logger(__name__)


class Clustering:
    """ Get Clustering """

    def __init__(self, flags_parameters, embedding, name_model_full, column_text):
        """
        Args:
            flags_parameters : Instance of Flags class object
            embedding (:class: Base_Embedding) : A children class from Base_Embedding
            name_model_full (str) : full name of model (embedding+classifier+tag)
            column_text (int) : column number with texts

        From flags_parameters:
            objective (str) : 'binary' or 'multi-class' or 'regression'
            average_scoring (str) : 'micro', 'macro' or 'weighted'
            seed (int)
            apply_mlflow (Boolean) save model in self.path_mlflow (str) directory
            experiment_name (str) name of the experiment, only if MLflow is activated
            apply_logs (Boolean) use manual logs
            apply_app (Boolean) if you want to use a model from model_deployment directory
            outdir (str) output directory
        """
        self.flags_parameters = flags_parameters

        # For clustering, need only a vector per document:
        dimension_embedding = "doc_embedding"
        self.embedding = Embedding(flags_parameters, embedding, dimension_embedding, column_text)
        self.dimension_embedding = dimension_embedding
        self.name_embedding = self.embedding.embedding.name_model

        self.name_model_full = name_model_full
        self.outdir = flags_parameters.outdir

        self.column_text = column_text
        self.average_scoring = flags_parameters.average_scoring
        self.apply_mlflow = flags_parameters.apply_mlflow
        self.path_mlflow = "../mlruns"
        self.experiment_name = flags_parameters.experiment_name
        self.apply_logs = flags_parameters.apply_logs
        self.apply_app = flags_parameters.apply_app
        self.seed = flags_parameters.seed

        self.map_cluster = None

        if self.apply_mlflow:
            import mlflow
            current_experiment = dict(mlflow.get_experiment_by_name(self.experiment_name))
            self.experiment_id = current_experiment['experiment_id']

    def save_params(self, outdir_model):
        """ Save all params as a json file needed to reuse the model

        Args:
            outdir_model (str)
        """
        params_all = dict()

        p_model = self.p.copy()

        params_all['p_model'] = p_model
        params_all['name_clustering'] = self.name_clustering
        params_all['language_text'] = self.flags_parameters.language_text
        params_all['map_cluster'] = {str(k): str(v) for k, v in self.map_cluster.items()}

        params_embedding = self.embedding.save_params(outdir_model)
        params_all.update(params_embedding)

        self.params_all = {self.name_model_full: params_all}

        if self.apply_logs:
            with open(os.path.join(outdir_model, "parameters.json"), "w") as outfile:
                json.dump(self.params_all, outfile)

    def load_params(self, params_all, outdir):
        """ Initialize all params from params_all

        Args:
            params_all (dict)
            outdir (str)
        """
        self.embedding.load_params(params_all, outdir)

        p_model = params_all['p_model']
        self.p = p_model
        self.map_cluster = params_all['map_cluster']

    def model_cluster(self):
        """ Abstract method.

            Initialize model architecture according to clustering model
        Return:
            model (sklearn Pipeline)
        """
        pass

    def model(self, data):
        """ Abstract method.

            Initialize model architecture according to embedding method, dimension reduction and clustering model
        Args:
            data (DataFrame) data to fit by model
        Return:
            model (sklearn model) fitted model
        """

        model_cluster = self.model_cluster()

        if self.name_clustering in ['NMF', 'LDA']:
            self.pipeline = model_cluster

        else:
            if 'acp' in self.name_model_full.lower():
                model_reduce = PCA(n_components=self.p["n_components"], random_state=15)
                self.pipeline = Pipeline([("acp", model_reduce),
                                          (self.name_clustering, model_cluster)])
            elif 'umap' in self.name_model_full.lower():
                model_reduce = umap.UMAP(n_neighbors=self.p["n_neighbors"], n_components=self.p["n_components"], metric='cosine')
                self.pipeline = Pipeline([("umap", model_reduce),
                                          (self.name_clustering, model_cluster)])
            else:
                self.pipeline = Pipeline([(self.name_clustering, model_cluster)])

        logger.info(self.pipeline)
        try:
            self.pipeline.fit(data)
        except:
            if 'umap' in self.name_model_full.lower():
                logger.info("Problem of Nan or np.float32 : use a scaler method for the first step")
                scaler = StandardScaler()
                model_reduce = umap.UMAP(n_neighbors=self.p["n_neighbors"], n_components=self.p["n_components"],
                                         metric='cosine')
                self.pipeline = Pipeline([("scaler", scaler),
                                          ("umap", model_reduce),
                                          (self.name_clustering, model_cluster)])
            self.pipeline.fit(data)

        outdir_embedding = os.path.join(self.outdir, 'clustering', self.name_embedding)
        os.makedirs(outdir_embedding, exist_ok=True)
        if self.name_embedding.lower() == "transformer":
            outdir_model = os.path.join(outdir_embedding, self.name_model_full)
        else:
            outdir_model = os.path.join(outdir_embedding, self.name_model_full.split('+')[1])
        os.makedirs(outdir_model, exist_ok=True)

        if self.apply_logs:
            dump(self.pipeline, '{}/{}.joblib'.format(outdir_model, self.name_clustering))

    def get_doc_topic(self, x_preprocessed, fit=False):
        """ Abstract method.

            Get a topic for each document
        Return:
            doc_topic (array) a topic for each document
            matrix_doc_topics (array) matrix document x topic (Optional)
        """
        pass

    def fit_transform(self, x_train_before, y_train=None, x_val_before=None, method_embedding=None,
                      doc_spacy_data_train=[], doc_spacy_data_val=[], show_plot=True):
        """ Apply fit_transform from :class:embedding on x_train_before and transform on x_val_before (Optional)
        Args:
            x_train_before (DataFrame)
            y_train (DataFrame)
            x_val_before (DataFrame)
            method_embedding (str) name of embedding method or path of embedding weights
            doc_spacy_data_train (List[spacy object])
            doc_spacy_data_val (List[spacy object])
            show_plot (bool)
        Return:
            dict_preprocessed (dict) with x_train_before preprocessed by embedding method and the topic for each document
                                    of x_train_before + x_val_before preprocessed by embedding method and
                                    the topic for each document of x_val_before
        """
        x_train = x_train_before.copy()
        if x_val_before is not None:
            x_val = x_val_before.copy()
        else:
            x_val = None

        self.method_embedding = method_embedding

        dict_preprocessed = self.embedding.fit_transform(x_train, x_val, self.method_embedding, doc_spacy_data_train,
                                                         doc_spacy_data_val)

        x_train_preprocessed = dict_preprocessed["x_train_preprocessed"]
        if x_val is not None:
            x_val_preprocessed = dict_preprocessed["x_val_preprocessed"]

        self.hyper_params()

        if self.name_clustering not in ["Similarity_voc", "Zero_shot_classification"]:
            self.model(x_train_preprocessed)
        else:
            outdir_embedding = os.path.join(self.outdir, 'clustering', self.name_embedding)
            os.makedirs(outdir_embedding, exist_ok=True)
            if self.name_embedding.lower() == "transformer":
                outdir_model = os.path.join(outdir_embedding, self.name_model_full)
            else:
                outdir_model = os.path.join(outdir_embedding, self.name_model_full.split('+')[1])
            os.makedirs(outdir_model, exist_ok=True)

        x_train_doc_topic, matrix_doc_topics = self.get_doc_topic(x_train_preprocessed)
        dict_preprocessed["x_train_doc_topic"] = x_train_doc_topic
        dict_preprocessed["matrix_doc_topics"] = matrix_doc_topics
        if x_val is not None:
            x_val_doc_topic, matrix_doc_topics = self.get_doc_topic(x_val_preprocessed)
            dict_preprocessed["x_val_doc_topic"] = x_val_doc_topic
            dict_preprocessed["matrix_doc_topics_val"] = x_val_doc_topic

        if self.flags_parameters.show_top_terms_topics:
            top_topic_terms = self.get_top_terms_per_topic(doc_spacy_data_train, x_train, x_train_doc_topic)

        # self.representation_without_label(x_train_preprocessed)
        if show_plot:
            if y_train is not None:
                self.representation_with_label(x_train_preprocessed, y_train)
            self.representation_with_label(x_train_preprocessed, x_train_doc_topic)

        # Compute metrics
        if self.flags_parameters.vocabulary_labels is not None or y_train is not None:
            self.map_cluster = self.metric_cluster(y_train, x_train_doc_topic, x_train_preprocessed)

        # save params in path : 'outdir/last_logs/name_embedding/name_model_full'
        if self.apply_logs:
            outdir_embedding = os.path.join(self.outdir, 'clustering', self.name_embedding)
            if self.name_embedding.lower() == "transformer":
                outdir_model = os.path.join(outdir_embedding, self.name_model_full)
            else:
                outdir_model = os.path.join(outdir_embedding, self.name_model_full.split('+')[1])
            self.save_params(outdir_model)
        else:
            self.save_params(None)

        return dict_preprocessed

    def transform(self, x_test_before_copy, doc_spacy_data_test=[], y_test_before_copy=None, loaded_models=None):
        """ Apply transform from :class:embedding on x_test_before_copy
        Args:
            x_test_before_copy (List or dict or DataFrame)
            doc_spacy_data_test (List[spacy object])
            y_test_before_copy (List or DataFrame)
            loaded_models (List)
        Return:
            dict_preprocessed (dict) with x_test_before_copy preprocessed by embedding method and the topic for each document
        """
        x_test = x_test_before_copy.copy()
        if y_test_before_copy is not None:
            y_test = y_test_before_copy.copy()
        else:
            y_test = None
        dict_preprocessed = {}

        try:
            _ = self.pipeline
        except:
            outdir_embedding = os.path.join(self.outdir, 'clustering', self.name_embedding)
            if self.name_embedding.lower() == "transformer":
                outdir_model = os.path.join(outdir_embedding, self.name_model_full)
            else:
                outdir_model = os.path.join(outdir_embedding, self.name_model_full.split('+')[1])

            with open(os.path.join(outdir_model, "parameters.json")) as json_file:
                params_all = json.load(json_file)
            params_all = params_all[self.name_model_full]
            self.load_params(params_all, os.path.join(self.outdir))

            if self.name_clustering not in ["Similarity_voc", "Zero_shot_classification"]:
                if loaded_models is None:
                    self.pipeline = load('{}/{}.joblib'.format(outdir_model, self.name_clustering))
                else:
                    self.pipeline = loaded_models[0]

        x_test_preprocessed = self.embedding.transform(x_test, doc_spacy_data_test)
        dict_preprocessed["x_test_preprocessed"] = x_test_preprocessed

        if self.name_clustering.lower() in ["hdbscan", "agglomerativeclustering"]:
            # Apply model fit because transform do not work for ["hdbscan", "agglomerativeclustering"]
            logger.info("Can't use previous {} clustering, refit {} on test data".format(self.name_clustering, self.name_clustering))
            self.hyper_params()
            self.model(x_test_preprocessed)

        x_test_doc_topic, matrix_doc_topics = self.get_doc_topic(x_test_preprocessed, fit=True)
        dict_preprocessed["x_test_doc_topic"] = x_test_doc_topic
        dict_preprocessed["matrix_doc_topics"] = matrix_doc_topics

        if y_test is not None and self.map_cluster is not None:
            if self.name_clustering.lower() in ["hdbscan", "agglomerativeclustering"]:
                logger.info("it is not possible to use previous {} clustering to predict target".format(self.name_clustering))
            else:
                self.transform_metric_cluster(y_test, x_test_doc_topic)

        return dict_preprocessed

    def get_data_reduced(self, data):
        """ Apply Dimension Reduction n=2, only use for visualization
        Args:
             data (DataFrame)
        """
        if self.name_clustering in ['NMF', 'LDA']:
            srp = TruncatedSVD(n_components=2, random_state=15)
            data_reduced = srp.fit_transform(data)
        else:
            try:
                data_reduced = self.pipeline[:-1].transform(data)
            except:
                model_reduce = PCA(n_components=2, random_state=15)
                data_reduced = model_reduce.fit_transform(data)

        return data_reduced

    def representation_without_label(self, data):
        """ 2D graphic representation
        Args:
            data (array) : reduced data
        """
        if self.name_clustering != "Zero_shot_classification":
            data_reduce = self.get_data_reduced(data)
            sns.set(rc={'figure.figsize': (10, 10)})
            palette = sns.color_palette("bright", 1)
            sns.scatterplot(data_reduce[:, 0], data_reduce[:, 1], palette=palette)
            plt.title('data reduced with no Labels')
            plt.show()

    def representation_with_label(self, data, labels_):
        """ 2D graphic representation
        Args:
            data (array) : reduced data
            labels_ (array) : label of reduced data clustering
        """
        try:
            # if gold labels:
            labels = labels_.values.copy()
            labels = labels.reshape(-1)
            map_label_y = {v: k for k, v in self.flags_parameters.map_label.items()}
            if map_label_y != {} and labels[0] in map_label_y.keys():
                labels = np.array([map_label_y[k] for k in labels])
            title = 'data reduced with gold label'
        except:
            labels = labels_.copy()
            title = 'data reduced with label cluster'

        try:
            if self.name_clustering != "Zero_shot_classification":
                data_reduce = self.get_data_reduced(data)
                sns.set(rc={'figure.figsize': (10, 10)})
                palette = sns.hls_palette(len(set(labels)), l=.4, s=.9)
                if -1 in list(set(labels)):
                    palette[0] = (1, 1, 1)

                sns.scatterplot(data_reduce[:, 0], data_reduce[:, 1], hue=labels, legend='full', palette=palette)
                plt.title(title)
                plt.show()
        except:
            pass

    def get_top_terms_per_topic(self, doc_spacy_data, x_train, doc_topic):
        """ Get top terms per topic with tf-idf weights
        Args:
             x_train (DataFrame)
             doc_spacy_data (array) documents from column_text preprocessed by spacy (Optional to replace x_train)
             doc_topic (array) a topic for each document
        Return:
            top_topic_terms (DataFrame) dataframe of the top terms for each topic
        """
        preprocess_topic = self.flags_parameters.preprocess_topic
        keep_pos_tag = preprocess_topic[0]
        lemmatize = preprocess_topic[1]

        if doc_spacy_data is not None:
            x_preprocessed = reduce_text_data(doc_spacy_data, keep_pos_tag, lemmatize)
        else:
            x_preprocessed = x_train

        vectorizer = TfidfVectorizer(ngram_range=(self.flags_parameters.min_ngram, self.flags_parameters.max_ngram),
                                     max_features=10000)
        tfidf = vectorizer.fit_transform(x_preprocessed)

        word_cluster = df_word_cluster(tfidf, doc_topic, vectorizer)
        top_topic_terms = topic_terms_cluster(word_cluster, self.flags_parameters.n_top_words, True, self.name_model_full)

        return top_topic_terms

    def metric_cluster(self, y_true_, doc_topic, x_preprocessed):
        """ For each Topic, associate the label with the most frequency in the topic :
            label can come from gold labels or from label using Similarity_voc method
        Args:
              y_true_ (DataFrame or None)
              doc_topic (array) a topic for each document
              x_preprocessed (array) documents preprocessed by embedding method
        """

        map_label = {}

        self.vocabulary_labels = self.flags_parameters.vocabulary_labels

        if self.name_clustering == "Zero_shot_classification":
            map_label = {k: k for k in self.vocabulary_labels.keys()}
            if y_true_ is not None:
                y_true = y_true_.values.copy()
                y_true = y_true.reshape(-1)
                map_label_y = {v: k for k, v in self.flags_parameters.map_label.items()}
                if map_label_y != {} and y_true[0] in map_label_y.keys():
                    y_true = np.array([map_label_y[k] for k in y_true])

                y_pred = [map_label[cluster] for cluster in doc_topic]

                self.info_scores = {}
                self.info_scores['accuracy_val'], self.info_scores['f1_' + self.average_scoring + '_val'], self.info_scores[
                    'recall_' + self.average_scoring + '_val'], self.info_scores[
                    'precision_' + self.average_scoring + '_val'] = calcul_metric_classification(y_true, y_pred,
                                                                                                 self.average_scoring, True)
            return map_label

        vect_vocabulary_labels = {}
        for k, list_words in self.vocabulary_labels.items():
            voc_preprocessed = self.embedding.transform(list_words, None)
            if voc_preprocessed is None:
                logger.error("No word of the {} label vocabulary could be vectorized".format(k))
                voc_preprocessed = np.array([[0]])
            else:
                if self.name_embedding not in ['tf-idf', 'tf']:
                    voc_preprocessed = np.array([v for v in voc_preprocessed if np.sum(v) != 0])
                    if voc_preprocessed.shape[0] == 0:
                        voc_preprocessed = np.array([[0]])
            vect_vocabulary_labels[k] = voc_preprocessed

        if y_true_ is not None:
            y_true = y_true_.values.copy()
            y_true = y_true.reshape(-1)
            map_label_y = {v: k for k, v in self.flags_parameters.map_label.items()}
            if map_label_y != {} and y_true[0] in map_label_y.keys():
                y_true = np.array([map_label_y[k] for k in y_true])
        else:
            logger.info("y_true is build from vocabulary_labels")
            y_true = clustering_similarity_voc(x_preprocessed, vect_vocabulary_labels)

        logger.info("Distribution y_true")
        logger.info(pd.Series(y_true).value_counts())

        for cluster in np.unique(doc_topic):
            #if cluster == -1:
            #    continue
            y_true_in_cluster = list(y_true[np.where(doc_topic == cluster)[0]])
            y_true_in_cluster_count = Counter(y_true_in_cluster)
            most_common_label = y_true_in_cluster_count.most_common(1)[0][0]
            map_label[cluster] = most_common_label

            doc_in_cluster = x_preprocessed[np.where(doc_topic == cluster)[0]]
            best_k = None
            best_score = -10e9
            for k, list_vect_voc in vect_vocabulary_labels.items():
                if (list_vect_voc == np.array([[0]])).all():
                    score = 0
                else:
                    score = np.mean(cosine_similarity(doc_in_cluster, list_vect_voc))
                if score > best_score:
                    best_score = score
                    best_k = k

            logger.info(
                "The most common label for cluster {} is {} and the label from vocabulary dictionary is {}".format(
                    cluster, most_common_label, best_k))

        y_pred = [map_label[cluster] for cluster in doc_topic]

        self.info_scores = {}
        self.info_scores['accuracy_val'], self.info_scores['f1_' + self.average_scoring + '_val'], self.info_scores[
            'recall_' + self.average_scoring + '_val'], self.info_scores[
            'precision_' + self.average_scoring + '_val'] = calcul_metric_classification(y_true, y_pred, self.average_scoring, True)

        return map_label

    def transform_metric_cluster(self, y_true_, doc_topic):
        """ Calcul all metric classification between y_true_ and y_pred, y_pred is obtained by previous function and
            doc_topic
        Args:
            y_true_ (DataFrame or None)
            doc_topic (array) a topic for each document
        """

        y_true = y_true_.values.copy()
        y_true = y_true.reshape(-1)
        map_label_y = {v: k for k, v in self.flags_parameters.map_label.items()}

        if map_label_y != {} and y_true[0] in map_label_y.keys():
            y_true = np.array([map_label_y[k] for k in y_true])

        logger.info("Distribution y_true")
        logger.info(pd.Series(y_true).value_counts())

        y_pred = []
        for cluster in doc_topic:
            try:
                y_pred.append(self.map_cluster[str(cluster)])
            except:
                y_pred.append(self.map_cluster[cluster])
        y_true = [str(yi) for yi in y_true]

        self.info_scores = {}
        self.info_scores['accuracy_test'], self.info_scores['f1_' + self.average_scoring + '_test'], \
        self.info_scores['recall_' + self.average_scoring + '_test'], \
        self.info_scores['precision_' + self.average_scoring + '_test'] = calcul_metric_classification(y_true, y_pred,
                                                                                         self.average_scoring, True)