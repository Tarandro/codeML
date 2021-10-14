import pytest
import pandas as pd
import numpy as np
import random as rd
from autonlp.autonlp import AutoNLP
from autonlp.flags import Flags
from autonlp.features.embeddings.word2vec import Word2Vec
from autonlp.models.clustering.kmeans import Kmeans_sklearn


class TestclassKmeans(object):
    """
    Test class Kmeans

    Input :
        flags : Instance of Flags class object
        x (DataFrame) : train data with the column flags.column_text
        x_val (DataFrame) : val data with the column flags.column_text
        y (DataFrame) : train data with the column flags.target
        y_val (DataFrame) : val data with the column flags.target
        column_text (int) : index column of flags.column_text in data input
        doc_spacy_data (array) : train data column_text preprocessed by a Spacy model
        doc_spacy_data_val (array) : val data column_text preprocessed by a Spacy model
        method_embedding (str) : name of the specific method to use for embedding
    """

    flags_dict_info = {
        "column_text": "column_text",
        "target": "target",
        "outdir": "./logs_test",
        "n_groups": 2,
        "w2v_size_vector": 10,
        "w2v_window": 3,
        "w2v_epochs": 3,
        "w2v_sg": 0,
        "w2v_maxlen": 10
    }

    flags = Flags().update(flags_dict_info)

    autonlp = AutoNLP(flags)

    x = pd.DataFrame({flags.column_text: [
        "text {} est écrit en noir.".format(rd.randint(0, 5)) + "text {} est écrit en bleu.".format(
            rd.randint(0, 5)) + "text {} est écrit en jaune.".format(rd.randint(0, 5)) for i in
        range(100)]})
    x_val = pd.DataFrame({flags.column_text: [
        "text {} est écrit en noir.".format(rd.randint(0, 5)) + "text {} est écrit en bleu.".format(
            rd.randint(0, 5)) + "text {} est écrit en jaune.".format(rd.randint(0, 5)) for i in
        range(30)]})
    y = pd.DataFrame({flags.target: [0 if i % 2 else 1 for i in range(100)]})
    y_val = pd.DataFrame({flags.target: [0 if i % 2 else 1 for i in range(30)]})

    folds = [('all', [i for i in range(y_val.shape[0])])]

    doc_spacy_data = None
    doc_spacy_data_val = None
    method_embedding = 'word2vec'

    column_text = 0

    def test_clustering_kmeans(self):

        model = Kmeans_sklearn(self.flags, Word2Vec, 'Word2Vec+Kmeans', self.column_text)
        dict_preprocessed = model.fit_transform(self.x, self.y, self.x_val, self.method_embedding,
                                                self.doc_spacy_data, self.doc_spacy_data_val, show_plot=False)

        assert dict_preprocessed["x_train_preprocessed"].shape[0] == len(self.x)
        assert len(np.unique(dict_preprocessed["x_train_doc_topic"])) <= self.flags.n_groups + 1

        assert dict_preprocessed["x_val_preprocessed"].shape[0] == len(self.x_val)
        assert len(np.unique(dict_preprocessed["x_val_doc_topic"])) <= self.flags.n_groups + 1