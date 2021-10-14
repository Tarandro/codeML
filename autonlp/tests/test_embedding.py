import pytest
import pandas as pd
import numpy as np
import random as rd
from autonlp.autonlp import AutoNLP
from autonlp.flags import Flags
from autonlp.features.embeddings.tfidf import Tfidf
from autonlp.models.embeddings.trainer import Embedding


class TestclassEmbeddingTfidf(object):
    """
    Test class Embedding

    Input :
        flags : Instance of Flags class object
        x (DataFrame) : train data with the column flags.column_text
        x_test (DataFrame) : test data with the column flags.column_text
        column_text (int) : index column of flags.column_text in data input
        doc_spacy_data (array) : train data column_text preprocessed by a Spacy model
        doc_spacy_data_test (array) : test data column_text preprocessed by a Spacy model
        method_embedding (str) : name of the specific method to use for embedding
    """

    flags_dict_info = {
        "column_text": "column_text",
        "target": "target",
        "outdir": "./logs_test",
        "tfidf_wde_maxlen": 10,
        "tfidf_wde_vector_size": 2
    }
    flags = Flags().update(flags_dict_info)

    autonlp = AutoNLP(flags)

    x = pd.DataFrame({flags.column_text: [
        "text" + str(rd.randint(0, 5)) + " text" + str(rd.randint(0, 5)) + " text" + str(rd.randint(0, 5)) for i in
        range(100)]})
    x_test = pd.DataFrame({flags.column_text: [
        "text" + str(rd.randint(0, 5)) + " text" + str(rd.randint(0, 5)) + " text" + str(rd.randint(0, 5)) for i in
        range(30)]})

    doc_spacy_data = None
    doc_spacy_data_test = None
    method_embedding = ('all', False)

    column_text = 0

    def test_doc_embedding(self):
        dimension_embedding = "doc_embedding"

        embedding = Embedding(self.flags, Tfidf, dimension_embedding, self.column_text)

        x_preprocessed = embedding.fit_transform(self.x, doc_spacy_data_train=self.doc_spacy_data,
                                                 method_embedding=self.method_embedding)
        x_test_preprocessed = embedding.transform(self.x_test, self.doc_spacy_data_test)

        assert x_preprocessed["x_train_preprocessed"].shape[0] == len(self.x)
        assert x_test_preprocessed.shape[0] == len(self.x_test)

    def test_word_embedding(self):
        dimension_embedding = "word_embedding"

        embedding = Embedding(self.flags, Tfidf, dimension_embedding, self.column_text)

        x_preprocessed = embedding.fit_transform(self.x, doc_spacy_data_train=self.doc_spacy_data,
                                                 method_embedding=self.method_embedding)
        x_test_preprocessed = embedding.transform(self.x_test, self.doc_spacy_data_test)

        assert x_preprocessed["x_train_preprocessed"].shape[0] == len(self.x)
        assert x_preprocessed["x_train_preprocessed"].shape[1] <= self.flags.tfidf_wde_maxlen