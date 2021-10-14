import pytest
import pandas as pd
import numpy as np
import random as rd
from autonlp.autonlp import AutoNLP
from autonlp.flags import Flags
from autonlp.features.embeddings.word2vec import Word2Vec


class TestclassWord2Vec(object):
    """
    Test class Word2Vec

    Input :
        flags : Instance of Flags class object
        x (DataFrame) : train data with the column flags.column_text
        x_test (DataFrame) : test data with the column flags.column_text
        column_text (int) : index column of flags.column_text in data input
        method_embedding (str) : name of the specific method to use for embedding
    """

    flags_dict_info = {
        "column_text": "column_text",
        "target": "target",
        "outdir": "./logs_test",
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
    x_test = pd.DataFrame({flags.column_text: [
        "text {} est écrit en noir.".format(rd.randint(0, 5)) + "text {} est écrit en bleu.".format(
            rd.randint(0, 5)) + "text {} est écrit en jaune.".format(rd.randint(0, 5)) for i in
        range(30)]})

    method_embedding = 'word2vec'

    column_text = 0

    def test_doc_embedding(self):
        dimension_embedding = "doc_embedding"

        embedding = Word2Vec(self.flags, self.column_text, dimension_embedding)

        x_preprocessed = embedding.preprocessing_fit_transform(self.x, method_embedding=self.method_embedding)
        x_test_preprocessed = embedding.preprocessing_transform(self.x_test)

        assert x_preprocessed.shape[0] == len(self.x)
        assert x_preprocessed.shape[1] == self.flags.w2v_size_vector
        assert x_test_preprocessed.shape[0] == len(self.x_test)
        assert x_test_preprocessed.shape[1] == self.flags.w2v_size_vector

    def test_word_embedding(self):
        dimension_embedding = "word_embedding"

        embedding = Word2Vec(self.flags, self.column_text, dimension_embedding)

        x_token = embedding.preprocessing_fit_transform(self.x, method_embedding=self.method_embedding)
        x_test_token = embedding.preprocessing_transform(self.x_test)

        embedding_matrix = embedding.embedding_matrix
        length_word_index = embedding.length_word_index
        embed_size = embedding.embed_size

        assert isinstance(x_token, dict)
        assert isinstance(x_test_token, dict)

        assert x_token["tok"].shape[1] == self.flags.w2v_maxlen
        assert x_test_token["tok"].shape[1] == self.flags.w2v_maxlen

        assert embedding_matrix.shape[0] == length_word_index + 1
        assert embedding_matrix.shape[1] == embed_size