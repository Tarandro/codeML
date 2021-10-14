import pytest
import pandas as pd
import numpy as np
import random as rd
from autonlp.autonlp import AutoNLP
from autonlp.flags import Flags
from autonlp.features.embeddings.transformernlp import TransformerNLP

method_embedding_transformers_to_try = "Bert"
try_doc_embedding = False
try_word_embedding = False


class TestclassTransformerNLP(object):
    """
    Test class TransformerNLP

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
        "tr_maxlen": 5
    }
    flags = Flags().update(flags_dict_info)

    autonlp = AutoNLP(flags)

    x = pd.DataFrame({flags.column_text: [
        "text " + str(rd.randint(0, 5)) + " text " + str(rd.randint(0, 5)) + " text " + str(rd.randint(0, 5)) for i in
        range(100)]})
    x_test = pd.DataFrame({flags.column_text: [
        "text " + str(rd.randint(0, 5)) + " text " + str(rd.randint(0, 5)) + " text " + str(rd.randint(0, 5)) for i in
        range(30)]})

    column_text = 0

    def test_doc_embedding(self):
        if try_doc_embedding:
            dimension_embedding = "doc_embedding"

            embedding = TransformerNLP(self.flags, self.column_text, dimension_embedding)

            x_preprocessed = embedding.preprocessing_fit_transform(self.x, method_embedding=method_embedding_transformers_to_try)
            x_test_preprocessed = embedding.preprocessing_transform(self.x_test)

            assert len(x_preprocessed) == len(self.x)
            assert len(x_test_preprocessed) == len(self.x_test)
        else:
            pass

    def test_word_embedding(self):
        if try_word_embedding:
            dimension_embedding = "word_embedding"

            embedding = TransformerNLP(self.flags, self.column_text, dimension_embedding)

            x_preprocessed = embedding.preprocessing_fit_transform(self.x, method_embedding=method_embedding_transformers_to_try)
            x_test_preprocessed = embedding.preprocessing_transform(self.x_test)

            assert len(x_preprocessed) == 3
            assert x_preprocessed[0].shape[0] == len(self.x)
            assert x_test_preprocessed[0].shape[0] == len(self.x_test)
            assert x_test_preprocessed[0].shape[1] == self.flags.tr_maxlen
        else:
            pass