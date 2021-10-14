import pytest
import pandas as pd
import numpy as np
import random as rd
from autonlp.autonlp import AutoNLP
from autonlp.flags import Flags
from autonlp.features.embeddings.tfidf import Tfidf
from autonlp.models.classifier.sgd_classifier import SGD_Classifier


class TestclassSGDClassifier(object):
    """
    Test class SGDClassifier

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
        "objective": 'binary',
        "outdir": "./logs_test",
        "max_trial_per_model": 1
    }

    flags = Flags().update(flags_dict_info)

    autonlp = AutoNLP(flags)

    x = pd.DataFrame({flags.column_text: [
        "text" + str(rd.randint(0, 5)) + " text" + str(rd.randint(0, 5)) + " text" + str(rd.randint(0, 5)) for i in
        range(100)]})
    y = pd.DataFrame({flags.target: [0 if i % 2 else 1 for i in range(100)]})
    x_val = pd.DataFrame({flags.column_text: [
        "text" + str(rd.randint(0, 5)) + " text" + str(rd.randint(0, 5)) + " text" + str(rd.randint(0, 5)) for i in
        range(30)]})
    y_val = pd.DataFrame({flags.target: [0 if i % 2 else 1 for i in range(30)]})
    folds = [('all', [i for i in range(y_val.shape[0])])]

    doc_spacy_data = None
    doc_spacy_data_val = None
    method_embedding = ('all', False)

    column_text = 0

    def test_optimization(self):

        model = SGD_Classifier(self.flags, Tfidf, 'Tf_idf+SGD_Classifier', self.column_text)
        model.autonlp(self.x, y_train=self.y, x_val_before=self.x_val, y_val=self.y_val,
                      apply_optimization=True, apply_validation=False, method_embedding=self.method_embedding,
                      doc_spacy_data_train=self.doc_spacy_data, doc_spacy_data_val=self.doc_spacy_data_val,
                      folds=self.folds)

        assert len(model.df_all_results) == 1
        assert model.best_cv_score > 0

    def test_validation(self):

        model = SGD_Classifier(self.flags, Tfidf, 'Tf_idf+SGD_Classifier', self.column_text)
        model.autonlp(self.x, y_train=self.y, x_val_before=self.x_val, y_val=self.y_val,
                      apply_optimization=False, apply_validation=True, method_embedding=self.method_embedding,
                      doc_spacy_data_train=self.doc_spacy_data, doc_spacy_data_val=self.doc_spacy_data_val,
                      folds=self.folds)

        assert model.info_scores["accuracy_train"] > 0
        assert model.info_scores["accuracy_val"] > 0
        assert model.info_scores["f1_train"] >= 0
        assert model.info_scores["f1_val"] >= 0