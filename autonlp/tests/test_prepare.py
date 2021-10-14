import pytest
import pandas as pd
import numpy as np
from autonlp.flags import Flags
from autonlp.data.prepare_data import Prepare


class TestclassPrepare(object):
    """
    Test class Prepare

    Input :
        flags : Instance of Flags class object
        data (DataFrame) : data with a column flags.column_text and a column flags.target (optional)
        doc_spacy_data (array) : column_text preprocessed by a Spacy model
    """

    flags_dict_info = {
        "column_text": "column_text",
        "target": "target",
        "outdir": "./logs_test",
        "frac_trainset": 0.7,
        "nfolds": 5,
        "nfolds_train": 3,
        "cv_strategy": "KFold"
    }
    flags = Flags().update(flags_dict_info)

    data = pd.DataFrame({flags.column_text: ["text"+str(i) for i in range(100)],
                         flags.target: ["target1" if i % 2 else "target2" for i in range(100)]})

    doc_spacy_data = np.array(["text"+str(i) for i in range(100)])

    def test_cross_validation(self):
        """ Test Separation X Y for cross validation scheme """
        dataset_val, doc_spacy_data_val = None, None

        prepare = Prepare(self.flags)
        column_text, X_train, doc_spacy_data_train, Y_train, X_val, doc_spacy_data_val, Y_val, X_test, doc_spacy_data_test, Y_test, folds = prepare.get_datasets(
            self.data, self.doc_spacy_data, dataset_val, doc_spacy_data_val)

        assert column_text == 0
        assert len(X_train) == int(self.flags.frac_trainset * len(self.data))
        assert len(Y_train) == int(self.flags.frac_trainset * len(self.data))
        assert doc_spacy_data_train.shape[0] == int(self.flags.frac_trainset * len(self.data))
        assert X_train.columns == [self.flags.column_text]
        if isinstance(self.flags.target, list):
            assert Y_train.columns == self.flags.target
        else:
            assert Y_train.columns == [self.flags.target]
        assert len(folds) == self.flags.nfolds_train

    def test_validation(self):
        """ Test Separation X Y for validation scheme """
        dataset_val = pd.DataFrame({self.flags.column_text: ["text" + str(i) for i in range(30)],
                                    self.flags.target: ["target1" if i // 2 else "target2" for i in range(30)]})

        doc_spacy_data_val = np.array(["text" + str(i) for i in range(30)])

        prepare = Prepare(self.flags)
        column_text, X_train, doc_spacy_data_train, Y_train, X_val, doc_spacy_data_val, Y_val, X_test, doc_spacy_data_test, Y_test, folds = prepare.get_datasets(
            self.data, self.doc_spacy_data, dataset_val, doc_spacy_data_val)

        assert column_text == 0
        assert len(X_train) == len(self.data)
        assert len(Y_train) == len(self.data)
        assert doc_spacy_data_train.shape[0] == len(self.data)
        assert X_train.columns == [self.flags.column_text]
        assert len(X_val) == len(dataset_val)
        assert len(Y_val) == len(dataset_val)
        assert doc_spacy_data_val.shape[0] == len(dataset_val)
        assert X_val.columns == [self.flags.column_text]
        if isinstance(self.flags.target, list):
            assert Y_train.columns == self.flags.target
            assert Y_val.columns == self.flags.target
        else:
            assert Y_train.columns == [self.flags.target]
            assert Y_val.columns == [self.flags.target]
        assert len(folds) == 1

    def test_seed(self):
        """ Test 2 Separations X Y with same seed """
        dataset_val, doc_spacy_data_val = None, None

        prepare = Prepare(self.flags)
        column_text, X_train, doc_spacy_data_train, Y_train, X_val, doc_spacy_data_val, Y_val, X_test, doc_spacy_data_test, Y_test, folds = prepare.get_datasets(
            self.data, self.doc_spacy_data, dataset_val, doc_spacy_data_val)

        prepare = Prepare(self.flags)
        column_text_, X_train_, doc_spacy_data_train_, Y_train_, X_val_, doc_spacy_data_val_, Y_val_, X_test_, doc_spacy_data_test_, Y_test_, folds_ = prepare.get_datasets(
            self.data, self.doc_spacy_data, dataset_val, doc_spacy_data_val)

        assert (X_train.index==X_train_.index).all()
        assert (Y_train.index==Y_train_.index).all()
        for (train_index, val_index), (train_index_, val_index_) in zip(folds, folds_):
            assert (train_index==train_index_).all()
            assert (val_index==val_index_).all()