import pandas as pd
import numpy as np
import random as rd
import os
from sklearn.model_selection import KFold, StratifiedKFold
from joblib import load, dump
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Prepare:
    """Class to compile full pipeline : Prepare data
            steps:
                - separate column text and target -> X and Y
                - Split data in train/test according to frac_trainset
                - create cross validation split or prepare validation dataset
    """

    def __init__(self, flags_parameters):
        """
        Args:
            flags_parameters : Instance of Flags class object
        From flags_parameters:
            column_text (str) : name of the column with texts (only one column)
            target (str or List) : names of target columns
            frac_trainset (float) pourcentage of data for train set
            map_label (dict) dictionary to map label to integer
            debug (bool) if True use only 50 data rows for training
            nfolds (int) number of folds to split dataset
            nfolds_train (int) number of folds to train during optimization/validation
            cv_strategy ("StratifiedKFold" or "KFold")
        """
        self.outdir = flags_parameters.outdir
        self.objective = flags_parameters.objective
        self.column_text = flags_parameters.column_text
        self.frac_trainset = flags_parameters.frac_trainset
        self.map_label = flags_parameters.map_label
        self.debug = flags_parameters.debug
        self.seed = flags_parameters.seed
        self.nfolds = flags_parameters.nfolds
        self.nfolds_train = flags_parameters.nfolds_train
        self.cv_strategy = flags_parameters.cv_strategy

        self.method_scaling = flags_parameters.method_scaling
        self.ordinal_features = flags_parameters.ordinal_features
        self.normalize = flags_parameters.normalize

        self.position_id = flags_parameters.position_id
        self.position_date = flags_parameters.position_date
        self.size_train_prc = flags_parameters.size_train_prc
        self.time_series_recursive = flags_parameters.time_series_recursive
        self.LSTM_date_features = flags_parameters.LSTM_date_features
        self.startDate_train = flags_parameters.startDate_train
        self.endDate_train = flags_parameters.endDate_train

        # self.target need to be a List
        self.target = flags_parameters.target
        if isinstance(self.target, list):
            self.target = self.target
        else:
            self.target = [self.target]

    def separate_X_Y(self, data):
        """ separate column text and target -> X and Y
        Args:
             data (DataFrame)
        Return:
            data (DataFrame) data from input with the column_text and without target columns
            Y (DataFrame) data from input with only target columns
            column_text (int) the column number of self.column_text (str) in data
        """

        if len([col for col in self.target if col in data.columns]) > 0:
            col_Y = [col for col in self.target if col in data.columns]
            Y = data[col_Y]

            # create a map label if labels are not numerics
            if Y.shape[1] == 1 and self.map_label == {}:
                if not pd.api.types.is_numeric_dtype(Y.iloc[:, 0]):
                    self.map_label = {label: i for i, label in enumerate(Y.iloc[:, 0].unique())}
            if Y.shape[1] == 1 and self.map_label != {}:
                if Y[Y.columns[0]].iloc[0] in self.map_label.keys():
                    Y[Y.columns[0]] = Y[Y.columns[0]].map(self.map_label)

            data = data.drop(col_Y, axis=1)

        else:
            Y = None

        if self.column_text is not None or self.column_text in data.columns:
            # for X, keep only the column 'self.column_text'
            # WARNING : self.column_text (int) is now the column number of self.column_text (str) in self.data
            column_text = list(data[[self.column_text]].columns).index(self.column_text)
            data = data[[self.column_text]]

        else:
            column_text = None

        return data, Y, column_text

    def split_data(self, data, Y, frac_trainset):
        """ split data, Y -> X_train, X_test, Y_train, Y_test
        Args:
            data (DataFrame) data with the column_text
            Y (DataFrame) data with target columns
            frac_trainset (float) fraction for training set
        Return:
            X_train (DataFrame) train data with column text
            Y_train (DataFrame) train data with target columns
            X_test (DataFrame) test data with column text
            Y_test (DataFrame) test data with target columns
        """
        # DEBUG
        if self.debug:
            logger.info("\n DEBUG MODE : only a small portion is use for training set")
            train_data = data.sample(n=min(50, len(data)), random_state=self.seed)
        else:
            train_data = data.sample(frac=frac_trainset, random_state=self.seed)

        # Train set
        X_train = train_data.copy()
        logger.info("\nTraining set size : {}".format(len(X_train)))
        if Y is not None:
            Y_train = Y.loc[train_data.index, :]
        else:
            Y_train = None

        # Test set
        if self.frac_trainset < 1:
            test_data = data.drop(train_data.index)
            X_test = test_data.copy()
            if Y is not None:
                Y_test = Y.drop(train_data.index)
            else:
                Y_test = None
            logger.info("Test set size : {}".format(len(X_test)))
        else:
            X_test, Y_test = None, None
            logger.info("Test set size : 0")

        return X_train, Y_train, X_test, Y_test

    def split_data_ts(self, data, Y, startDate_train, endDate_train):
        """ split data, Y -> X_train, X_test, Y_train, Y_test
        Args:
            data (DataFrame) data with the column_text
            Y (DataFrame) data with target columns
            frac_trainset (float) fraction for training set
        Return:
            X_train (DataFrame) train data with column text
            Y_train (DataFrame) train data with target columns
            X_test (DataFrame) test data with column text
            Y_test (DataFrame) test data with target columns
        """
        # DEBUG
        if self.debug:
            logger.info("\n DEBUG MODE : only a small dataset portion is used")
            data = data.sample(n=min(50, len(data)), random_state=self.seed)

        # Position ID for time_series objective
        if self.position_id is not None and self.position_id in data.columns and isinstance(self.position_id, str):
            position_id = data[[self.position_id]]
        else:
            position_id = self.position_id

        if data.shape[1] == 0:  # can't do tabular prediction (case if only column text)
            if startDate_train == 'all' and endDate_train == 'all':
                Y_train = Y.copy()
            elif startDate_train == 'all':
                Y_train = Y.loc[:endDate_train, :].copy()
            elif endDate_train == 'all':
                Y_train = Y.loc[startDate_train:, :].copy()
            else:
                Y_train = Y.loc[startDate_train:endDate_train, :].copy()
            if endDate_train != 'all' and endDate_train != Y.index[-1]:
                Y_test = Y.loc[endDate_train:, :].copy()
                if position_id is not None:
                    position_id_test = position_id[position_id.index.isin(list(Y_test.index))]
                else:
                    position_id_test = None
            else:
                Y_test = None
                position_id_test = None

            if position_id is not None:
                position_id_train = position_id[position_id.index.isin(list(Y_train.index))]
            else:
                position_id_train = None
            X_train = None
            X_test = None

        else:
            if startDate_train == 'all' and endDate_train == 'all':
                X_train = data
            elif startDate_train == 'all':
                X_train = data[data[self.position_date] <= endDate_train]
            elif endDate_train == 'all':
                X_train = data[data[self.position_date] >= startDate_train]
            else:
                X_train = data[(data[self.position_date] >= startDate_train) & (data[self.position_date] <= endDate_train)]

            if position_id is not None:
                position_id_train = position_id[position_id.index.isin(list(X_train.index))]
            else:
                position_id_train = None

            if endDate_train != 'all' and endDate_train != np.max(data[self.position_date]):
                X_test = data[data[self.position_date] > endDate_train]
                if position_id is not None:
                    position_id_test = position_id[position_id.index.isin(list(X_test.index))]
                else:
                    position_id_test = None
            else:
                X_test = None
                position_id_test = None

            # del self.data

            Y_train = Y.loc[X_train.index, :]
            try:
                Y_test = Y.loc[X_test.index, :]
            except:
                Y_test = None
                pass

        return X_train, Y_train, X_test, Y_test, position_id_train, position_id_test

    def fit_transform_normalize_data(self, X_train):
        self.features = X_train.columns.values

        if self.method_scaling == 'MinMaxScaler':
            self.scaler = MinMaxScaler(feature_range=(0, 1), copy=False)  # or (-1,1)
        elif self.method_scaling == 'RobustScaler':
            self.scaler = RobustScaler(copy=False)
        else:
            self.scaler = StandardScaler(copy=False)

        col_object = [col for col in X_train.columns if
                       str(X_train[col].dtypes) in ['O', 'object', 'category', 'bool']]

        self.column_to_normalize = [col for col in self.features if
                                    col not in self.ordinal_features + [self.column_text] + col_object]  # from pre because int

        if len(self.column_to_normalize) > 0:
            self.scaler.fit(X_train[self.column_to_normalize])

            dump(self.scaler, os.path.join(self.outdir, "scaler.pkl"))
            self.scaler_info = [self.scaler, self.column_to_normalize].copy()

            #for col in self.column_to_normalize:
            #    self.scaler.fit(self.X_train[[col]])
            #    a = self.X_train[[col]].values
            #    self.X_train[[col]] = self.scaler.transform(a)
            #    del a
            #    try:
            #        a = self.X_test[[col]].values
            #        self.X_test[[col]] = self.scaler.transform(a)
            #        del a
            #    except:
            #        pass

            ### take a lot of memory to do it all together !
            X_train[self.column_to_normalize] = self.scaler.transform(X_train[self.column_to_normalize].values)
            #try:
            #    X_val[self.column_to_normalize] = self.scaler.transform(X_val[self.column_to_normalize].values)
            #except:
            #    pass
            #try:
            #    X_test[self.column_to_normalize] = self.scaler.transform(X_test[self.column_to_normalize].values)
            #except:
            #    pass
        else:
            self.scaler_info = None

        return X_train

    def transform_normalize_data(self, X):
        self.features = X.columns.values

        col_object = [col for col in X.columns if
                      str(X[col].dtypes) in ['O', 'object', 'category', 'bool']]

        self.column_to_normalize = [col for col in self.features if
                                    col not in self.ordinal_features + [self.column_text] + col_object]  # from pre because int

        if len(self.column_to_normalize) > 0:
            try:
                scaler = self.scaler
            except:
                scaler = load(os.path.join(self.outdir, "scaler.pkl"))
            self.scaler_info = [scaler, self.column_to_normalize].copy()

            ### take a lot of memory to do it all together !
            X[self.column_to_normalize] = scaler.transform(X[self.column_to_normalize].values)
        else:
            self.scaler_info = None
        return X

    def create_validation(self, dataset_val):
        """ separate column text and target -> X and Y
        Args:
            dataset_val (DataFrame) validation dataset
        Return:
            dataset_val_copy (DataFrame) dataset_val from input with the column_text and without target columns
            Y_val (DataFrame) dataset_val from input with only target columns
            folds (List) length 1, format = [('all', [all index of dataset_val])], 'all' means that all train set
                        will be used for training and validated on dataset_val
        """

        if len([col for col in self.target if col in dataset_val.columns]) > 0:
            Y_val = dataset_val[[col for col in self.target if col in dataset_val.columns]]
            dataset_val_copy = dataset_val.drop([col for col in self.target if col in dataset_val.columns], axis=1)
        else:
            dataset_val_copy = dataset_val.copy()
            Y_val = None

        # use label map if labels are not numerics
        if self.map_label != {}:
            if Y_val[Y_val.columns[0]].iloc[0] in self.map_label.keys():
                Y_val[Y_val.columns[0]] = Y_val[Y_val.columns[0]].map(self.map_label)
                if Y_val[Y_val.columns[0]].isnull().sum() > 0:
                    logger.error("Unknown label name during map of test labels")

        folds = [('all', [i for i in range(Y_val.shape[0])])]

        return dataset_val_copy, Y_val, folds

    def create_cross_validation(self, X_train, Y_train):
        """ Create Cross-validation scheme
        Cross-validation split in self.nfolds but train only on self.nfolds_train chosen randomly
        Args:
            X_train (DataFrame)
            Y_train (DataFrame)
        Return:
            folds (List[tuple]) list of length self.nfolds_train with tuple (train_index, val_index)
        """
        # Cross-validation split in self.nfolds but train only on self.nfolds_train chosen randomly
        rd.seed(self.seed)
        fold_to_train = rd.sample([i for i in range(self.nfolds)], k=max(min(self.nfolds_train, self.nfolds), 1))

        if self.cv_strategy == "StratifiedKFold" and Y_train is not None:
            skf = StratifiedKFold(n_splits=self.nfolds, random_state=self.seed, shuffle=True)
            folds_sklearn = skf.split(np.array(Y_train), np.array(Y_train))
        else:
            if Y_train is None:
                kf = KFold(n_splits=self.nfolds, random_state=self.seed, shuffle=True)
                folds_sklearn = kf.split(np.array(X_train))
            else:
                kf = KFold(n_splits=self.nfolds, random_state=self.seed, shuffle=True)
                folds_sklearn = kf.split(Y_train)

        folds = []
        for num_fold, (train_index, val_index) in enumerate(folds_sklearn):
            if num_fold not in fold_to_train:
                continue
            folds.append((train_index, val_index))

        return folds

    def get_datasets(self, data, dataset_val=None):
        """ Use previous function of the class to prepare all needed dataset
        Args:
            data (DataFrame)
            dataset_val (DataFrame)
        """

        data, Y, column_text = self.separate_X_Y(data)

        if dataset_val is None:
            if 'time_series' in self.objective:
                X_tr, Y_tr, X_test, Y_test, position_id_train, position_id_test = self.split_data_ts(data, Y,
                                                                      self.startDate_train, self.endDate_train)
            else:
                X_tr, Y_tr, X_test, Y_test = self.split_data(data, Y, self.frac_trainset)
                position_id_train, position_id_test = None, None

            self.ordinal_features = [col for col in self.ordinal_features if col in X_tr.columns]
            self.LSTM_date_features = [col for col in self.LSTM_date_features if col in X_tr.columns]
            self.len_unique_value = {}
            for col in list(set(self.ordinal_features + self.LSTM_date_features)):
                self.len_unique_value[col] = len(data[col].unique())

            X_val, Y_val = None, None
            folds = self.create_cross_validation(X_tr, Y_tr)

            X_train, Y_train = [], []
            for n, (tr, te) in enumerate(folds):
                if X_tr is not None:
                    x_tr, x_val = X_tr.iloc[tr, :], X_tr.iloc[te, :]
                    X_train.append([x_tr, x_val])
                if Y_tr is not None:
                    y_tr, y_val = Y_tr.iloc[tr, :], Y_tr.iloc[te, :]
                    Y_train.append([y_tr, y_val])

            if len(X_train) == 0:
                X_train = None
            if len(Y_train) == 0:
                Y_train = None

        else:
            # if a validation dataset is provided, data is not split in train/test and validation data will
            # also be the test set -> frac_trainset = 1

            if 'time_series' in self.objective:
                self.startDate_train, self.endDate_train = "all", "all"
                X_train, Y_train, X_test, Y_test, position_id_train, position_id_test = self.split_data_ts(data, Y,
                                                                      self.startDate_train, self.endDate_train)
            else:
                frac_trainset = 1
                X_train, Y_train, X_test, Y_test = self.split_data(data, Y, frac_trainset)
                position_id_train, position_id_test = None, None

            self.ordinal_features = [col for col in self.ordinal_features if col in X_train.columns]
            self.LSTM_date_features = [col for col in self.LSTM_date_features if col in X_train.columns]
            self.len_unique_value = {}
            for col in list(set(self.ordinal_features + self.LSTM_date_features)):
                self.len_unique_value[col] = len(data[col].unique())

            X_val, Y_val, folds = self.create_validation(dataset_val)
            X_test, Y_test = X_val, Y_val

        if self.normalize:
            if not isinstance(X_train, list):
                X_train = self.fit_transform_normalize_data(X_train)
                X_val = self.transform_normalize_data(X_val)
                X_test = self.transform_normalize_data(X_test)
            else:
                for i in range(len(X_train)):
                    x_tr = self.fit_transform_normalize_data(X_train[i][0])
                    x_val = self.transform_normalize_data(X_train[i][1])
                    X_train[i] = [x_tr, x_val]
                if X_test is not None:
                    X_test = self.transform_normalize_data(X_test)
        else:
            self.scaler_info = None

        return column_text, X_train, Y_train, X_val, Y_val, X_test, Y_test, folds, position_id_train, position_id_test

    def get_test_datasets(self, data_test):

        data, Y, column_text = self.separate_X_Y(data_test)

        if self.normalize:
            data = self.transform_normalize_data(data)
        else:
            self.scaler_info = None
        return data, Y, column_text