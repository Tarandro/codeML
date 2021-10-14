import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import KFold, StratifiedKFold
from joblib import dump, load
import random as rd
import os

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from .utils.metrics import roc, calcul_metric_binary, calcul_metric_classification, calcul_metric_regression
from .utils.class_weight import compute_dict_class_weight

import logging
from .utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


class Validation:
    """ Class validation/cross-validation """

    def __init__(self, objective, seed=15, is_NN=False, name_embedding=None, name_model_full=None, class_weight=None,
                 average_scoring="weighted", apply_mlflow=False, experiment_name="Experiment", apply_logs=True):
        """
        Args:
            objective (str) : 'binary' or 'multi-class' or 'regression'
            seed (int)
            is_NN (Boolean) True if the model is a Neural Network
            name_embedding (str) name of the embedding method
            name_model_full (str) : full name of model (embedding+classifier+tag)
            class_weight (None or 'balanced')
            average_scoring (str) : 'micro', 'macro' or 'weighted'
            apply_mlflow (Boolean) : use MLflow Tracking
            experiment_name (str) : name of MLflow experiment
            apply_logs (Boolean) : use manual logs
        """
        self.seed = seed
        self.objective = objective
        self.is_NN = is_NN
        self.name_embedding = name_embedding
        self.name_model_full = name_model_full
        self.class_weight = class_weight
        self.average_scoring = average_scoring
        self.apply_mlflow = apply_mlflow
        self.path_mlflow = "../mlruns"
        self.experiment_name = experiment_name
        self.apply_logs = apply_logs
        self.oof_val = None
        self.fold_id = None

    def fit(self, model, x, y, folds, x_valid=None, y_valid=None, cv_strategy="StratifiedKFold",
            scoring='accuracy', outdir='./', params_all=dict(), batch_size=16, patience=4, epochs=60, min_lr=1e-4):
        """ Fit model on train set and predict on cross-validation / validation
        Args:
            model (self.model) function of the model architecture not instantiated : self.model and not self.model()
            x (List or dict or DataFrame)
            y (Dataframe)
            x_valid (List or dict or DataFrame)
            y_valid (Dataframe)
            folds (List[tuple]) list of length self.nfolds_train with tuple (train_index, val_index)
            cv_strategy ("StratifiedKFold" or "KFold")
            scoring (str) score to optimize
            outdir (str) logs path
            params_all (dict) params to save in order to reuse the model
            Only for NN models:
                batch_size (int)
                patience (int)
                epochs (int)
                min_lr (float) minimum for learning rate reduction
        """

        if x_valid is None:
            self.fold_id = np.ones((len(y),)) * -1
        else:
            self.fold_id = np.ones((len(y_valid),)) * -1

        if self.apply_logs:
            outdir_embedding = os.path.join(outdir, 'last_logs', self.name_embedding)
            os.makedirs(outdir_embedding, exist_ok=True)
            if self.name_embedding.lower() == "transformer":
                outdir_model = os.path.join(outdir_embedding, self.name_model_full)
            else:
                outdir_model = os.path.join(outdir_embedding, self.name_model_full.split('+')[1])
            os.makedirs(outdir_model, exist_ok=True)

        if self.apply_mlflow:
            import mlflow
            from mlflow.tracking import MlflowClient
            # mlflow : (mlflow ui --backend-store-uri ./mlruns)
            client = MlflowClient()
            experiment_name = self.experiment_name
            #experiment_id = client.create_experiment(experiment_name)
            try:
                mlflow.set_experiment(experiment_name=experiment_name)
            except:
                current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
                experiment_id = current_experiment['experiment_id']
                client.restore_experiment(experiment_id)
            current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
            experiment_id = current_experiment['experiment_id']

        all_metrics_train = {}

        total_epochs = 0
        first_fold = True

        for num_fold, (train_index, val_index) in enumerate(folds):
            if self.apply_mlflow:
                mlflow.start_run(run_name=self.name_model_full, tags={'name_embedding': self.name_embedding})
            logger.info("Fold {}:".format(num_fold))

            if train_index == 'all':
                # validation
                if isinstance(x, dict):
                    x_train, x_val = x, x_valid
                    y_train, y_val = y.values, y_valid.values
                elif isinstance(x, list):
                    x_train, x_val = x, x_valid
                    y_train, y_val = y.values, y_valid.values
                else:
                    if isinstance(x, pd.DataFrame):
                        x_train, x_val = x.values, x_valid.values
                    else:
                        x_train, x_val = x, x_valid
                    if isinstance(y, pd.DataFrame):
                        y_train, y_val = y.values, y_valid.values
                    else:
                        y_train, y_val = y, y_valid
            else:
                # cross-validation
                if isinstance(x, dict):
                    x_train, x_val = {}, {}
                    for col in x.keys():
                        x_train[col], x_val[col] = x[col][train_index], x[col][val_index]
                elif isinstance(x, list):
                    x_train, x_val = [], []
                    for col in range(len(x)):
                        x_train.append(x[col][train_index])
                        x_val.append(x[col][val_index])
                else:
                    if isinstance(x, pd.DataFrame):
                        x_train, x_val = x.values[train_index], x.values[val_index]
                    else:
                        x_train, x_val = x[train_index], x[val_index]
                if isinstance(y, pd.DataFrame):
                    y_train, y_val = y.values[train_index], y.values[val_index]
                else:
                    y_train, y_val = y[train_index], y[val_index]

            if self.is_NN:
                K.clear_session()

                model_nn = model()

                if first_fold:
                    logger.info(model_nn.summary())

                if 'regression' in self.objective:
                    if 'mean_squared_error' in scoring or 'mse' in scoring:
                        monitor = 'mean_squared_error'
                    else:
                        monitor = 'loss'
                    monitor_checkpoint = 'mean_squared_error'
                else:
                    if y.shape[1] == 1:
                        if scoring == 'accuracy':
                            monitor = 'accuracy'
                            monitor_checkpoint = 'accuracy'
                        else:
                            monitor = 'loss'
                            monitor_checkpoint = 'loss' #'accuracy'
                    else:
                        monitor = 'binary_crossentropy'
                        monitor_checkpoint = 'binary_crossentropy'

                rlr = ReduceLROnPlateau(monitor='val_' + monitor, factor=0.1, patience=patience - 1,
                                        verbose=1, min_delta=1e-4, mode='auto', min_lr=min_lr)

                es = EarlyStopping(monitor='val_' + monitor, min_delta=0.0001, patience=patience, mode='auto',
                                   baseline=None, restore_best_weights=True, verbose=0)

                if self.apply_logs:
                    ckp = ModelCheckpoint('{}/fold{}.hdf5'.format(outdir_model, num_fold),
                                          monitor='val_' + monitor_checkpoint, verbose=1,
                                          save_best_only=True, save_weights_only=True, mode='auto')
                    callbacks = [rlr, es, ckp]
                else:
                    callbacks = [rlr, es]

                if self.apply_mlflow:
                    mlflow.tensorflow.autolog(every_n_iter=2, silent=True)

                train_history = model_nn.fit(x_train, y_train,
                                             validation_data=(x_val, y_val),
                                             class_weight=compute_dict_class_weight(y_train, self.class_weight,
                                                                                        self.objective),
                                             epochs=int(epochs), batch_size=batch_size, verbose=1, callbacks=callbacks)

                if self.apply_logs:
                    outdir_plot = os.path.join(outdir_model, "plot_metrics_fold{}.png".format(num_fold))
                    save_plot_metrics(train_history.history, outdir_plot)

                if len(train_history.history[monitor]) <= (patience+1):
                    p = 0
                else:
                    p = patience

                logger.info('Kfold #{} : train {} = {} and val {} = {}'.format(num_fold, monitor,
                                                                               train_history.history[monitor][-(p + 1)],
                                                                               monitor,
                                                                               train_history.history['val_' + monitor][
                                                                                       -(p + 1)]))
                total_epochs += len(train_history.history[monitor][:-(p + 1)])

                pred_train = model_nn.predict(x_train)
                pred_val = model_nn.predict(x_val)

            else:
                model_skl = model()
                # mlflow.sklearn.autolog(log_models=False, exclusive=True)
                model_skl.fit(x_train, y_train)
                if self.apply_mlflow:
                    mlflow.sklearn.log_model(model_skl, self.name_model_full)

                if self.apply_logs:
                    dump(model_skl, '{}/fold{}.joblib'.format(outdir_model, num_fold))

                if 'regression' in self.objective:
                    pred_train = model_skl.predict(x_train)
                    pred_val = model_skl.predict(x_val)
                else:
                    pred_train = model_skl.predict_proba(x_train)
                    pred_val = model_skl.predict_proba(x_val)

            if first_fold:
                first_fold = False
                if 'binary' in self.objective or ('regression' in self.objective and y.shape[1] == 1):
                    if train_index == 'all':
                        self.oof_val = np.zeros((y_val.shape[0],))
                    else:
                        self.oof_val = np.zeros((y.shape[0],))
                else:
                    if train_index == 'all':
                        self.oof_val = np.zeros((y_val.shape[0], pred_val.shape[1]))
                    else:
                        self.oof_val = np.zeros((y.shape[0], pred_val.shape[1]))

            if self.is_NN:
                if 'binary' in self.objective or ('regression' in self.objective and y.shape[1] == 1):
                    pred_train = pred_train.reshape(-1)
                    pred_val = pred_val.reshape(-1)
            else:
                if 'binary' in self.objective:
                    pred_train = pred_train[:, 1].reshape(x_train.shape[0], )
                    pred_val = pred_val[:, 1].reshape(x_val.shape[0], )
                elif 'regression' in self.objective and y.shape[1] == 1:
                    pred_train = pred_train.reshape(x_train.shape[0], )
                    pred_val = pred_val.reshape(x_val.shape[0], )
            self.oof_val[val_index] = pred_val
            self.fold_id[val_index] = num_fold

            # log_metrics for Train set :
            if 'binary' in self.objective:
                m_binary, roc_binary = self.get_metrics(y_train, pred_train, False)
                acc_train, f1_train, recall_train, pre_train, roc_auc_train = m_binary
                m_binary, roc_binary = self.get_metrics(y_val, pred_val, False)
                acc_val, f1_val, recall_val, pre_val, roc_auc_val = m_binary
                metrics_train = {"acc_train": acc_train, "f1_train": f1_train, "recall_train": recall_train,
                                 "pre_train": pre_train, "roc_auc_train": roc_auc_train}
                metrics = {"acc_val": acc_val, "f1_val": f1_val, "recall_val": recall_val, "pre_val": pre_val,
                           "roc_auc_val": roc_auc_val}
            elif 'multi-class' in self.objective:
                acc_train, f1_train, recall_train, pre_train = self.get_metrics(y_train, pred_train, False)
                acc_val, f1_val, recall_val, pre_val = self.get_metrics(y_val, pred_val, False)
                metrics_train = {"acc_train": acc_train, "f1_train": f1_train, "recall_train": recall_train,
                                 "pre_train": pre_train}
                metrics = {"acc_val": acc_val, "f1_val": f1_val, "recall_val": recall_val, "pre_val": pre_val}
            elif 'regression' in self.objective:
                mse_train, rmse_train, expl_var_train, r2_train = self.get_metrics(y_train, pred_train, False)
                mse_val, rmse_val, expl_var_val, r2_val = self.get_metrics(y_val, pred_val, False)
                metrics_train = {"mse_train": mse_train, "rmse_train": rmse_train, "expl_var_train": expl_var_train,
                                 "r2_train": r2_train}
                metrics = {"mse_val": mse_val, "rmse_val": rmse_val, "expl_var_val": expl_var_val,
                           "r2_val": r2_val}

            if all_metrics_train == {}:
                all_metrics_train = {k: [v] for k, v in metrics_train.items()}
            else:
                all_metrics_train = {k: v + [metrics_train[k]] for k, v in all_metrics_train.items()}

            if self.apply_mlflow:
                mlflow.log_metrics(metrics)

                # log params
                params = {"seed": self.seed, "objective": self.objective,
                          "scoring": scoring, "average_scoring": self.average_scoring}
                if x_valid is None:
                    params["nfolds"] = len(folds)
                    params["num_fold"] = num_fold
                    params["cv_strategy"] = cv_strategy
                mlflow.log_params(params)

                # log params_all
                mlflow.log_dict(params_all, "parameters.json")
                mlflow.end_run()

                if self.is_NN:
                    mlflow.tensorflow.autolog(every_n_iter=2, log_models=False, disable=True, silent=True)

        prediction_oof_val = self.oof_val.copy()

        if isinstance(y, pd.DataFrame):
            if x_valid is None:
                # cross-validation
                y_true_sample = y.values[np.where(self.fold_id >= 0)[0]].copy()
            else:
                # validation
                y_true_sample = y_valid.values[np.where(self.fold_id >= 0)[0]].copy()
        else:
            if x_valid is None:
                # cross-validation
                y_true_sample = y[np.where(self.fold_id >= 0)[0]].copy()
            else:
                # validation
                y_true_sample = y_valid[np.where(self.fold_id >= 0)[0]].copy()
        prediction_oof_val = prediction_oof_val[np.where(self.fold_id >= 0)[0]]

        all_metrics_train = {k: np.mean(v) for k, v in all_metrics_train.items()}

        if 'binary' in self.objective:
            m_binary, roc_binary = self.get_metrics(y_true_sample, prediction_oof_val)
            self.acc_val, self.f1_val, self.recall_val, self.pre_val, self.roc_auc_val = m_binary
            self.fpr, self.tpr = roc_binary
        elif 'multi-class' in self.objective:
            self.acc_val, self.f1_val, self.recall_val, self.pre_val = self.get_metrics(y_true_sample, prediction_oof_val)
        elif 'regression' in self.objective:
            self.mse_val, self.rmse_val, self.expl_var_val, self.r2_val = self.get_metrics(y_true_sample, prediction_oof_val)

        if "acc_train" in all_metrics_train:
            self.acc_train, self.f1_train, self.recall_train, self.pre_train = all_metrics_train["acc_train"], all_metrics_train["f1_train"], all_metrics_train["recall_train"], all_metrics_train["pre_train"]
        if "roc_auc_train" in all_metrics_train:
            self.roc_auc_train = all_metrics_train["roc_auc_train"]
        if "mse_train" in all_metrics_train:
            self.mse_train, self.rmse_train, self.expl_var_train, self.r2_train = all_metrics_train["mse_train"], all_metrics_train["rmse_train"], all_metrics_train["expl_var_train"], all_metrics_train["r2_train"]

        del x_train, x_val, y_train, y_val, model

    def fit_no_val(self, model, x, y, scoring='accuracy', outdir='./', params_all=dict(), batch_size=16, epochs=60):
        """ Fit model on train set and no prediction on validation set
        Args:
            model (self.model) function of the model architecture not instantiated : self.model and not self.model()
            x (List or dict or DataFrame)
            y (Dataframe)
            scoring (str) score to optimize
            outdir (str) logs path
            params_all (dict) params to save in order to reuse the model
            Only for NN models:
                batch_size (int)
                epochs (int)
        """

        if self.apply_logs:
            outdir_embedding = os.path.join(outdir, 'last_logs', self.name_embedding)
            os.makedirs(outdir_embedding, exist_ok=True)
            if self.name_embedding.lower() == "transformer":
                outdir_model = os.path.join(outdir_embedding, self.name_model_full)
            else:
                outdir_model = os.path.join(outdir_embedding, self.name_model_full.split('+')[1])
            os.makedirs(outdir_model, exist_ok=True)

        if self.apply_mlflow:
            import mlflow
            from mlflow.tracking import MlflowClient
            # mlflow : (mlflow ui --backend-store-uri ./mlruns)
            client = MlflowClient()
            experiment_name = self.experiment_name
            #experiment_id = client.create_experiment(experiment_name)
            try:
                mlflow.set_experiment(experiment_name=experiment_name)
            except:
                current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
                experiment_id = current_experiment['experiment_id']
                client.restore_experiment(experiment_id)
            current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
            experiment_id = current_experiment['experiment_id']

        if self.apply_mlflow:
            mlflow.start_run(run_name=self.name_model_full, tags={'name_embedding': self.name_embedding})

        x_train = x
        if isinstance(y, pd.DataFrame):
            y_train = y.values
        else:
            y_train = y

        if self.is_NN:
            K.clear_session()

            model_nn = model()

            logger.info(model_nn.summary())

            if self.apply_logs:
                ckp = ModelCheckpoint('{}/fold{}.hdf5'.format(outdir_model, "_train"),verbose=1,
                                      save_best_only=False, save_weights_only=True)
                callbacks = [ckp]
            else:
                callbacks = []

            if self.apply_mlflow:
                mlflow.tensorflow.autolog(every_n_iter=2, silent=True)

            train_history = model_nn.fit(x_train, y_train,
                                         class_weight=compute_dict_class_weight(y_train, self.class_weight,
                                                                                        self.objective),
                                         epochs=int(epochs), batch_size=batch_size, verbose=1, callbacks=callbacks)

            if self.apply_logs:
                outdir_plot = os.path.join(outdir_model, "plot_metrics_fold{}.png".format("_train"))
                save_plot_metrics(train_history.history, outdir_plot)

        else:
            model_skl = model()
            # mlflow.sklearn.autolog(log_models=False, exclusive=True)
            model_skl.fit(x_train, y_train)
            if self.apply_mlflow:
                mlflow.sklearn.log_model(model_skl, self.name_model_full)

            if self.apply_logs:
                dump(model_skl, '{}/fold{}.joblib'.format(outdir_model, "_train"))

        if self.apply_mlflow:
            # log_metrics
            if 'binary' in self.objective:
                metrics = {"acc_val": 0, "f1_val": 0, "recall_val": 0, "pre_val": 0, "roc_auc_val": 0}
            elif 'multi-class' in self.objective:
                metrics = {"acc_val": 0, "f1_val": 0, "recall_val": 0, "pre_val": 0}
            elif 'regression' in self.objective:
                metrics = {"mse_val": 0, "rmse_val": 0, "expl_var_val": 0, "r2_val": 0}

            mlflow.log_metrics(metrics)

            # log params
            params = {"seed": self.seed, "objective": self.objective, "class_weight": self.class_weight,
                      "scoring": scoring, "average_scoring": self.average_scoring}

            mlflow.log_params(params)

            # log params_all
            mlflow.log_dict(params_all, "parameters.json")
            mlflow.end_run()

            if self.is_NN:
                mlflow.tensorflow.autolog(every_n_iter=2, log_models=False, disable=True, silent=True)

        if 'binary' in self.objective:
            self.acc_train, self.f1_train, self.recall_train, self.pre_train, self.roc_auc_train = 0, 0, 0, 0, 0
            self.acc_val, self.f1_val, self.recall_val, self.pre_val, self.roc_auc_val = 0, 0, 0, 0, 0
            self.fpr, self.tpr = 0, 0
        elif 'multi-class' in self.objective:
            self.acc_train, self.f1_train, self.recall_train, self.pre_train = 0, 0, 0, 0
            self.acc_val, self.f1_val, self.recall_val, self.pre_val = 0, 0, 0, 0
        elif 'regression' in self.objective:
            self.mse_train, self.rmse_train, self.expl_var_train, self.r2_train = 0, 0, 0, 0
            self.mse_val, self.rmse_val, self.expl_var_val, self.r2_val = 0, 0, 0, 0
        del x_train, y_train, model

    def get_metrics(self, y_true, oof_val, print_score=True):
        """ Compute metric scores between y_true and oof_val according to self.objective and self.scoring
        Args:
            y_true, oof_val (DataFrame or array)
        Returns:
            tuple of metric scores
        """
        if 'regression' not in self.objective:
            if y_true.shape[1] == 1 and 'binary' not in self.objective:
                prediction_oof_val = np.argmax(oof_val, axis=1).reshape(-1).copy()
            else:
                prediction_oof_val = np.where(oof_val > 0.5, 1, 0).copy()
        else:
            prediction_oof_val = oof_val.copy()

        if 'binary' in self.objective:
            m_binary = calcul_metric_binary(y_true, prediction_oof_val, 0.5, print_score)
            try:
                roc_binary = roc(y_true.values, oof_val)
            except:
                roc_binary = roc(y_true, oof_val)
            return m_binary, roc_binary

        elif 'multi-class' in self.objective:
            return calcul_metric_classification(y_true, prediction_oof_val, self.average_scoring, print_score)
        elif 'regression' in self.objective:
            return calcul_metric_regression(y_true, prediction_oof_val, print_score)

    def get_cv_prediction(self):
        """
        Returns:
            fold_id (array) number of fold of each data, -1 if it was not use for validation
            oof_val (array) validation prediction, data not use for validation are removed
        """
        if self.oof_val is None:
            return self.fold_id, self.oof_val
        else:
            return self.fold_id, self.oof_val[np.where(self.fold_id >= 0)[0]]

    def get_train_scores(self):
        """
        Returns:
            scores (tuple(float)) score between y_true and oof_val according to objective
        """
        if 'binary' in self.objective:
            return self.acc_train, self.f1_train, self.recall_train, self.pre_train, self.roc_auc_train
        elif 'multi-class' in self.objective:
            return self.acc_train, self.f1_train, self.recall_train, self.pre_train
        elif 'regression' in self.objective:
            return self.mse_train, self.rmse_train, self.expl_var_train, self.r2_train

    def get_scores(self):
        """
        Returns:
            scores (tuple(float)) score between y_true and oof_val according to objective
        """
        if 'binary' in self.objective:
            return self.acc_val, self.f1_val, self.recall_val, self.pre_val, self.roc_auc_val
        elif 'multi-class' in self.objective:
            return self.acc_val, self.f1_val, self.recall_val, self.pre_val
        elif 'regression' in self.objective:
            return self.mse_val, self.rmse_val, self.expl_var_val, self.r2_val

    def get_roc(self):
        """
        Returns:
            fpr (array) Increasing false positive rates
            tpr (array) Increasing true positive rates
        """
        return self.fpr, self.tpr


def save_plot_metrics(history, outdir):
    """ Save plot of metrics from history for each epoch
    Args:
        history : history from tensorflow fit model
        outdir (str) save plot in outdir directory
    """
    # Plotting
    metrics = [x for x in history.keys() if 'val' not in x and 'lr' not in x]

    fig = make_subplots(rows=len(metrics), cols=1)
    # f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))

    for i, metric in enumerate(metrics):
        fig.add_trace(
            go.Scatter(x=np.arange(1, len(history[metric]) + 1), y=history[metric], name=metric, mode='lines+markers',
                       line=dict(color='royalblue'), legendgroup=str(i + 1)),
            row=i + 1, col=1
        )
        # axs[i].plot(range(1, len(history[metric]) + 2), history[metric], label=metric)
        if 'val_' + metric in history.keys():
            fig.add_trace(
                go.Scatter(x=np.arange(1, len(history['val_' + metric]) + 1), y=history['val_' + metric],
                           name='val_' + metric, mode='lines+markers', line=dict(color='firebrick'),
                           legendgroup=str(i + 1)),
                row=i + 1, col=1
            )
            # axs[i].plot(range(1, len(history['val_' + metric]) + 2), history['val_' + metric], label='val_' + metric)
        # axs[i].legend()
        # axs[i].grid()
        fig.update_xaxes(title_text="epoch", row=i + 1, col=1)
        fig.update_yaxes(title_text=metric, row=i + 1, col=1)

    fig.update_layout(title='Model metrics', legend_tracegroupgap=160)

    # plt.tight_layout()

    try:
        fig.write_image(outdir)
    except:
        pass
