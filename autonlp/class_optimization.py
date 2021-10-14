import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time
from sklearn.metrics import *
import os
import shutil

import warnings

warnings.filterwarnings("ignore")

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import gc
from hyperopt import hp, fmin, tpe, Trials
import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter

from .utils.class_weight import compute_dict_class_weight

import logging
from .utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


def get_score_optimization(y, y_pred, shape_y, objective, scoring, average_scoring, train_set=False,
                           fold_id=None, x_val=None, y_val=None):
    """ Compute score between y and y_pred according to objective and scoring """
    metrics = []
    for i in range(shape_y):
        if train_set:
            y_true_sample = y
            prediction_sample = y_pred
        else:
            if x_val is None:
                # cross_validation
                y_true = y.iloc[:, i].copy()
            else:
                # validation
                y_true = y_val.iloc[:, i].copy()
            # subset, only use data where fold_id >= 0 :
            y_true_sample = y_true.values[np.where(fold_id >= 0)[0]]
            prediction_sample = y_pred[:, i][np.where(fold_id >= 0)[0]]
        if 'regression' in objective:
            if 'explained_variance' == scoring:
                metrics.append(-explained_variance_score(y_true_sample, prediction_sample))
            elif 'r2' == scoring:
                metrics.append(-r2_score(y_true_sample, prediction_sample))
            else:
                metrics.append(mean_squared_error(y_true_sample, prediction_sample))
        else:
            if 'f1' in scoring:
                if 'binary' in objective:
                    metrics.append(-f1_score(y_true_sample, prediction_sample))
                else:
                    metrics.append(
                        -f1_score(y_true_sample, prediction_sample, average=average_scoring))
            elif 'recall' in scoring:
                if 'binary' in objective:
                    metrics.append(-recall_score(y_true_sample, prediction_sample))
                else:
                    metrics.append(-recall_score(y_true_sample, prediction_sample,
                                                 average=average_scoring))
            elif 'precision' in scoring:
                if 'binary' in objective:
                    metrics.append(-precision_score(y_true_sample, prediction_sample))
                else:
                    metrics.append(-precision_score(y_true_sample, prediction_sample,
                                                    average=average_scoring))
            elif 'roc' in scoring or 'auc' in scoring:
                if 'binary' in objective:
                    metrics.append(-roc_auc_score(y_true_sample, prediction_sample))
                else:
                    metrics.append(-roc_auc_score(y_true_sample, prediction_sample,
                                                  average=average_scoring))
            else:
                metrics.append(-accuracy_score(y_true_sample, prediction_sample))

    score = -np.mean(metrics)
    return score

##############################
##############################
##############################


class Optimiz_hyperopt:
    """ Class Hyperopt optimization """

    def __init__(self, Model, hyper_params, apply_optimization):
        """
        Args:
            Model (Model class)
            hyper_params (dict) a hyperopt range for each hyperparameters
            apply_optimization (Boolean) if False, initialize random hyperparameters and give a null score
        """
        self.Model = Model
        self.hyper_params = hyper_params
        self.apply_optimization = apply_optimization
        self.apply_ray = self.Model.flags_parameters.apply_ray

    def optimise(self, params, checkpoint_dir=None):
        """ function to optimize by hyperopt library
        Args :
            params (dict) a hyperopt range for each hyperparameters
        Return:
            score (float) result of metric on validation set
        """

        self.Model.initialize_params(self.y, params)

        start = time.perf_counter()

        if self.x_val is None:
            # cross-validation
            fold_id = np.ones((len(self.y),)) * -1
            oof_val = np.zeros((self.y.shape[0], self.y.shape[1]))
        else:
            # validation
            fold_id = np.ones((len(self.y_val),)) * -1
            oof_val = np.zeros((self.y_val.shape[0], self.y_val.shape[1]))

        all_scores_train = []
        total_epochs = 0

        for n, (tr, te) in enumerate(self.folds):

            if tr == 'all':
                # validation
                if isinstance(self.x, pd.DataFrame):
                    x_tr, x_val = self.x.values, self.x_val.values
                else:
                    x_tr, x_val = self.x, self.x_val
                if isinstance(self.y, pd.DataFrame):
                    y_tr, y_val = self.y.values, self.y_val.values
                else:
                    y_tr, y_val = self.y, self.y_val
            else:
                # cross-validation
                if isinstance(self.x, pd.DataFrame):
                    x_tr, x_val = self.x.values[tr], self.x.values[te]
                elif isinstance(self.x, dict):
                    x_tr, x_val = {}, {}
                    for col in self.x.keys():
                        x_tr[col], x_val[col] = self.x[col][tr], self.x[col][te]
                elif isinstance(self.x, list):
                    x_tr, x_val = [], []
                    for col in range(len(self.x)):
                        x_tr.append(self.x[col][tr])
                        x_val.append(self.x[col][te])
                else:
                    x_tr, x_val = self.x[tr], self.x[te]
                if isinstance(self.y, pd.DataFrame):
                    y_tr, y_val = self.y.values[tr], self.y.values[te]
                else:
                    y_tr, y_val = self.y[tr], self.y[te]

            model = self.Model.model()

            if not self.Model.is_NN:
                model.fit(x_tr, y_tr)

                y_train_pred = model.predict(x_tr)
                score_train = get_score_optimization(y_tr, y_train_pred, self.Model.shape_y, self.Model.objective,
                                                     self.scoring, self.Model.average_scoring, train_set=True)
                all_scores_train.append(score_train)

                if self.Model.shape_y == 1:
                    oof_val[te, :] = model.predict(x_val).reshape(-1, 1)
                else:
                    oof_val[te, :] = model.predict(x_val)
                fold_id[te] = n
                del model

            else:
                if 'regression' in self.Model.objective:
                    if 'mean_squared_error' in self.scoring:
                        monitor = 'mean_squared_error'
                    else:
                        monitor = 'loss'
                else:
                    if self.Model.shape_y == 1:
                        if self.scoring == 'accuracy':
                            monitor = 'accuracy'
                        else:
                            monitor = 'loss'
                    else:
                        monitor = 'binary_crossentropy'

                rlr = ReduceLROnPlateau(monitor='val_' + monitor, factor=0.1, patience=self.Model.patience - 1,
                                        verbose=0, min_delta=1e-4, mode='auto', min_lr=self.Model.min_lr)

                # ckp = ModelCheckpoint(f'model_{n}.hdf5', monitor = 'val_loss', verbose = 0,
                #                      save_best_only = True, save_weights_only = True, mode = 'min')

                es = EarlyStopping(monitor='val_' + monitor, min_delta=0.0001, patience=self.Model.patience,
                                   mode='auto',
                                   baseline=None, restore_best_weights=True, verbose=0)

                history = model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
                                    epochs=int(self.Model.epochs), batch_size=self.Model.batch_size,
                                    class_weight=compute_dict_class_weight(y_tr, self.Model.class_weight,
                                                                           self.Model.objective),
                                    callbacks=[rlr, es], verbose=0)

                if len(history.history['val_' + monitor]) <= (self.Model.patience + 1):
                    p = 0
                else:
                    p = self.Model.patience
                total_epochs += len(history.history['val_' + monitor][:-(p + 1)])

                y_train_pred = model.predict(x_tr)
                y_val_pred = model.predict(x_val)
                if 'regression' in self.Model.objective:
                    oof_val[te, :] = y_val_pred
                else:
                    if self.Model.shape_y == 1 and 'binary' not in self.Model.objective:
                        y_train_pred = np.argmax(y_train_pred, axis=1).reshape(-1, 1)
                        oof_val[te, :] = np.argmax(y_val_pred, axis=1).reshape(-1, 1)
                    else:
                        y_train_pred = np.where(y_train_pred > 0.5, 1, 0)
                        oof_val[te, :] = np.where(y_val_pred > 0.5, 1, 0)
                fold_id[te] = n
                score_train = get_score_optimization(y_tr, y_train_pred, self.Model.shape_y,
                                                     self.Model.objective,
                                                     self.scoring, self.Model.average_scoring, train_set=True)
                all_scores_train.append(score_train)

                K.clear_session()
                del model, history
                d = gc.collect()

        score = get_score_optimization(self.y, oof_val, self.Model.shape_y, self.Model.objective,
                                       self.scoring, self.Model.average_scoring, train_set=False,
                                       fold_id=fold_id, x_val=self.x_val, y_val=self.y_val)
        logger.info('oof_val score {} Metric {}'.format(self.scoring, score))

        if self.Model.is_NN:
            params["epochs"] = total_epochs / len(self.folds)

        # store hyperparameters optimization in a Dataframe self.df_all_results:
        self.df_all_results['mean_fit_time'].append(time.perf_counter() - start)
        self.df_all_results['params'].append(params)
        self.df_all_results['train_score'].append(np.mean(all_scores_train))
        self.df_all_results['mean_test_score'].append(score)
        self.df_all_results['std_test_score'].append(0)  # just 0

        if self.apply_ray:
            tune.report(metric=score)
        else:
            return -score

    def optimise_no_optimiz(self, params):
        """ function to optimize by hyperopt library
            use when apply_optimization is False
            initialize random hyperparameters and give a null score
        Args :
            params (dict) a hyperopt range for each hyperparameters
        Return:
            score (float) result of metric on validation set
        """

        self.Model.initialize_params(self.y, params)

        if self.Model.is_NN:
            params["epochs"] = self.Model.epochs

        score = 0

        self.df_all_results['mean_fit_time'].append(0)
        self.df_all_results['params'].append(params)
        self.df_all_results['train_score'].append(score)
        self.df_all_results['mean_test_score'].append(score)
        self.df_all_results['std_test_score'].append(0)  # just 0

        if self.apply_ray:
            tune.report(metric=score)
        else:
            return score

    def train(self, x_, y_, folds, x_val, y_val, scoring='accuracy',
              time_limit_per_model=60, trials=None, max_trials=1000):
        """ Compute the function to minimize with hyperopt TPE optimization
            TPE optimization is a Naive Bayes Optimization
        Args:
            x_ (List or dict or DataFrame)
            y_ (Dataframe)
            x_val (List or dict or DataFrame) (Optional)
            y_val (Dataframe) (Optional)
            folds (List[tuple]) list of length self.nfolds_train with tuple (train_index, val_index)
            scoring (str) score to optimize
            time_limit_per_model (int) maximum Hyperparameters Optimization time in seconds
            trials (None or Trials object from hyperopt) if a Trials object is given, it will continue optimization
                    with this Trials
            max_trials (int) maximum number of trials
        """
        self.x = x_  # .copy().reset_index(drop=True)
        self.y = y_  # .copy().reset_index(drop=True)
        self.folds = folds
        self.x_val = x_val
        self.y_val = y_val
        self.scoring = scoring
        # keep an hyperparameters optimization history :
        self.df_all_results = {'mean_fit_time': [], 'params': [], 'mean_test_score': [], 'std_test_score': [],
                               'train_score': []}

        if not self.apply_ray:
            # Apply TPE Optimization with Hyperopt Library
            if trials is None:
                self.trials = Trials()
            else:
                self.trials = trials

            if self.apply_optimization:
                objective_function = self.optimise
            else:
                objective_function = self.optimise_no_optimiz
                max_trials = 1

            if time_limit_per_model == -1:
                time_limit_per_model = None

            self.hopt = fmin(fn=objective_function,
                             space=self.hyper_params,
                             algo=tpe.suggest,
                             max_evals=max_trials,
                             timeout=time_limit_per_model,
                             trials=self.trials,
                             )

        else:
            # Apply TPE Optimization with Ray Tune Framework and Ray Tune use Hyperopt Library
            os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

            hyperopt_search = HyperOptSearch(space=self.hyper_params, metric="metric", mode="max",
                                             random_state_seed=None)

            dir_hyperopt = os.path.join(self.Model.flags_parameters.outdir, "hyperopt")
            os.makedirs(dir_hyperopt, exist_ok=True)

            checkpoint_path = os.path.join(dir_hyperopt, "{}.pkl".format(self.Model.name_model_full))
            if os.path.exists(checkpoint_path):
                hyperopt_search.restore(checkpoint_path)

            max_concurrent = self.Model.flags_parameters.ray_max_model_parallel
            search_alg = ConcurrencyLimiter(hyperopt_search, max_concurrent=max_concurrent, batch=False)

            if self.apply_optimization:
                objective_function = self.optimise
            else:
                objective_function = self.optimise_no_optimiz
                max_trials = 1

            analysis = tune.run(objective_function, search_alg=search_alg,
                                num_samples=max_trials,
                                stop=TimeoutStopper(time_limit_per_model, max_concurrent),
                                verbose=self.Model.flags_parameters.ray_verbose,
                                resources_per_trial={"cpu": self.Model.flags_parameters.ray_cpu_per_model,
                                                     "gpu": self.Model.flags_parameters.ray_gpu_per_model},
                                raise_on_failed_trial=False,
                                name=self.Model.name_model_full,
                                local_dir=dir_hyperopt)

            hyperopt_search.save(os.path.join(dir_hyperopt, "{}.pkl".format(self.Model.name_model_full)))

            df_results = analysis.results_df
            df_results = df_results[~df_results.metric.isnull()]

            del analysis
            del hyperopt_search

            df_results_config = df_results[[col for col in df_results.columns if "config." in col]]
            df_results_config.columns = [col.replace("config.", "") for col in df_results_config.columns]

            self.df_all_results = {'mean_fit_time': list(df_results.time_this_iter_s),
                                   'params': list(df_results_config.to_dict(orient="index").values()),
                                   'train_score': [0 for i in range(len(df_results.metric))],
                                   'mean_test_score': list(df_results.metric),
                                   'std_test_score': [0 for i in range(len(df_results.metric))]}

            for path in os.listdir(dir_hyperopt):
                if os.path.isdir(path):
                    try:
                        shutil.rmtree(path)
                    except:
                        pass

        self.df_all_results = pd.DataFrame(self.df_all_results)
        self.index_best_score = self.df_all_results.mean_test_score.argmax()

    def show_distribution_score(self):
        plt.hist(self.df_all_results.mean_test_score)
        plt.show()

    def search_best_params(self):
        """ Look in history ensemble hyperparameters with best score
        Return:
            params (dict) best parameters from hyperparameters optimization
        """
        return self.df_all_results.loc[self.index_best_score, 'params'].copy()

    def transform_list_stopwords(self, params):
        """ Transform list stop_words in params to boolean type
        Args:
            params (dict)
        """
        if 'vect__text__tfidf__stop_words' in params.keys() and params['vect__text__tfidf__stop_words'] is not None:
            params['vect__text__tfidf__stop_words'] = True
        if 'vect__text__tf__stop_words' in params.keys() and params['vect__text__tf__stop_words'] is not None:
            params['vect__text__tf__stop_words'] = True
        if 'vect__tfidf__stop_words' in params.keys() and params['vect__tfidf__stop_words'] is not None:
            params['vect__tfidf__stop_words'] = True
        if 'vect__tf__stop_words' in params.keys() and params['vect__tf__stop_words'] is not None:
            params['vect__tf__stop_words'] = True
        return params

    def best_params(self):
        """
        Return:
            params (dict) best parameters from hyperparameters optimization
        """
        print_params = self.search_best_params()
        print_params = self.transform_list_stopwords(print_params)
        logger.info('Best parameters: {}'.format(print_params))
        return self.search_best_params()

    def best_score(self):
        """
        Return:
            score (int) : best score from hyperparameters optimization
        """
        score = self.df_all_results.loc[self.index_best_score, 'mean_test_score'].copy()
        logger.info('Mean cross-validated score of the best_estimator: {}'.format(np.round(score, 4)))
        return score

    def best_estimator(self):
        return None

    def get_summary(self, sort_by='mean_test_score'):
        """ Get hyperparameters optimization history
        Return:
            df_all_results (Dataframe)
        """
        df_all_results = self.df_all_results[
            ['mean_fit_time', 'params', 'train_score', 'mean_test_score', 'std_test_score']].sort_values(
            by=sort_by, ascending=False).reset_index(drop=True)
        df_all_results.params = df_all_results.params.apply(lambda d: self.transform_list_stopwords(d))
        return df_all_results


class TimeoutStopper(ray.tune.Stopper):
    """Stops all trials after a certain timeout.

    This stopper is automatically created when the `time_budget_s`
    argument is passed to `tune.run()`.

    Args:
        timeout (int|float|datetime.timedelta): Either a number specifying
            the timeout in seconds, or a `datetime.timedelta` object.
    """

    def __init__(self, timeout, max_concurrent):
        from datetime import timedelta
        if isinstance(timeout, timedelta):
            self._timeout_seconds = timeout.total_seconds()
        elif isinstance(timeout, (int, float)):
            self._timeout_seconds = timeout
        else:
            raise ValueError(
                "`timeout` parameter has to be either a number or a "
                "`datetime.timedelta` object. Found: {}".format(type(timeout)))

        self.max_concurrent = max_concurrent
        # To account for setup overhead, set the start time only after
        # the first call to `stop_all()`.
        self._start = None
        self.nb_done = 0

    def __call__(self, trial_id, result):
        if result["done"]:
            self.nb_done += 1
        return False

    def stop_all(self):
        if not self._start:
            self._start = time.time()
            return False

        now = time.time()
        if now - self._start >= self._timeout_seconds and self.nb_done >= self.max_concurrent and self._timeout_seconds > 0:
            logger.info(f"Reached timeout of {self._timeout_seconds} seconds and Number Trials DONE >= max_concurrent. "
                        f"Stopping all trials.")
            return True
        return False