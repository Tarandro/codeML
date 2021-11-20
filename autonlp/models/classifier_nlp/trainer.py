import json
import pickle
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras.layers import Dense
from transformers import AdamWeightDecay
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from ...class_optimization import *
from ...validation import *
from ...prediction import *

import logging
from ...utils.logging import get_logger

logger = get_logger(__name__)


class Model:
    """ Parent class of NLP models
        Model steps:
            - parameters management : functions : hyper_params(), initialize_params(), save_params(), load_params()
            - compute model architecture : function : model()
            - hyperparameters optimization : function : fit_optimization()
            - validation/cross-validation : function : validation()
            - compute hyperparameters optimization and validation on train set : function : autonlp()
            - prediction on test set for each fold of a model : function : prediction()
    """

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        """
        Args:
            flags_parameters : Instance of Flags class object
            embedding (:class: Base_Embedding) : A children class from Base_Embedding
            column_text (int) : column number with texts
            name_model_full (str) : full name of model (embedding+classifier+tag)
            class_weight (None or 'balanced')

        From flags_parameters:
            objective (str) : 'binary' or 'multi-class' or 'regression'
            average_scoring (str) : 'micro', 'macro' or 'weighted'
            seed (int)
            apply_mlflow (Boolean) save model in self.path_mlflow (str) directory
            experiment_name (str) name of the experiment, only if MLflow is activated
            apply_logs (Boolean) use manual logs
            apply_app (Boolean) if you want to use a model from model_deployment directory
        """
        self.flags_parameters = flags_parameters

        self.embedding = embedding(flags_parameters, column_text, self.dimension_embedding)

        self.column_text = column_text
        self.objective = flags_parameters.objective
        self.average_scoring = flags_parameters.average_scoring
        self.apply_mlflow = flags_parameters.apply_mlflow
        self.path_mlflow = "../mlruns"
        self.experiment_name = flags_parameters.experiment_name
        self.apply_logs = flags_parameters.apply_logs
        self.apply_app = flags_parameters.apply_app
        self.seed = flags_parameters.seed
        self.name_model = None
        self.name_model_full = name_model_full
        self.class_weight = class_weight
        self.best_cv_score = 0.0
        self.df_all_results = pd.DataFrame()
        self.info_scores = {}

        if self.apply_mlflow:
            import mlflow
            current_experiment = dict(mlflow.get_experiment_by_name(self.experiment_name))
            self.experiment_id = current_experiment['experiment_id']

    def hyper_params(self, size_params='small'):
        """ Abstract method.

            Instantiate hyperparameters range for embedding method and classifier model that will be use for
            hyperopt optimization

        Args:
            size_params ('small' or 'big') size of parameters range for optimization
        Return:
            parameters (dict) a hyperopt range for each hyperparameters
        """
        pass

    def initialize_params(self, y, params):
        """ Initialize params to self.p / number of columns of y to self.y_shape /
            get number of classes (1 for regression)

        Args:
            y (Dataframe)
            params (dict) a hyperopt range for each hyperparameters
        """
        self.shape_y = y.shape[1]

        if self.shape_y == 1:
            if 'regression' in self.objective:
                self.nb_classes = 1
            else:
                self.nb_classes = len(np.unique(y))
        else:
            self.nb_classes = self.shape_y

        self.p = params

        if self.flags_parameters.language_text == 'fr':
            stopwords = list(fr_stop)
        else:
            stopwords = list(en_stop)
        # list of stop_words need to be boolean
        if 'vect__tf__stop_words' in self.p.keys() and self.p['vect__tf__stop_words']:
            self.p['vect__tf__stop_words'] = stopwords
        if 'vect__tfidf__stop_words' in self.p.keys() and self.p['vect__tfidf__stop_words']:
            self.p['vect__tfidf__stop_words'] = stopwords

    def save_params(self, outdir_model):
        """ Save all params as a json file needed to reuse the model
            + tensorflow tokenizer (pickle file) in outdir_model
        Args:
            outdir_model (str)
        """
        params_all = dict()
        p_model = self.p.copy()
        # list of stop_words is transformed in boolean
        if 'vect__tf__stop_words' in p_model.keys() and p_model['vect__tf__stop_words'] is not None:
            p_model['vect__tf__stop_words'] = True
        if 'vect__tfidf__stop_words' in p_model.keys() and p_model['vect__tfidf__stop_words'] is not None:
            p_model['vect__tfidf__stop_words'] = True
        params_all['p_model'] = p_model
        params_all['language_text'] = self.flags_parameters.language_text
        params_all['name_classifier'] = self.name_classifier

        params_all['nb_classes'] = self.nb_classes
        params_all['shape_y'] = self.shape_y

        params_embedding = self.embedding.save_params(outdir_model)
        params_all.update(params_embedding)

        self.params_all = {self.name_model_full: params_all}

        if self.apply_logs:
            with open(os.path.join(outdir_model, "parameters.json"), "w") as outfile:
                json.dump(self.params_all, outfile)

    def load_params(self, params_all, outdir):
        """ Initialize all params from params_all
            + tensorflow tokenizer (pickle file) from outdir path

        Args:
            params_all (dict)
            outdir (str)
        """
        if params_all['language_text'] == 'fr':
            stopwords = list(fr_stop)
        else:
            stopwords = list(en_stop)
        p_model = params_all['p_model']
        # list of stop_words need to be boolean
        if 'vect__tf__stop_words' in p_model.keys() and p_model['vect__tf__stop_words']:
            p_model['vect__tf__stop_words'] = stopwords
        if 'vect__tfidf__stop_words' in p_model.keys() and p_model['vect__tfidf__stop_words']:
            p_model['vect__tfidf__stop_words'] = stopwords

        self.p = p_model

        self.nb_classes = params_all['nb_classes']
        self.shape_y = params_all['shape_y']

        self.embedding.load_params(params_all, outdir)

    def model_classif(self, **kwargs):
        """ Abstract method.

            Initialize model architecture according to classifier model
        Return:
            model (tensorflow Model or sklearn Pipeline)
        """
        pass

    def model(self):
        """ Abstract method.

            Initialize model architecture according to embedding method and classifier model
        Return:
            model (tensorflow Model or sklearn Pipeline)
        """

        # model building is not the same between sklearn and Neural Network (NN) models
        if not self.is_NN:

            clf = self.model_classif()

            if self.embedding.name_model in ['tf', 'tf-idf']:
                vect = self.embedding.model()
                pipeline = Pipeline(steps=[('vect', vect), ('clf', clf)])
                pipeline.set_params(**self.p)

            else:
                pipeline = Pipeline(steps=[('clf', clf)])
                pipeline.set_params(**self.p)
            return pipeline
        else:
            x, inp = self.embedding.model()

            x = self.model_classif(x)

            if 'binary' in self.objective:
                out = Dense(1, 'sigmoid')(x)
            elif 'regression' in self.objective:
                out = Dense(self.nb_classes, 'linear')(x)
            else:
                if self.shape_y == 1:
                    out = Dense(self.nb_classes, activation="softmax")(x)
                else:
                    out = Dense(self.nb_classes, activation="sigmoid")(x)

            model = tf.keras.models.Model(inputs=inp, outputs=out)

            if self.embedding.name_model == "transformer":
                optimizer = AdamWeightDecay(learning_rate=self.p['learning_rate'])
            else:
                optimizer = tf.keras.optimizers.Adam(learning_rate=self.p['learning_rate'])

            if 'binary' in self.objective:
                model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            elif 'regression' in self.objective:
                model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
            else:
                if self.shape_y == 1:
                    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
                else:
                    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer,
                                  metrics=['binary_crossentropy'])
            return model

    def fit_optimization(self, x, y, folds, x_val=None, y_val=None):
        """ Apply Hyperopt optimization for the model by optimizing 'scoring' with a time less than time_limit_per_model
            or number of try less than max_trials
            Save and load optimization : Use trials object for the model saved in hyperopt directory
            (use class_optimization.py)
        Args:
            x (List or dict or DataFrame)
            y (DataFrame)
            x_val (List or dict or DataFrame)
            y_val (DataFrame)
            folds (List[tuple]) list with tuple (train_index, val_index)
        """

        # Look for a saved trials object in hyperopt directory :
        if self.apply_logs:
            dir_hyperopt = os.path.join(self.flags_parameters.outdir, "hyperopt")
        elif self.apply_mlflow:
            dir_hyperopt = os.path.join(self.path_mlflow, self.experiment_id, "hyperopt")
        try:  # try to load an already saved trials object, and increase the max
            if self.flags_parameters.apply_optimization:
                trials_step = self.flags_parameters.max_trial_per_model
                if self.flags_parameters.apply_ray:
                    checkpoint_path = os.path.join(dir_hyperopt, "{}.pkl".format(self.name_model_full))
                    trials = pickle.load(open(checkpoint_path, "rb"))
                    trials = trials['hyperopt_trials']
                    logger.info("Found saved Trials! Loading...")
                    max_trials = trials_step
                else:
                    dir_hyperopt_model = os.path.join(dir_hyperopt, self.name_model_full + ".hyperopt")
                    trials = pickle.load(open(dir_hyperopt_model, "rb"))
                    logger.info("Found saved Trials! Loading...")
                    if trials_step != -1:
                        max_trials = len(trials.trials) + trials_step
                    else:
                        max_trials = 1000
                    print(trials.best_trial)
                if trials_step != -1:
                    logger.info(
                        "Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), len(trials.trials) + trials_step, trials_step))
                else:
                    logger.info("Rerunning from {} trials ".format(len(trials.trials)))
            else:
                trials = None
                max_trials = 1
        except:  # class Optimiz_hyperopt will create a new trials object and start searching
            trials = None
            max_trials = self.flags_parameters.max_trial_per_model
            if not self.flags_parameters.apply_ray and max_trials == -1:
                max_trials = 1000

        # there are two different optimizations depending on whether the model is a NN or another models
        self.optimiz = Optimiz_hyperopt(self, self.hyper_params(self.flags_parameters.size_params),
                                        self.flags_parameters.apply_optimization)
        self.optimiz.train(x, y, folds, x_val, y_val, self.flags_parameters.scoring,
                           self.flags_parameters.max_run_time_per_model, trials, max_trials)

        # Save Trials Hyperopt:
        if not self.flags_parameters.apply_ray:
            if self.apply_logs:
                dir_hyperopt = os.path.join(self.flags_parameters.outdir, "hyperopt")
                os.makedirs(dir_hyperopt, exist_ok=True)
                with open(os.path.join(dir_hyperopt, self.name_model_full + ".hyperopt"), "wb") as f:
                    pickle.dump(self.optimiz.trials, f)
            if self.apply_mlflow:
                dir_hyperopt = os.path.join(self.path_mlflow, self.experiment_id, "hyperopt")
                os.makedirs(dir_hyperopt, exist_ok=True)
                with open(os.path.join(dir_hyperopt, self.name_model_full + ".hyperopt"), "wb") as f:
                    pickle.dump(self.optimiz.trials, f)

        # get information from Optimization :
        self.best_params = self.optimiz.best_params()
        self.best_cv_score = self.optimiz.best_score()
        # self.best_model = self.optimiz.best_estimator()
        self.df_all_results = self.optimiz.get_summary()
        self.df_all_results['model'] = self.name_model_full
        # self.optimiz.show_distribution_score()

    def validation(self, model, x_train, y_train, folds, x_val=None, y_val=None):
        """ Apply validation/cross-validation for the model on (x_train,y_train)
            (use validation.py)
        Args:
              model (self.model) function of the model architecture not instantiated : self.model and not self.model()
              x_train (List or dict or DataFrame)
              y_train (Dataframe)
              x_val (List or dict or DataFrame)
              y_val (Dataframe)
              folds (List[tuple]) list with tuple (train_index, val_index)
        """

        val = Validation(self.objective, self.seed, self.is_NN, self.embedding.name_model, self.name_model_full,
                         self.class_weight, self.average_scoring, self.apply_mlflow, self.experiment_name,
                         self.apply_logs)
        if self.flags_parameters.path_data_validation != 'empty':
            val.fit(model, x_train, y_train, folds, x_val, y_val, self.flags_parameters.cv_strategy,
                    self.flags_parameters.scoring, self.flags_parameters.outdir, self.params_all,
                    self.flags_parameters.batch_size, self.flags_parameters.patience, self.flags_parameters.epochs)
        else:
            if "epochs" in self.p.keys():
                epochs = self.p["epochs"]
            else:
                epochs = self.flags_parameters.epochs
            val.fit_no_val(model, x_train, y_train, self.flags_parameters.scoring, self.flags_parameters.outdir,
                           self.params_all, self.flags_parameters.batch_size, epochs)

        # store information from validation in self.info_scores :
        self.info_scores['fold_id'], self.info_scores['oof_val'] = val.get_cv_prediction()

        if 'binary' in self.objective:
            self.info_scores['accuracy_train'], self.info_scores['f1_train'], self.info_scores['recall_train'], \
            self.info_scores['precision_train'], self.info_scores['roc_auc_train'] = val.get_train_scores()
            self.info_scores['accuracy_val'], self.info_scores['f1_val'], self.info_scores['recall_val'], \
            self.info_scores['precision_val'], self.info_scores['roc_auc_val'] = val.get_scores()
            self.info_scores['fpr'], self.info_scores['tpr'] = val.get_roc()
        elif 'multi-class' in self.objective:
            self.info_scores['accuracy_train'], self.info_scores['f1_' + self.average_scoring + '_train'], self.info_scores[
                'recall_' + self.average_scoring + '_train'], self.info_scores[
                'precision_' + self.average_scoring + '_train'] = val.get_train_scores()
            self.info_scores['accuracy_val'], self.info_scores['f1_' + self.average_scoring + '_val'], self.info_scores[
                'recall_' + self.average_scoring + '_val'], self.info_scores[
                'precision_' + self.average_scoring + '_val'] = val.get_scores()
        elif 'regression' in self.objective:
            self.info_scores['mse_train'], self.info_scores['rmse_train'], self.info_scores['explained_variance_train'], \
            self.info_scores['r2_train'] = val.get_scores()
            self.info_scores['mse_val'], self.info_scores['rmse_val'], self.info_scores['explained_variance_val'], \
            self.info_scores['r2_val'] = val.get_scores()

    def prediction(self, x_test_before_copy=None, y_test_before_copy=None, doc_spacy_data_test=[],
                   name_logs='last_logs', loaded_models=None):
        """ Apply prediction for the model on (x_test,) or (x_test,y_test)
            Models are loaded from the outdir/name_logs/name_embedding/name_model_full directory
            Average all folds prediction of a name_model to get final prediction
            (use prediction.py)
        Args:
            x_test_before_copy (List or dict or DataFrame)
            y_test_before_copy (Dataframe)
            doc_spacy_data_test (List[spacy object])
            name_logs ('last_logs' or 'best_logs')
            loaded_models (Sklearn or TF model) model already loaded
        """

        if x_test_before_copy is not None:
            x_test = x_test_before_copy.copy()

        # get path of models :
        has_saved_model = False
        if self.apply_logs:
            if self.embedding.name_model == "transformer":
                outdir_model = os.path.join(self.flags_parameters.outdir, name_logs, self.embedding.name_model,
                                            self.name_model_full)
            else:
                outdir_model = os.path.join(self.flags_parameters.outdir, name_logs, self.embedding.name_model,
                                            self.name_model_full.split('+')[1])
            # get path of model folds :
            try:
                model_fold_paths = glob(outdir_model + '/fold*')
                if len(model_fold_paths) > 0:
                    has_saved_model = True
            except FileNotFoundError:
                logger.critical(
                    "Didn't find checkpoint model for {} in '{}'".format(self.name_model_full, outdir_model))
        if self.apply_mlflow:
            path_mlflow_experiment_id = os.path.join(self.path_mlflow, self.experiment_id)
            for i, dir_run in enumerate(os.listdir(path_mlflow_experiment_id)):
                if os.path.exists(os.path.join(path_mlflow_experiment_id, dir_run, "tags")):
                    file1 = open(os.path.join(path_mlflow_experiment_id, dir_run, "tags", "mlflow.runName"), 'r')
                    Lines = file1.readlines()
                    if Lines[0] == self.name_model_full:
                        has_saved_model = True
                        break

        if self.apply_app and loaded_models is None:
            outdir_model = self.flags_parameters.outdir
            # get path of model folds :
            try:
                model_fold_paths = glob(outdir_model + '/' + '*.joblib') + glob(outdir_model + '/' + '*.hdf5')
                if len(model_fold_paths) > 0:
                    has_saved_model = True
            except FileNotFoundError:
                logger.critical(
                    "Didn't find checkpoint model for {} in '{}'".format(self.name_model_full, outdir_model))
        elif self.apply_app and loaded_models is not None:
            has_saved_model = True

        if y_test_before_copy is not None:
            y_test = y_test_before_copy.copy()
        else:
            y_test = None

        if has_saved_model:
            pred = Prediction(self.objective, self.embedding.name_model, self.name_model_full, self.flags_parameters.outdir,
                              name_logs, self.is_NN, self.average_scoring, self.apply_mlflow, self.experiment_name,
                              self.apply_logs, self.apply_app)

            # preprocess text on x_test :
            if self.embedding.name_model not in ['tf', 'tf-idf']:
                x_test = self.embedding.preprocessing_transform(x_test)
            else:
                x_test_preprocessed = self.embedding.preprocessing_transform(x_test, doc_spacy_data_test)
                if isinstance(x_test_preprocessed, dict):
                    x_test = x_test_preprocessed
                else:
                    x_test[x_test.columns[self.column_text]] = x_test_preprocessed

            # use label map if labels are not numerics
            if y_test is not None:
                reverse_map_label = None
                if self.flags_parameters.map_label != {}:
                    if y_test[y_test.columns[0]].iloc[0] in self.flags_parameters.map_label.keys():
                        y_test[y_test.columns[0]] = y_test[y_test.columns[0]].map(self.flags_parameters.map_label)
                        if y_test[y_test.columns[0]].isnull().sum() > 0:
                            logger.error("Unknown label name during map of test labels")
                    reverse_map_label = {v: k for k, v in self.flags_parameters.map_label.items()}

            # Init model architecture : (need for tensorflow model because we only save model weights)
            if not (self.apply_app and loaded_models is not None):
                model = self.model()
            else:
                model = None

            pred.fit(model, x_test, y_test, loaded_models)

            # store information from prediction in self.info_scores :
            self.info_scores['prediction'] = pred.get_prediction()

            if y_test is not None:
                if 'binary' in self.objective:
                    self.info_scores['accuracy_test'], self.info_scores['f1_test'], self.info_scores['recall_test'], \
                    self.info_scores['precision_test'], self.info_scores['roc_auc_test'] = pred.get_scores()
                elif 'multi-class' in self.objective:
                    self.info_scores['accuracy_test'], self.info_scores['f1_' + self.average_scoring + '_test'], \
                    self.info_scores['recall_' + self.average_scoring + '_test'], \
                    self.info_scores['precision_' + self.average_scoring + '_test'] = pred.get_scores()
                elif 'regression' in self.objective:
                    self.info_scores['mse_test'], self.info_scores['rmse_test'], self.info_scores[
                        'explained_variance_test'], self.info_scores['r2_test'] = pred.get_scores()

        # if no model provided in model directory, fill null information in self.info_scores :
        else:
            self.info_scores['prediction'] = np.zeros((1, 1))
            if y_test is not None:
                self.info_scores['prediction'] = np.zeros(y_test.shape)
                if 'binary' in self.objective:
                    self.info_scores['accuracy_test'], self.info_scores['f1_test'], self.info_scores['recall_test'], \
                    self.info_scores['precision_test'], self.info_scores['roc_auc_test'] = 0.0, 0.0, 0.0, 0.0, 0.0
                elif 'multi-class' in self.objective:
                    self.info_scores['accuracy_test'], self.info_scores['f1_' + self.average_scoring + '_test'], \
                    self.info_scores['recall_' + self.average_scoring + '_test'], \
                    self.info_scores['precision_' + self.average_scoring + '_test'] = 0.0, 0.0, 0.0, 0.0
                elif 'regression' in self.objective:
                    self.info_scores['mse_test'], self.info_scores['rmse_test'], self.info_scores[
                        'explained_variance_test'], self.info_scores['r2_test'] = 0.0, 0.0, 0.0, 0.0

    def autonlp(self, x_train_before, y_train=None, x_val_before=None, y_val=None,
                apply_optimization=True, apply_validation=True, method_embedding={},
                doc_spacy_data_train=[], doc_spacy_data_val=[], folds=None):
        """ Apply fit_optimization and validation on the best model from hyperopt optimization if apply_validation
            is True else on model parameters from self.flags_parameters.path_models_parameters
        Args:
            x_train_before (Dataframe)
            y_train (Dataframe)
            x_val_before (Dataframe)
            y_val (Dataframe)
            apply_optimization (Boolean)
            apply_validation (Boolean)
            method_embedding (str) name of embedding method or path of embedding weights
            doc_spacy_data_train (List[spacy object])
            doc_spacy_data_val (List[spacy object])
            folds (List[tuple]) list with tuple (train_index, val_index)
        """

        x_train = x_train_before.copy()
        if x_val_before is not None:
            x_val = x_val_before.copy()
        else:
            x_val = None

        self.method_embedding = method_embedding

        # use a model from self.flags_parameters.path_models_parameters if do not want to apply Optimization
        if not apply_optimization:
            logger.info("Load parameters from models_parameters path...")
            try:
                with open(self.flags_parameters.path_models_parameters) as json_file:
                    params_all = json.load(json_file)
                params_all = params_all[self.name_model_full]
                self.load_params(params_all, os.path.dirname(self.flags_parameters.path_models_parameters))  # dirname!
                self.embedding.load_params(params_all, os.path.dirname(self.flags_parameters.path_models_parameters))
            except:
                if self.apply_logs:
                    path_models_parameters = os.path.join(self.flags_parameters.outdir, "models_best_parameters.json")
                elif self.apply_mlflow:
                    path_models_parameters = os.path.join(self.path_mlflow, self.experiment_id,
                                                          "models_best_parameters.json")
                try:
                    with open(path_models_parameters) as json_file:
                        params_all = json.load(json_file)
                    logger.info(
                        "apply_optimization is False and models_parameters path isn't provided, use best model parameters from {}.".format(
                            path_models_parameters))
                    params_all = params_all[self.name_model_full]
                    self.load_params(params_all, os.path.dirname(path_models_parameters))
                    self.embedding.load_params(params_all, os.path.dirname(path_models_parameters))
                except:
                    logger.error(
                        "Did not find name model : {} in '{}', Random parameters from Parameters optimization are used".format(
                            self.name_model_full, self.flags_parameters.path_models_parameters))
                    apply_optimization = True

        # preprocess text on x_train :
        if self.embedding.name_model not in ['tf', 'tf-idf']:
            x_train = self.embedding.preprocessing_fit_transform(x_train, self.flags_parameters.size_params,
                                                                 self.method_embedding)
            if x_val is not None:
                x_val = self.embedding.preprocessing_transform(x_val)
        else:
            x_train_preprocessed = self.embedding.preprocessing_fit_transform(x_train, doc_spacy_data_train,
                                                                              self.method_embedding)
            if x_val is not None:
                x_val_preprocessed = self.embedding.preprocessing_transform(x_val, doc_spacy_data_val)
            if isinstance(x_train_preprocessed, dict):
                x_train = x_train_preprocessed
                if x_val is not None:
                    x_val = x_val_preprocessed
            else:
                x_train[x_train.columns[self.column_text]] = x_train_preprocessed
                if x_val is not None:
                    x_val[x_val.columns[self.column_text]] = x_val_preprocessed

        ###############
        # Optimization
        ###############
        if apply_optimization:
            logger.info("- Optimization of parameters:")
            start = time.perf_counter()
            self.fit_optimization(x_train, y_train, folds, x_val, y_val)
            logger.info('Time search : {}'.format(time.perf_counter() - start))

            self.initialize_params(y_train, self.best_params)

        # save params in path : 'outdir/last_logs/name_embedding/name_model_full'
        if self.apply_logs:
            outdir_embedding = os.path.join(self.flags_parameters.outdir, 'last_logs', self.embedding.name_model)
            os.makedirs(outdir_embedding, exist_ok=True)
            if self.embedding.name_model == "transformer":
                outdir_model = os.path.join(outdir_embedding, self.name_model_full)
            else:
                outdir_model = os.path.join(outdir_embedding, self.name_model_full.split('+')[1])
            os.makedirs(outdir_model, exist_ok=True)
            self.save_params(outdir_model)
        else:
            self.save_params(None)

        ###############
        # Validation
        ###############
        if apply_validation:
            if self.flags_parameters.path_data_validation == "empty":
                logger.info("\n- Training with no validation:")
            elif self.flags_parameters.path_data_validation == "" or self.flags_parameters.path_data_validation is None:
                logger.info("\n- Training & Cross-Validation:")
            else:
                logger.info("\n- Training & Validation:")
            start = time.perf_counter()
            self.validation(self.model, x_train, y_train, folds, x_val, y_val)
            logger.info('Time validation : {}'.format(time.perf_counter() - start))

        # if no validation applied, fill null information in self.info_scores :
        else:
            self.info_scores['fold_id'], self.info_scores['oof_val'] = np.zeros(y_train.shape[0]), np.zeros(
                y_train.shape)

            if 'binary' in self.objective:
                self.info_scores['accuracy_train'], self.info_scores['f1_train'], self.info_scores['recall_train'], \
                self.info_scores['precision_train'], self.info_scores['roc_auc_train'] = 0.0, 0.0, 0.0, 0.0, 0.0
                self.info_scores['accuracy_val'], self.info_scores['f1_val'], self.info_scores['recall_val'], \
                self.info_scores['precision_val'], self.info_scores['roc_auc_val'] = 0.0, 0.0, 0.0, 0.0, 0.0
                self.info_scores['fpr'], self.info_scores['tpr'] = 0.0, 0.0
            elif 'multi-class' in self.objective:
                self.info_scores['accuracy_train'], self.info_scores['f1_' + self.average_scoring + '_train'], \
                self.info_scores['recall_' + self.average_scoring + '_train'], \
                self.info_scores['precision_' + self.average_scoring + '_train'] = 0.0, 0.0, 0.0, 0.0
                self.info_scores['accuracy_val'], self.info_scores['f1_' + self.average_scoring + '_val'], \
                self.info_scores['recall_' + self.average_scoring + '_val'], \
                self.info_scores['precision_' + self.average_scoring + '_val'] = 0.0, 0.0, 0.0, 0.0
            elif 'regression' in self.objective:
                self.info_scores['mse_train'], self.info_scores['rmse_train'], self.info_scores['explained_variance_train'], \
                self.info_scores['r2_train'] = 0.0, 0.0, 0.0, 0.0
                self.info_scores['mse_val'], self.info_scores['rmse_val'], self.info_scores['explained_variance_val'], \
                self.info_scores['r2_val'] = 0.0, 0.0, 0.0, 0.0
