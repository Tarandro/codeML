import os
import json
from ...utils.logging import get_logger

logger = get_logger(__name__)


class Base_Embedding:
    """ Parent class of embedding methods """

    def __init__(self, flags_parameters, column_text, dimension_embedding):
        """
        Args:
            flags_parameters : Instance of Flags class object
            column_text (int) : column number with texts
            dimension_embedding (str) : 'word_embedding' or 'doc_embedding'

        From flags_parameters:
            seed (int)
            apply_mlflow (Boolean) save model in self.path_mlflow (str) directory
            experiment_name (str) name of the experiment, only if MLflow is activated
            apply_logs (Boolean) use manual logs
            apply_app (Boolean) if you want to use a model from model_deployment directory
        """
        self.flags_parameters = flags_parameters
        self.apply_mlflow = flags_parameters.apply_mlflow
        self.path_mlflow = "../mlruns"
        self.experiment_name = flags_parameters.experiment_name
        self.apply_logs = flags_parameters.apply_logs
        self.apply_app = flags_parameters.apply_app
        self.seed = flags_parameters.seed
        self.column_text = column_text
        self.dimension_embedding = dimension_embedding
        self.name_model = None

        if self.apply_mlflow:
            import mlflow
            current_experiment = dict(mlflow.get_experiment_by_name(self.experiment_name))
            self.experiment_id = current_experiment['experiment_id']

    def hyper_params(self, size_params='small'):
        """ Abstract method.

            Instantiate hyperparameters range for embedding method that will be use for hyperopt optimization

        Args:
            size_params ('small' or 'big') size of parameters range for optimization
        Return:
            parameters (dict) a hyperopt range for each hyperparameters
        """
        pass

    def preprocessing_fit_transform(self, **kwargs):
        """ Abstract method.
            Fit preprocessing and transform x according to embedding method and dimension embedding """
        pass

    def preprocessing_transform(self, **kwargs):
        """ Abstract method.
            Transform x according to latest fit preprocessing """
        pass

    def save_params(self, outdir_model):
        """ Save all params as a json file needed to reuse the embedding method in outdir_model
        Args:
            outdir_model (str)
        Return:
            params_all (dict)
        """

        params_all = dict()

        params_all['name_embedding'] = self.name_model
        params_all['method_embedding'] = self.method_embedding
        params_all['dimension_embedding'] = self.dimension_embedding
        params_all['language_text'] = self.flags_parameters.language_text

        if self.name_model == "transformer":
            params_all['maxlen'] = self.maxlen
        elif self.dimension_embedding == 'word_embedding':
            params_all['maxlen'] = self.maxlen
            params_all['embed_size'] = self.embed_size
            params_all['max_features'] = self.max_features
            params_all['length_word_index'] = self.length_word_index

        self.params_all = {self.name_model: params_all}

        if self.apply_logs:
            with open(os.path.join(outdir_model, "parameters_embedding.json"), "w") as outfile:
                json.dump(self.params_all, outfile)

        return params_all

    def load_tokenizer(self, outdir):
        """ Abstract method.
            load tokenizer depending on self.method_embedding """
        pass

    def load_params(self, params_all, outdir):
        """ Abstract method.

            Initialize all params from params_all
            + tensorflow tokenizer (pickle file) from outdir path

        Args:
            params_all (dict)
            outdir (str)
        """
        self.method_embedding = params_all['method_embedding']
        self.dimension_embedding = params_all['dimension_embedding']

        if self.name_model == "transformer":
            self.maxlen = params_all['maxlen']
            self.load_tokenizer(outdir)
        elif self.dimension_embedding == 'word_embedding':
            self.maxlen = params_all['maxlen']
            self.embed_size = params_all['embed_size']
            self.max_features = params_all['max_features']
            self.length_word_index = params_all['length_word_index']

            self.load_tokenizer(outdir)

        for name_gensim_model in ['Word2Vec', 'FastText', 'Doc2Vec']:
            if isinstance(self.method_embedding, str) and self.method_embedding.lower() == name_gensim_model.lower():
                dir_gensim = os.path.join(outdir, name_gensim_model)
                if os.path.exists(dir_gensim):
                    self.method_embedding = os.path.join(dir_gensim, "{}.wordvectors".format(name_gensim_model.lower()))
                else:
                    logger.error("A directory {} with the model must be provided in {}".format(name_gensim_model, outdir))