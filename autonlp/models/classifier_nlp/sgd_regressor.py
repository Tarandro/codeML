from ...models.classifier_nlp.trainer import Model
from sklearn.linear_model import SGDRegressor
from hyperopt import hp
import numpy as np


class SGD_Regressor(Model):
    name_classifier = 'SGD_Regressor'
    dimension_embedding = "doc_embedding"
    is_NN = False

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        Model.__init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight)

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            # parameters['clf__alpha'] = loguniform(self.flags_parameters.sgd_alpha_min,
            #                                           self.flags_parameters.sgd_alpha_max)
            # parameters['clf__penalty'] = self.flags_parameters.sgd_penalty
            # parameters['clf__loss'] = self.flags_parameters.sgd_loss
            if self.flags_parameters.sgd_alpha_min == self.flags_parameters.sgd_alpha_max:
                parameters['clf__alpha'] = hp.choice('clf__alpha', [self.flags_parameters.sgd_alpha_min])
            else:
                parameters['clf__alpha'] = hp.loguniform('clf__alpha', np.log(self.flags_parameters.sgd_alpha_min),
                                                         np.log(self.flags_parameters.sgd_alpha_max))
            parameters['clf__penalty'] = hp.choice('clf__penalty', self.flags_parameters.sgdr_penalty)
            parameters['clf__loss'] = hp.choice('clf__loss', self.flags_parameters.sgdr_loss)
        else:
            # parameters['clf__alpha'] = loguniform(self.flags_parameters.sgd_alpha_min,
            #                                           self.flags_parameters.sgd_alpha_max)
            # parameters['clf__penalty'] = self.flags_parameters.sgd_penalty  # ['l2', 'l1' ,'elasticnet']
            # parameters['clf__loss'] = self.flags_parameters.sgd_loss
            if self.flags_parameters.sgd_alpha_min == self.flags_parameters.sgd_alpha_max:
                parameters['clf__alpha'] = hp.choice('clf__alpha', [self.flags_parameters.sgd_alpha_min])
            else:
                parameters['clf__alpha'] = hp.loguniform('clf__alpha', np.log(self.flags_parameters.sgd_alpha_min),
                                                         np.log(self.flags_parameters.sgd_alpha_max))
            parameters['clf__penalty'] = hp.choice('clf__penalty', self.flags_parameters.sgdr_penalty)
            parameters['clf__loss'] = hp.choice('clf__loss', self.flags_parameters.sgdr_loss)

        if self.embedding.name_model in ['tf', 'tf-idf']:
            parameters_embedding = self.embedding.hyper_params()
            parameters.update(parameters_embedding)

        return parameters

    def model_classif(self):
        clf = SGDRegressor(
            random_state=self.seed,
            early_stopping=True
        )
        return clf