from ...models.classifier_nlp.trainer import Model
from sklearn.linear_model import LogisticRegression
from hyperopt import hp
import numpy as np


class Logistic_Regression(Model):
    name_classifier = 'Logistic_Regression'
    dimension_embedding = "doc_embedding"
    is_NN = False

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        Model.__init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight)

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            # parameters['clf__C'] = loguniform(self.flags_parameters.logr_C_min, self.flags_parameters.logr_C_max)
            # parameters['clf__penalty'] = self.flags_parameters.logr_penalty
            if self.flags_parameters.logr_C_min == self.flags_parameters.logr_C_max:
                parameters['clf__C'] = hp.choice('clf__C', [self.flags_parameters.logr_C_min])
            else:
                parameters['clf__C'] = hp.loguniform('clf__C', np.log(self.flags_parameters.logr_C_min),
                                                     np.log(self.flags_parameters.logr_C_max))
            parameters['clf__penalty'] = hp.choice('clf__penalty', self.flags_parameters.logr_penalty)
        else:
            # parameters['clf__C'] = loguniform(self.flags_parameters.logr_C_min, self.flags_parameters.logr_C_max)
            # parameters['clf__penalty'] = self.flags_parameters.logr_penalty  # ['l2', 'l1', 'elasticnet', 'None']
            # parameters['clf__max__iter'] = randint(50, 150)
            if self.flags_parameters.logr_C_min == self.flags_parameters.logr_C_max:
                parameters['clf__C'] = hp.choice('clf__C', [self.flags_parameters.logr_C_min])
            else:
                parameters['clf__C'] = hp.loguniform('clf__C', np.log(self.flags_parameters.logr_C_min),
                                                     np.log(self.flags_parameters.logr_C_max))
            parameters['clf__penalty'] = hp.choice('clf__penalty', self.flags_parameters.logr_penalty)
            parameters['clf__max__iter'] = hp.uniform('clf__max__iter', 50, 150)

        if self.embedding.name_model in ['tf', 'tf-idf']:
            parameters_embedding = self.embedding.hyper_params()
            parameters.update(parameters_embedding)

        return parameters

    def model_classif(self):
        clf = LogisticRegression(
            random_state=self.seed,
            class_weight=self.class_weight,
            solver="saga"
        )
        return clf