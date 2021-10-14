from ...models.classifier.trainer import Model
from hyperopt import hp
from tensorflow.keras.layers import Dropout, GlobalAveragePooling1D


class Global_Average(Model):
    name_classifier = 'Global_Average'
    dimension_embedding = "word_embedding"
    is_NN = True

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        Model.__init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight)
        self.batch_size = self.flags_parameters.batch_size
        self.patience = self.flags_parameters.patience
        self.epochs = self.flags_parameters.epochs
        self.min_lr = self.flags_parameters.min_lr

    def hyper_params(self, size_params='small'):
        # Default : self.parameters = {'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)}
        parameters = dict()
        if size_params == 'small':
            if self.flags_parameters.ga_dropout_rate_min == self.flags_parameters.ga_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.ga_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.ga_dropout_rate_min,
                                                             self.flags_parameters.ga_dropout_rate_max)
        else:
            if self.flags_parameters.ga_dropout_rate_min == self.flags_parameters.ga_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.ga_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.ga_dropout_rate_min,
                                                        self.flags_parameters.ga_dropout_rate_max)

        parameters_embedding = self.embedding.hyper_params()
        parameters.update(parameters_embedding)
        return parameters

    def model_classif(self, x):

        x = Dropout(self.p['dropout_rate'])(x)
        x = GlobalAveragePooling1D()(x)
        return x