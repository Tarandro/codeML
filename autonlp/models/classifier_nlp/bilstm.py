from ...models.classifier_nlp.trainer import Model
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.layers import Bidirectional
from hyperopt import hp


class Bilstm(Model):
    name_classifier = 'Bilstm'
    dimension_embedding = "word_embedding"
    is_NN = True

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        Model.__init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight)
        self.batch_size = self.flags_parameters.batch_size
        self.patience = self.flags_parameters.patience
        self.epochs = self.flags_parameters.epochs
        self.min_lr = self.flags_parameters.min_lr

    def hyper_params(self, size_params='small'):
        # Default : parameters = {'hidden_unit': hp.randint('hidden_unit_1', 120, 130),
        #                         'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)}
        parameters = dict()
        if size_params == 'small':
            if self.flags_parameters.lstm_hidden_unit_min == self.flags_parameters.lstm_hidden_unit_max:
                parameters['hidden_unit'] = hp.choice('hidden_unit_1', [self.flags_parameters.lstm_hidden_unit_min])
            else:
                parameters['hidden_unit'] = hp.randint('hidden_unit_1', self.flags_parameters.lstm_hidden_unit_min,
                                                       self.flags_parameters.lstm_hidden_unit_max)
            if self.flags_parameters.lstm_dropout_rate_min == self.flags_parameters.lstm_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.lstm_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.lstm_dropout_rate_min,
                                                        self.flags_parameters.lstm_dropout_rate_max)
        else:
            if self.flags_parameters.lstm_hidden_unit_min == self.flags_parameters.lstm_hidden_unit_max:
                parameters['hidden_unit'] = hp.choice('hidden_unit_1', [self.flags_parameters.lstm_hidden_unit_min])
            else:
                parameters['hidden_unit'] = hp.randint('hidden_unit_1', self.flags_parameters.lstm_hidden_unit_min,
                                                       self.flags_parameters.lstm_hidden_unit_max)
            if self.flags_parameters.lstm_dropout_rate_min == self.flags_parameters.lstm_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.lstm_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.lstm_dropout_rate_min,
                                                        self.flags_parameters.lstm_dropout_rate_max)
        parameters_embedding = self.embedding.hyper_params()
        parameters.update(parameters_embedding)
        return parameters

    def model_classif(self, x):

        x = Bidirectional(LSTM(int(self.p['hidden_unit']), return_sequences=False))(x)
        # x = Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
        x = Dropout(self.p['dropout_rate'])(x)
        return x