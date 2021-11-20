from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.python.keras.layers import Layer
from tensorflow.keras.layers import Dropout

from ...models.classifier_nlp.trainer import Model
from hyperopt import hp


class Attention(Model):
    name_classifier = 'Attention'
    dimension_embedding = "word_embedding"
    is_NN = True

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        Model.__init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight)
        self.batch_size = self.flags_parameters.batch_size
        self.patience = self.flags_parameters.patience
        self.epochs = self.flags_parameters.epochs
        self.min_lr = self.flags_parameters.min_lr

    def hyper_params(self, size_params='small'):
        # Default : parameters = {'dropout_rate': hp.uniform('dropout_rate', 0, 0.5)}
        parameters = dict()
        if size_params == 'small':
            if self.flags_parameters.att_dropout_rate_min == self.flags_parameters.att_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.att_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.att_dropout_rate_min,
                                                        self.flags_parameters.att_dropout_rate_max)
        else:
            if self.flags_parameters.lstm_dropout_rate_min == self.flags_parameters.att_dropout_rate_max:
                parameters['dropout_rate'] = hp.choice('dropout_rate', [self.flags_parameters.att_dropout_rate_min])
            else:
                parameters['dropout_rate'] = hp.uniform('dropout_rate', self.flags_parameters.att_dropout_rate_min,
                                                        self.flags_parameters.att_dropout_rate_max)
        parameters_embedding = self.embedding.hyper_params()
        parameters.update(parameters_embedding)
        return parameters

    def model_classif(self, x):

        x = Dropout(self.p['dropout_rate'])(x)
        x = Attention_layer(self.embedding.maxlen)(x)
        return x


#################
# Build Attention Layer tensorflow :
#################


class Attention_layer(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                              K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.features_dim