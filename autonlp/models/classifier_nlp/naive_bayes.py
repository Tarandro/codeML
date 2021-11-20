from ...models.classifier_nlp.trainer import Model
from sklearn.naive_bayes import MultinomialNB
from hyperopt import hp


class Naive_Bayes(Model):
    name_classifier = 'Naive_Bayes'
    dimension_embedding = "doc_embedding"
    is_NN = False

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        Model.__init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight)

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            # parameters['clf__alpha'] = uniform(self.flags_parameters.nb_alpha_min, self.flags_parameters.nb_alpha_max)
            if self.flags_parameters.nb_alpha_min == self.flags_parameters.nb_alpha_max:
                parameters['clf__alpha'] = hp.choice('clf__alpha', [self.flags_parameters.nb_alpha_min])
            else:
                parameters['clf__alpha'] = hp.uniform('clf__alpha', self.flags_parameters.nb_alpha_min,
                                                      self.flags_parameters.nb_alpha_max)
        else:
            # parameters['clf__alpha'] = uniform(self.flags_parameters.nb_alpha_min, self.flags_parameters.nb_alpha_max)
            if self.flags_parameters.nb_alpha_min == self.flags_parameters.nb_alpha_max:
                parameters['clf__alpha'] = hp.choice('clf__alpha', [self.flags_parameters.nb_alpha_min])
            else:
                parameters['clf__alpha'] = hp.uniform('clf__alpha', self.flags_parameters.nb_alpha_min,
                                                      self.flags_parameters.nb_alpha_max)

        if self.embedding.name_model in ['tf', 'tf-idf']:
            parameters_embedding = self.embedding.hyper_params()
            parameters.update(parameters_embedding)

        return parameters

    def model_classif(self):
        clf = MultinomialNB()
        return clf