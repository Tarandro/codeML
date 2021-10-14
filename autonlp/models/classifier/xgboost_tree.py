from ...models.classifier.trainer import Model
from hyperopt import hp
import xgboost as xgb


class XGBoost(Model):
    name_classifier = 'XGBoost'
    dimension_embedding = "doc_embedding"
    is_NN = False

    def __init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight=None):
        Model.__init__(self, flags_parameters, embedding, name_model_full, column_text, class_weight)

    def hyper_params(self, size_params='small'):
        parameters = dict()
        if size_params == 'small':
            # self.parameters = dict(n_estimators = randint(20,200), #[75, 100, 125, 150],
            #                                    max_depth = randint(3,10),    #[7,8,10,20,30]
            #                                    learning_rate = uniform(0.04,0.3),
            #                                   subsample = uniform(0.5,0.5))
            if self.flags_parameters.xgb_n_estimators_min == self.flags_parameters.xgb_n_estimators_max:
                parameters['clf__n_estimators'] = hp.choice('clf__n_estimators', [self.flags_parameters.xgb_n_estimators_min])
            else:
                parameters['clf__n_estimators'] = hp.choice('clf__n_estimators', [i for i in range(
                    self.flags_parameters.xgb_n_estimators_min, self.flags_parameters.xgb_n_estimators_max+1)])
            if self.flags_parameters.xgb_max_depth_min == self.flags_parameters.xgb_max_depth_max:
                parameters['clf__max_depth'] = hp.choice('clf__max_depth', [self.flags_parameters.xgb_max_depth_min])
            else:
                parameters['clf__max_depth'] = hp.choice('clf__max_depth', [i for i in range(
                    self.flags_parameters.xgb_max_depth_min, self.flags_parameters.xgb_max_depth_max+1)])
            if self.flags_parameters.xgb_learning_rate_min == self.flags_parameters.xgb_learning_rate_max:
                parameters['clf__learning_rate'] = hp.choice('clf__learning_rate',
                                                             [self.flags_parameters.xgb_learning_rate_min])
            else:
                parameters['clf__learning_rate'] = hp.uniform('clf__learning_rate',
                                                              self.flags_parameters.xgb_learning_rate_min,
                                                              self.flags_parameters.xgb_learning_rate_max)
            if self.flags_parameters.xgb_subsample_min == self.flags_parameters.xgb_subsample_max:
                parameters['clf__subsample'] = hp.choice('clf__subsample', [self.flags_parameters.xgb_subsample_min])
            else:
                parameters['clf__subsample'] = hp.uniform('clf__subsample', self.flags_parameters.xgb_subsample_min,
                                                          self.flags_parameters.xgb_subsample_max)
        else:
            if self.flags_parameters.xgb_n_estimators_min == self.flags_parameters.xgb_n_estimators_max:
                parameters['clf__n_estimators'] = hp.choice('clf__n_estimators',
                                                            [self.flags_parameters.xgb_n_estimators_min])
            else:
                parameters['clf__n_estimators'] = hp.choice('clf__n_estimators', [i for i in range(
                    self.flags_parameters.xgb_n_estimators_min, self.flags_parameters.xgb_n_estimators_max + 1)])
            if self.flags_parameters.xgb_max_depth_min == self.flags_parameters.xgb_max_depth_max:
                parameters['clf__max_depth'] = hp.choice('clf__max_depth', [self.flags_parameters.xgb_max_depth_min])
            else:
                parameters['clf__max_depth'] = hp.choice('clf__max_depth', [i for i in range(
                    self.flags_parameters.xgb_max_depth_min, self.flags_parameters.xgb_max_depth_max + 1)])
            if self.flags_parameters.xgb_learning_rate_min == self.flags_parameters.xgb_learning_rate_max:
                parameters['clf__learning_rate'] = hp.choice('clf__learning_rate',
                                                             [self.flags_parameters.xgb_learning_rate_min])
            else:
                parameters['clf__learning_rate'] = hp.uniform('clf__learning_rate',
                                                              self.flags_parameters.xgb_learning_rate_min,
                                                              self.flags_parameters.xgb_learning_rate_max)
            if self.flags_parameters.xgb_subsample_min == self.flags_parameters.xgb_subsample_max:
                parameters['clf__subsample'] = hp.choice('clf__subsample', [self.flags_parameters.xgb_subsample_min])
            else:
                parameters['clf__subsample'] = hp.uniform('clf__subsample', self.flags_parameters.xgb_subsample_min,
                                                          self.flags_parameters.xgb_subsample_max)

        if self.embedding.name_model in ['tf', 'tf-idf']:
            parameters_embedding = self.embedding.hyper_params()
            parameters.update(parameters_embedding)

        return parameters

    def model_classif(self):
        if 'regression' in self.flags_parameters.objective:
            clf = xgb.XGBRegressor(
                random_state=self.seed
            )
        else:
            clf = xgb.XGBClassifier(
                random_state=self.seed
                # scale_pos_weight = count(negative examples)/count(Positive examples)
            )
        return clf