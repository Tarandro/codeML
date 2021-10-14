from autonlp.flags import Flags

#####################
# Parameters
#####################

### Classifier Sklearn
# Naive Bayes
flags_dict_naive_bayes = {
    "nb_alpha_min": 0.0,
    "nb_alpha_max": 1.0
}

# Logistic Regression
flags_dict_logr = {
    "logr_C_min": 1e-2,
    "logr_C_max": 1e2,
    "logr_penalty": ['l2', 'l1']
}

# SGD Classifier or Regressor
flags_dict_sgd = {
    "sgd_alpha_min": 1e-4,
    "sgd_alpha_max": 1e-2,
    "sgdc_penalty": ['l2', 'l1'],
    "sgdc_loss": ['log', 'modified_huber'], # don't use 'hinge' (can't use predict_proba):
    "sgdr_penalty": ['l2', 'l1'],
    "sgdr_loss": ['squared_loss', 'huber', 'epsilon_insensitive']
}

# XGBoost
flags_dict_xgb = {
    "xgb_n_estimators_min": 20,
    "xgb_n_estimators_max": 200,
    "xgb_max_depth_min": 3,
    "xgb_max_depth_max": 10,
    "xgb_learning_rate_min": 0.04,
    "xgb_learning_rate_max": 0.3,
    "xgb_subsample_min": 0.5,
    "xgb_subsample_max": 1.0
}

### Classifier Neural Network

# GlobalAverage
flags_dict_ga = {
    "ga_dropout_rate_min": 0,
    "ga_dropout_rate_max": 0.5
}

# RNN
flags_dict_rnn = {
    "rnn_hidden_unit_min": 120,
    "rnn_hidden_unit_max": 130,
    "rnn_dropout_rate_min": 0,
    "rnn_dropout_rate_max": 0.5
}

# LSTM
flags_dict_lstm = {
    "lstm_hidden_unit_min": 120,
    "lstm_hidden_unit_max": 130,
    "lstm_dropout_rate_min": 0,
    "lstm_dropout_rate_max": 0.5
}

# GRU
flags_dict_gru = {
    "gru_hidden_unit_min": 120,
    "gru_hidden_unit_max": 130,
    "gru_dropout_rate_min": 0,
    "gru_dropout_rate_max": 0.5
}

# Attention
flags_dict_attention = {
    "att_dropout_rate_min": 0,
    "att_dropout_rate_max": 0.5
}

flags = Flags().update(flags_dict_naive_bayes)
flags = flags.update(flags_dict_logr)
flags = flags.update(flags_dict_sgd)
flags = flags.update(flags_dict_xgb)
flags = flags.update(flags_dict_ga)
flags = flags.update(flags_dict_rnn)
flags = flags.update(flags_dict_lstm)
flags = flags.update(flags_dict_gru)
flags = flags.update(flags_dict_attention)