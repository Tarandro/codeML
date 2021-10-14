from autonlp.flags import Flags

#####################
# Parameters
#####################

flags_dict_info = {
    "path_data": "C:/Users/agassmann/Documents/data/FinancialPhraseBank.csv",
    "path_data_validation": "",
    "apply_logs": True,
    "outdir": "./logs",
    "seed": 15,
    "debug": False  # for debug : use only 50 data rows for training
}

flags_dict_preprocessing = {
    "column_text": "text_fr",  # name column with texts
    "target": "sentiment",
    "language_text": "fr",   # en
    "apply_small_clean": True,
    "name_spacy_model": "fr_core_news_md",  # en_core_web_md
    "apply_spacy_preprocessing": True,
    "apply_entity_preprocessing": True
}

flags_dict_autonlp = {
    "objective": 'multi-class',    # 'binary' or 'multi-class' or 'regression'

    "embedding": {"tf": 1, "tf-idf": 2, "word2vec": None, "fasttext": None, "doc2vec": None, "transformer": None},

    "classifier": {"Naive_Bayes": [1, 2], "Logistic_Regression": [1, 2], "SGD_Classifier": [1, 2], "SGD_Regressor": [],
                   "XGBoost": [1, 2], "Global_Average": [], "Attention": [], "BiRNN": [], "BiRNN_Attention": [],
                   "biLSTM": [], "BiLSTM_Attention": [], "biGRU": [], "BiGRU_Attention": []},

    "max_run_time_per_model": 100,
    "max_trial_per_model": -1,

    "frac_trainset": 0.7,
    "scoring": 'f1',
    "average_scoring": "macro",

    "nfolds": 5,
    "nfolds_train": 5,
    "cv_strategy": "KFold",

    "class_weight": True,
    "apply_blend_model": True,
    "verbose": 2,
    "method_embedding": {'spacy': [('all', False)]},

    "apply_optimization": True,
    "apply_validation": True,
}

flags_dict_display = {
    "sort_leaderboard": 'f1'
}

### Only for Neural Network NN
flags_dict_nn_params = {
    "batch_size": 16,
    "patience": 4,
    "epochs": 60,
    "min_lr": 1e-4
}

### Embedding parameters
flags_dict_tf = {
    "tf_binary": True,
    "tf_ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tf_stop_words": True,

    # TF word matrix
    # use an unique TF matrix to get word vectors in order to use it for embedding
    "tf_wde_binary": False,
    "tf_wde_stop_words": True,
    "tf_wde_ngram_range": (1, 1),
    "tf_wde_vector_size": 200,
    "tf_wde_max_features": 20000,
    "tf_wde_maxlen": 250,
    "tf_wde_learning_rate": [1e-3]
}

flags_dict_tfidf = {
    "tfidf_binary": False,
    "tfidf_ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf_stop_words": True,

    # TF-IDF word matrix
    # use an unique TF-IDF matrix to get word vectors in order to use it for embedding
    "tfidf_wde_binary": False,
    "tfidf_wde_stop_words": True,
    "tfidf_wde_ngram_range": (1, 1),
    "tfidf_wde_vector_size": 200,
    "tfidf_wde_max_features": 20000,
    "tfidf_wde_maxlen": 250,
    "tfidf_wde_learning_rate": [1e-3]
}

flags = Flags().update(flags_dict_info)
flags = flags.update(flags_dict_preprocessing)
flags = flags.update(flags_dict_autonlp)
flags = flags.update(flags_dict_display)
flags = flags.update(flags_dict_tf)
flags = flags.update(flags_dict_tfidf)