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

    "embedding": {"tf": None, "tf-idf": None, "word2vec": 1, "fasttext": 2, "doc2vec": None, "transformer": None},

    "classifier": {"Naive_Bayes": [], "Logistic_Regression": [], "SGD_Classifier": [1, 2], "SGD_Regressor": [],
                   "XGBoost": [], "Global_Average": [1, 2], "Attention": [], "BiRNN": [], "BiRNN_Attention": [],
                   "biLSTM": [], "BiLSTM_Attention": [1, 2], "biGRU": [], "BiGRU_Attention": []},

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
    "method_embedding": {'Word2vec': 'Word2Vec',
                         'Fasttext': 'FastText'},

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
flags_dict_word2vec = {
    # if you want to create a scratch model:
    "w2v_size_vector": 300,
    "w2v_window": 5,
    "w2v_epochs": 10,
    "w2v_sg": 0,
    # Training :
    "w2v_maxlen": 250,
    "w2v_max_features": 20000,
    "w2v_learning_rate": [1e-3]
}

flags_dict_fasttext = {
    # if you want to create a scratch model:
    "ft_size_vector": 300,
    "ft_window": 5,
    "ft_epochs": 10,
    "ft_thr_grams": 10,
    "ft_sg": 0,
    # Training :
    "ft_maxlen": 250,
    "ft_max_features": 20000,
    "ft_learning_rate": [1e-3]
}

flags = Flags().update(flags_dict_info)
flags = flags.update(flags_dict_preprocessing)
flags = flags.update(flags_dict_autonlp)
flags = flags.update(flags_dict_display)
flags = flags.update(flags_dict_word2vec)
flags = flags.update(flags_dict_fasttext)