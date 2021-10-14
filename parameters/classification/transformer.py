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

    "embedding": {"tf": None, "tf-idf": None, "word2vec": None, "fasttext": None, "doc2vec": None, "transformer": 1},

    "classifier": {"Naive_Bayes": [], "Logistic_Regression": [], "SGD_Classifier": [1, 2], "SGD_Regressor": [],
                   "XGBoost": [], "Global_Average": [1, 2], "Attention": [1,2], "BiRNN": [], "BiRNN_Attention": [],
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
    "method_embedding": {'Transformer': 'CamemBERT'},

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

flags_dict_transformer = {
    "tr_maxlen": 258,
    "tr_learning_rate": [2e-5]
}

flags = Flags().update(flags_dict_info)
flags = flags.update(flags_dict_preprocessing)
flags = flags.update(flags_dict_autonlp)
flags = flags.update(flags_dict_display)
flags = flags.update(flags_dict_transformer)