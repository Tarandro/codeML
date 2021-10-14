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

    "embedding": {"tf": 1, "tf-idf": 2, "word2vec": None, "fasttext": None, "doc2vec": None, "transformer": 3},

    "dimension_embedding": "doc_embedding",

    "frac_trainset": 0.7,

    "verbose": 2,
    "method_embedding": {'Word2vec': 'Word2Vec',
                         'Fasttext': 'FastText',
                         'Doc2Vec': 'Doc2Vec',
                         'Transformer': 'sentence-bert',
                         'spacy': [('all', False)]},
}

### Embedding parameters
flags_dict_tf = {
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

flags_dict_word2vec = {
    # if you want to create a scratch model:
    "w2v_size_vector": 300,
    "w2v_window": 5,
    "w2v_epochs": 10,
    "w2v_sg": 0,
    "w2v_maxlen": 250,
    "w2v_max_features": 20000
}

flags_dict_fasttext = {
    # if you want to create a scratch model:
    "ft_size_vector": 300,
    "ft_window": 5,
    "ft_epochs": 10,
    "ft_thr_grams": 10,
    "ft_sg": 0,
    "ft_maxlen": 250,
    "ft_max_features": 20000
}

flags_dict_doc2vec = {
    # if you want to create a scratch model:
    "d2v_size_vector": 300,
    "d2v_window": 5,
    "d2v_epochs": 10,
    "d2v_sg": 0,
    "d2v_maxlen": 250,
    "d2v_max_features": 20000
}

flags_dict_transformer = {
    "tr_maxlen": 258
}

flags = Flags().update(flags_dict_info)
flags = flags.update(flags_dict_preprocessing)
flags = flags.update(flags_dict_autonlp)
flags = flags.update(flags_dict_tf)
flags = flags.update(flags_dict_tfidf)
flags = flags.update(flags_dict_word2vec)
flags = flags.update(flags_dict_fasttext)
flags = flags.update(flags_dict_doc2vec)
flags = flags.update(flags_dict_transformer)