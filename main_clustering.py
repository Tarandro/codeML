from autonlp.autonlp import AutoNLP
from autonlp.flags import Flags


#####################
# Parameters
#####################

flags_dict_info = {
    "debug": False,  # for debug : use only 50 data rows for training
    "path_data": "C:/Users/agassmann/Documents/data/data_cross_validation_label.csv",  # FinancialPhraseBank
    "path_data_validation": "",
    "apply_logs": True,
    "outdir": "./logs",
    "apply_mlflow": False,
    "experiment_name": "AutoNLP_3",
    "seed": 15,
    "apply_app": False
}
flags_dict_preprocessing = {
    "column_text": "opinion",  # name column with texts   # text_fr
    "target": "sentiment_true",           # sentiment
    "language_text": "fr",
    "apply_small_clean": True,
    "name_spacy_model": "fr_core_news_md",  # en_core_web_md
    "apply_spacy_preprocessing": True,
    "apply_entity_preprocessing": True
}

flags_dict_autonlp = {
    "objective": 'multi-class',    # 'binary' or 'multi-class' or 'regression'
    "embedding": {"tf": None, "tf-idf": None, "word2vec": None, "fasttext": 1, "doc2vec": None, "transformer": 2},

    "clustering": {"NMF_frobenius": [], "NMF_kullback": [], "LDA": [],
                   "hdbscan": [], "ACP_hdbscan": [], "UMAP_hdbscan": [],
                   "kmeans": [], "ACP_kmeans": [1,2], "UMAP_kmeans": [1,2],
                   "agglomerativeclustering": [], "ACP_agglomerativeclustering": [], "UMAP_agglomerativeclustering": [],
                   "Similarity_voc": [1,2], "zero_shot": []},

    "n_groups": 20,
    "average_scoring": "macro",

    "max_run_time_per_model": 10,
    "frac_trainset": 0.7,
    "scoring": 'f1',
    "nfolds": 5,
    "nfolds_train": 1,
    "class_weight": True,
    "apply_blend_model": True,
    "verbose": 2,
    "method_embedding": {'Word2vec': 'Word2Vec',
                         'Fasttext': 'FastText',
                         'Doc2Vec': 'Doc2Vec',
                         'Transformer': 'sentence-bert',
                         'spacy': [('all', False)]},

    "apply_optimization": True,
    "apply_validation": True,
    "path_models_parameters": None,
    "path_models_best_parameters": None,

    "show_top_terms_topics": True,
    "preprocess_topic": (['ADJ', 'NOUN', 'VERB'], True),
    "n_top_words": 10,
    "min_ngram": 2,
    "max_ngram": 3
}

flags_dict_display = {
    "sort_leaderboard": 'f1'
}

flags_dict_clustering = {
    # NMF
    "alpha_nmf": 0.1,
    "l1_ratio": 0.5,
    # LDA
    "max_iter_lda": 5,
    # ACP
    "acp_n_components": 2,
    # UMAP
    "umap_n_components": 2,
    "umap_n_neighbors": 15,
    # HDBSCAN
    "min_cluster_size": 15,
    # AgglomerativeClustering
    "aglc_linkage": "ward",  # 'ward', 'complete', 'average', 'single'

    ## unsupervised
    #"vocabulary_labels": {"positive": ["positif", "avantage", "remporté", "gagné"],
    #                      "neutral": ["neutre"],
    #                      "negative": ["négatif", "mauvais", "inconvénient", "perdu", "tombé", "baisse"]}
    "vocabulary_labels": {"positif": ["positif", "bien", "agréable", "joli", "avantage", "parfait"],
                          "négatif": ["négatif", "mauvais", "horrible", "moche", "inconvénient"]}
}

flags = Flags().update(flags_dict_info)
flags = flags.update(flags_dict_preprocessing)
flags = flags.update(flags_dict_autonlp)
flags = flags.update(flags_dict_clustering)
flags = flags.update(flags_dict_display)
print("flags :", flags)
debug = flags.debug

# mlflow : (mlflow ui --backend-store-uri ./mlruns)

if __name__ == '__main__':
    #####################
    # AutoNLP
    #####################

    autonlp = AutoNLP(flags)

    #####################
    # Preprocessing
    #####################

    autonlp.data_preprocessing()

    #####################
    # Train
    #####################

    dict_clustering_train = autonlp.fit_transform_clustering()

    #####################
    # Testing
    #####################

    #on_test_data = True
    #name_logs = 'last_logs'
    #dict_clustering_test = autonlp.transform_clustering(name_logs=name_logs, on_test_data=on_test_data)

    import pandas as pd
    data_test = pd.read_csv(flags.path_data)

    X_test, doc_spacy_data_test, Y_test = autonlp.preprocess_test_data(data_test)

    name_logs = 'best_logs'
    on_test_data = False
    dict_clustering_test = autonlp.transform_clustering(name_logs=name_logs, on_test_data=on_test_data,
                                                        x=X_test, y=Y_test,
                                                        doc_spacy_data_test=doc_spacy_data_test)

    leaderboard_test = autonlp.get_leaderboard(sort_by=flags.sort_leaderboard, dataset='test',
                                               info_models=autonlp.info_models)
    print('\nTest Leaderboard')
    print(leaderboard_test)