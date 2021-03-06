from autonlp.autonlp import AutoNLP
from autonlp.flags import Flags


#####################
# Parameters
#####################

flags_dict_info = {
    "debug": False,  # for debug : use only 50 data rows for training
    "path_data": "C:/Users/agassmann/Documents/data/FinancialPhraseBank.csv",
    "path_data_validation": "",
    "apply_logs": True,
    "outdir": "./logs",
    "apply_mlflow": False,
    "experiment_name": "AutoNLP_3",
    "seed": 15,
    "apply_app": False,

    "apply_ray": False,
    "ray_max_model_parallel": 1,
    "ray_cpu_per_model": 8
}
flags_dict_preprocessing = {
    "column_text": "text_fr",  # name column with texts
    "target": "sentiment",
    "language_text": "fr",
    "apply_small_clean": True,
    "name_spacy_model": "fr_core_news_md",  # en_core_web_md
    "apply_spacy_preprocessing": True,
    "apply_entity_preprocessing": True
}

flags_dict_autonlp = {
    "objective": 'multi-class',    # 'binary' or 'multi-class' or 'regression'
    "embedding": {"tf": 1, "tf-idf": None, "word2vec": None, "fasttext": None, "doc2vec": None, "transformer": None},

    "classifier": {"Naive_Bayes": [], "Logistic_Regression": [], "SGD_Classifier": [1],
                   "XGBoost": [], "Global_Average": [], "Attention": [], "BiRNN": [], "BiRNN_Attention": [],
                   "biLSTM": [], "BiLSTM_Attention": [], "biGRU": [], "BiGRU_Attention": []},

    "max_run_time_per_model": 20,
    "max_trial_per_model": -1,
    "frac_trainset": 0.7,
    "scoring": 'f1',
    "average_scoring": "macro",
    "nfolds": 5,
    "nfolds_train": 2,
    "class_weight": True,
    "apply_blend_model": True,
    "verbose": 2,
    "method_embedding": {'Word2vec': 'Word2Vec',
                         'Fasttext': 'FastText',
                         'Doc2Vec': 'Doc2Vec',
                         'Transformer': 'CamemBERT',
                         'spacy': [('all', False)]},

    "apply_optimization": True,
    "apply_validation": True,
    "path_models_parameters": None,
    "path_models_best_parameters": None
}

flags_dict_display = {
    "sort_leaderboard": 'f1'
}

flags = Flags().update(flags_dict_info)
flags = flags.update(flags_dict_preprocessing)
flags = flags.update(flags_dict_autonlp)
flags = flags.update(flags_dict_display)
print("flags :", flags)
debug = flags.debug

# mlflow : (mlflow ui --backend-store-uri ./mlruns)

# run docker compose : docker-compose up -d --build
# docker streamlit : http://host.docker.internal:8501

if __name__ == '__main__':
    #####################
    # AutoNLP
    #####################

    autonlp = AutoNLP(flags)

    #####################
    # Preprocessing
    #####################

    autonlp.data_preprocessing()

    autonlp.X_train.to_csv('./results/X_train.csv', index=False)
    autonlp.X_test.to_csv('./results/X_test.csv', index=False)
    autonlp.Y_train.to_csv('./results/Y_train.csv', index=False)
    autonlp.Y_test.to_csv('./results/Y_test.csv', index=False)

    #####################
    # Training
    #####################

    autonlp.train()

    #####################
    # Leaderboard (Train score)
    #####################

    leaderboard_train = autonlp.get_leaderboard(sort_by=flags.sort_leaderboard, dataset='train')
    print('\nTrain Leaderboard')
    print(leaderboard_train)
    # autonlp.save_scores_plot(leaderboard_val, 'last_logs')
    leaderboard_train.to_csv('./results/leaderboard_train.csv', index=False)

    #####################
    # Leaderboard (Validation score)
    #####################

    leaderboard_val = autonlp.get_leaderboard(sort_by=flags.sort_leaderboard, dataset='val')
    print('\nValidation Leaderboard')
    print(leaderboard_val)
    #autonlp.save_scores_plot(leaderboard_val, 'last_logs')
    leaderboard_val.to_csv('./results/leaderboard_val.csv', index=False)

    autonlp.correlation_models()

    df_all_results = autonlp.get_df_all_results()
    df_all_results.to_csv('./results/df_all_results.csv', index=False)

    if len(df_all_results) > 0:
        df_all_results_mean = df_all_results.groupby('model').mean().sort_values('mean_test_score', ascending=False)
        print('\nGridSearch information Leaderboard')
        print(df_all_results_mean)
        df_all_results.to_csv('./results/df_all_results_mean.csv', index=False)
    # autonlp.show_distribution_scores()

    # df_oof_val = autonlp.Y_train.copy()
    # for name in autonlp.models.keys():
    #     df_oof_val[name] = np.argmax(autonlp.models[name].info_scores['oof_val'], axis=1).reshape(-1)
    # df_oof_val.to_csv('./results/df_oof_val.csv', index=False)

    if 'binary' in autonlp.objective:
        autonlp.get_roc_curves()

    #####################
    # Testing
    #####################

    on_test_data = True
    name_logs = 'last_logs'
    autonlp.leader_predict(name_logs=name_logs, on_test_data=on_test_data)

    df_prediction = autonlp.dataframe_predictions
    df_prediction.to_csv('./results/df_prediction.csv', index=False)

    leaderboard_test = autonlp.get_leaderboard(sort_by=flags.sort_leaderboard, dataset='test')
    print('\nTest Leaderboard')
    print(leaderboard_test)
    #autonlp.save_scores_plot(leaderboard_test, 'last_logs')
    leaderboard_test.to_csv('./results/leaderboard_test.csv', index=False)

    autonlp.launch_to_model_deployment('tf-idf+SGD_Classifier')

if False:
    import pandas as pd
    data_test = pd.read_csv(flags.path_data)

    X_test, doc_spacy_data_test, Y_test = autonlp.preprocess_test_data(data_test)

    print(X_test)
    print(Y_test)

    name_logs = 'best_logs'
    on_test_data = False
    autonlp.leader_predict(name_logs=name_logs, on_test_data=on_test_data, x=X_test, y=Y_test,
                           doc_spacy_data_test=doc_spacy_data_test)

    leaderboard_test = autonlp.get_leaderboard(sort_by=flags.sort_leaderboard, dataset='test',
                                               info_models=autonlp.info_models)
    print('\nTest Leaderboard')
    print(leaderboard_test)
