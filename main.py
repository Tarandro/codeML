from autonlp.autonlp import AutoNLP
from autonlp.flags import Flags


#####################
# Parameters
#####################

# todo : arg remove column

flags_dict_info = {
    "debug": False,  # for debug : use only 50 data rows for training
    "path_data": "C:/Users/AlexisGassmann/Documents/code/achats_reactivation_small.csv",
    "path_data_validation": "",
    "apply_logs": True,
    "outdir": "./logs",
    "apply_mlflow": False,
    "experiment_name": "AutoML",
    "seed": 15,
    "apply_app": False,

    "target": ["reactivation_HMU_12M"],
}

flags_dict_autonlp = {
    "objective": 'binary',    # 'binary' or 'multi-class' or 'regression'
    "classifier_ml": {'logistic_regression':2, 'randomforest': 1, 'lightgbm': 3,
                                 'xgboost': None, 'catboost': None, 'dense_network': None, 'lstm': None},
    "max_run_time_per_model": 20,
    "frac_trainset": 0.8,
    "scoring": 'f1',
    "nfolds": 5,
    "nfolds_train": 5,
    "class_weight": False,
    "apply_blend_model": False,
    "verbose": 2,
    "method_embedding": {'Word2vec': 'Word2Vec',
                         'Fasttext': 'FastText',
                         'Doc2Vec': 'Doc2Vec',
                         'Transformer': 'CamemBERT',
                         'spacy': [('all', False), (['ADJ', 'NOUN', 'VERB', 'DET'], False)]},

    "apply_optimization": True,
    "apply_validation": True,
    "path_models_parameters": None,
    "path_models_best_parameters": None
}

flags_dict_ml_preprocessing = {

    #"ordinal_features": ["item_id", "dept_id", "store_id", "cat_id", "state_id", "wday", "month",
    #                     "year", 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'],
    "columns_to_keep": ['tx_reactivite_12M_av_score',
                         'cl_dis_min',
                         'cl_age_client',
                         'delai_inactivite',
                         'flag_enf_p15ans',
                         'cl_nb_tck_12M_av_HMU_CS',
                         'moy_dt_entre_ticket',
                         'NB_ENFANTS',
                         'achat_adulte_12M_av_HMU_CS',
                         'cl_CA_12M_av_HMU_CS',
                         'nb_ticket',
                         'rens_ADR',
                         'corr_depense_time',
                         'flag_enf_15ans_ou_m',
                         'rens_ENF',
                         'seg_multicanal_life',
                         'mean_week_end_day',
                         'cl_anc_first_ach_dda',
                         'nb_produit_aband_6M_av_score',
                         'cl_IDV_12M_av_HMU_CS',
                         'eshop_dern_ticket1',
                         'rens_MOB',
                         'tx_ouverture_12M_av_score',
                         'cl_anc_adh_fid_dda',
                         'nb_panier_aband_6M_av_score',
                         'achat_naiss_bebe_enf_12M_av_HMU_CS'],

    "columns_to_remove": ["ID_SC_INDIV"],
    "normalize": False,
    "method_scaling": 'MinMaxScaler',   # 'MinMaxScaler', 'RobustScaler', 'StandardScaler'
    "type_columns": None,
    "apply_preprocessing_mandatory": True,
    "remove_categorical": False,

    "method_nan_categorical": 'mode',
    "method_nan_numeric": 'mean',
    "subsample": 0.3,
    "feature_interaction": False,
    "feature_ratio": False,
    "polynomial_features": False,
    "remove_multicollinearity": False,
    "multicollinearity_threshold": 0.9,
    "feature_selection": False,
    "feature_selection_threshold": 0.8,
    "bin_numeric_features": [],
    "remove_low_variance": False,
    "remove_percentage": 0.8,
    "info_pca": {}, # {'all':('all',2)},
    "info_tsne": {},
    "info_stats": {}  # {'BalanceSalaryRatio':('div',['Balance','EstimatedSalary']),
                      # 'TenureByAge':('div',['Tenure','Age']),
                      # 'CreditScoreGivenAge':('div',['CreditScore','Age'])}
}

flags_dict_ts_preprocessing = {
    "startDate_train": 'all',  # or int  need to be a continuous numeric column
    "endDate_train": 1920,    # or int
    "position_id": "id",   # can be a dataframe
    "position_date": "d",   # need to be a continuous numeric column
    "size_train_prc": 0.9,
    "time_series_recursive": False,
    "LSTM_date_features": ['wday', 'month', 'year', 'event_name_1', 'event_type_1', 'event_name_2',
                           'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI'],
    "timesteps": 14,
    "step_lags": [], #[1,3,7],
    "step_rolling": [], #[7],
    "win_type": None
}

flags_dict_nlp_preprocessing = {
    "column_text": None,  # name column with texts
    "language_text": "fr",
    "apply_small_clean": True,
    "name_spacy_model": "fr_core_news_md",  # en_core_web_md
    "apply_spacy_preprocessing": True,
    "apply_entity_preprocessing": True
}

flags_dict_display = {
    "sort_leaderboard": 'f1'
}

flags = Flags().update(flags_dict_info)
flags = flags.update(flags_dict_ml_preprocessing)
flags = flags.update(flags_dict_nlp_preprocessing)
flags = flags.update(flags_dict_ts_preprocessing)
flags = flags.update(flags_dict_autonlp)
flags = flags.update(flags_dict_display)
print("flags :", flags)
debug = flags.debug

# mlflow : (mlflow ui --backend-store-uri ./mlruns)

if __name__ == '__main__':
    #####################
    # AutoML
    #####################

    autonlp = AutoNLP(flags)

    #####################
    # Preprocessing
    #####################

    autonlp.data_preprocessing()
    #import pandas as pd
    #data = pd.read_csv(flags.path_data)
    #c = data[100:150].reset_index(drop=True)
    #data_test, doc_spacy_data_test, y_test = autonlp.preprocess_test_data(c)

    #autonlp.data.to_csv('./results/data_preprocessed.csv', index=False)
    #autonlp.X_train.to_csv('./results/X_train.csv', index=False)
    #autonlp.X_test.to_csv('./results/X_test.csv', index=False)
    #autonlp.Y_train.to_csv('./results/Y_train.csv', index=False)
    #autonlp.Y_test.to_csv('./results/Y_test.csv', index=False)

    #####################
    # Training
    #####################

    autonlp.train()

    #####################
    # Leaderboard (Validation score)
    #####################

    leaderboard_val = autonlp.get_leaderboard(sort_by=flags.sort_leaderboard, dataset='val')
    print('\nValidation Leaderboard')
    print(leaderboard_val)
    #autonlp.save_scores_plot(leaderboard_val, 'last_logs')
    #leaderboard_val.to_csv('./results/leaderboard_val.csv', index=False)

    autonlp.correlation_models()

    df_all_results = autonlp.get_df_all_results()
    #df_all_results.to_csv('./results/df_all_results.csv', index=False)

    if len(df_all_results) > 0:
        df_all_results_mean = df_all_results.groupby('model').mean().sort_values('mean_test_score', ascending=False)
        print('\nGridSearch information Leaderboard')
        print(df_all_results_mean)
        #df_all_results.to_csv('./results/df_all_results_mean.csv', index=False)
    # autonlp.show_distribution_scores()

    #import numpy as np
    #df_oof_val = autonlp.Y_train.copy()
    #for name in autonlp.models.keys():
    #    df_oof_val[name] = autonlp.models[name].info_scores['oof_val'].reshape(-1)
    #df_oof_val.to_csv('./results/df_oof_val.csv', index=False)
    #print(df_oof_val)

    if 'binary' in autonlp.objective:
        autonlp.get_roc_curves()

    #####################
    # Testing
    #####################

    on_test_data = True
    name_logs = 'last_logs'
    autonlp.leader_predict(name_logs=name_logs, on_test_data=on_test_data)

    df_prediction = autonlp.dataframe_predictions
    #df_prediction.to_csv('./results/df_prediction.csv', index=False)

    leaderboard_test = autonlp.get_leaderboard(sort_by=flags.sort_leaderboard, dataset='test')
    print('\nTest Leaderboard')
    print(leaderboard_test)
    #autonlp.save_scores_plot(leaderboard_test, 'last_logs')
    #leaderboard_test.to_csv('./results/leaderboard_test.csv', index=False)

    #autonlp.launch_to_model_deployment('tf+Logistic_Regression')

    #import pandas as pd
    #data_test = pd.read_csv(flags.path_data)

    #X_test = data_test[[flags.column_text]].copy()
    #X_test = data_test.copy()
    #if isinstance(flags.target, list):
    #    Y_test = data_test[flags.target].copy()
    #else:
    #    Y_test = data_test[[flags.target]].copy()

    #X_test, doc_spacy_data_test, position_id_test, y_test = autonlp.preprocess_test_data(X_test)

    name_logs = 'best_logs'
    on_test_data = True
    autonlp.leader_predict(name_logs=name_logs, on_test_data=on_test_data)

    leaderboard_test = autonlp.get_leaderboard(sort_by=flags.sort_leaderboard, dataset='test',
                                               info_models=autonlp.info_models)
    print('\nTest Leaderboard')
    print(leaderboard_test)
