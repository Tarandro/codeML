import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging
from ..utils.logging import get_logger, verbosity_to_loglevel

logger = get_logger(__name__)


def matrice_doc_topic(model, document_word, fit):
    """ Creates the document x topic matrix and associates label -1 with documents with no topic membership (NMF and LDA only)
    Args:
        model : nmf or lda model
        document_word (sparse matrix) : tf or tfidf matrix
        print_results (Boolean)

    Returns :
        labels (array) : label of reduced data clustering
        n_clusters (int)
        n_noise (int) : number of documents with label -1
    """
    if fit:
        matrix_doc_topics = model.transform(document_word)
    else:
        matrix_doc_topics = model.fit_transform(document_word)
    index_is_zero = np.where(np.sum(matrix_doc_topics, axis=1) == 0)[0]
    doc_topic = np.argmax(matrix_doc_topics, axis=1)
    doc_topic[index_is_zero] = -1
    logger.info(
        'Documents associated with no topic : {} %\n'.format(np.round(len(index_is_zero) / len(doc_topic) * 100, 1)))
    logger.info("Value counts labels :")
    logger.info(pd.Series(doc_topic).value_counts())
    return doc_topic, matrix_doc_topics


def clustering_sklearn_predict_doc_topic(pipeline, data):
    """ sklearn clustering, number of topics must be assigned
    """
    doc_topic = pipeline.predict(data)
    logger.info("Value counts labels :")
    logger.info(pd.Series(doc_topic).value_counts())
    n_noise = list(doc_topic).count(-1)
    logger.info('Documents associated with no topic : {} %\n'.format(np.round(n_noise / len(doc_topic) * 100, 1)))
    return doc_topic, None


def clustering_sklearn_label_doc_topic(pipeline):
    """ hdbscan clustering, number of topics automatically assigned
    Args:
        data_reduc (array) : reduced data
        min_cluster_size (int) : min size of cluster

    Returns :
        labels (array) : label of reduced data clustering
        n_clusters (int)
        n_noise (int) : number of documents with label -1
    """
    doc_topic = None
    for cluster in ['Hdbscan', 'AgglomerativeClustering']:
        try:
            doc_topic = pipeline[cluster].labels_
        except:
            pass
    #doc_topic = pipeline["hdbscan"].labels_
    n_noise = list(doc_topic).count(-1)
    logger.info('Documents associated with no topic : {} %\n'.format(np.round(n_noise / len(doc_topic) * 100, 1)))
    logger.info("Value counts labels :")
    logger.info(pd.Series(doc_topic).value_counts())
    return doc_topic, None


def clustering_similarity_voc(data, vect_vocabulary_labels):
    score_by_cluster = np.zeros((data.shape[0], len(vect_vocabulary_labels.keys())))
    map_dict = {}
    dict_sim = {}
    for i, (k, list_vect_voc) in enumerate(vect_vocabulary_labels.items()):
        score = np.mean(cosine_similarity(data, list_vect_voc), axis=1)
        score_by_cluster[:, i] = score
        map_dict[i] = k
        dict_sim[k] = score

    doc_topic = list(np.argmax(score_by_cluster, axis=1))
    doc_topic = np.array([map_dict[i] for i in doc_topic])

    return doc_topic, dict_sim


def df_word_cluster(document_word, doc_topic, vectorizer):
    """ Create a dataframe of the word x topic matrix
    Args:
        document_word (sparse matrix) : tf or tf-idf matrix
        doc_topic (array) : the label topic of each document
        vectorizer : tf or tf-if vectorizer

    Returns :
        word_cluster (dataframe) : dataframe words x clusters
    """
    dict_df = {'terms': vectorizer.get_feature_names()}
    for i in np.unique(doc_topic):
        if i != -1:
            dict_df[i] = list(np.array(np.sum(document_word[np.where(doc_topic == i)[0], :], axis=0))[0])
    word_cluster = pd.DataFrame(dict_df)

    return word_cluster


def topic_terms_cluster(word_cluster, n_top_words, print_results, name_model):
    """ Create a dataframe of the top terms for each topic
    Args:
        word_cluster (dataframe) : dataframe words x clusters
        n_top_words (int)
        print_results (boolean)
        name_model (str)

    Returns :
        top_topic_terms (dataframe) dataframe of the top terms for each topic
    """
    top_topic_terms = {'terms': [], 'freq': [], 'cluster': []}

    for cluster_idx in range(word_cluster.shape[1] - 1):
        cluster_idx = word_cluster.iloc[:, cluster_idx + 1].name
        dataframe = word_cluster[['terms', cluster_idx]]
        dataframe = dataframe.sort_values(cluster_idx, ascending=False)
        top_words_cluster = list(dataframe['terms'])
        top_words_cluster_freq = list(dataframe[cluster_idx])

        message = "Topic {}: ".format(cluster_idx)
        top_topic_terms['terms'] += top_words_cluster[:n_top_words]
        top_topic_terms['freq'] += top_words_cluster_freq[:n_top_words]
        top_topic_terms['cluster'] += [cluster_idx for i in range(len(top_words_cluster[:n_top_words]))]
        message += " - ".join([text for text in top_words_cluster[:n_top_words]])

        if print_results:
            logger.info(message)

    top_topic_terms = pd.DataFrame(top_topic_terms)
    top_topic_terms['modele'] = name_model

    return top_topic_terms