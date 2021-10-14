from autonlp.flags import Flags

#####################
# Parameters
#####################

### Topic Modeling display
flags_dict_topic_display = {
    "show_top_terms_topics": False,
    # format (keep_pos_tag, lemmatize). keep_pos_tag can be 'all' for no pos_tag else list of tags to keeps.
    # lemmatize is boolean to know if you wan to apply lemmatization by Spacy model.
    "preprocess_topic": (['ADJ', 'NOUN', 'VERB'], True),
    "n_top_words": 10,
    "min_ngram": 1,
    "max_ngram": 1,
}


flags = Flags().update(flags_dict_topic_display)