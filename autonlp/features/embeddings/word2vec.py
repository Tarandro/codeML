from ...features.embeddings.base_embedding import Base_Embedding
import numpy as np
from tqdm import tqdm
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import re
import string
from hyperopt import hp
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
import gensim
from ...features.embeddings.gensim_model.scratch_gensim_model import build_word2vec_model

from ...utils.logging import get_logger

logger = get_logger(__name__)


class Word2Vec(Base_Embedding):
    """ Base_Embedding class with Word2Vec embedding method from gensim or pre-trained word2vec weights """

    def __init__(self, flags_parameters, column_text, dimension_embedding):
        Base_Embedding.__init__(self, flags_parameters, column_text, dimension_embedding)
        self.name_model = 'word2vec'
        self.tokenizer = None
        self.embed_size = None
        self.max_features = None
        self.maxlen = None
        self.method_embedding = None
        self.embedding_matrix = None

    def hyper_params(self, size_params='small'):

        self.parameters = dict()

        self.parameters['learning_rate'] = hp.choice('learning_rate', self.flags_parameters.w2v_learning_rate)
        return self.parameters

    def init_params(self, size_params='small', method_embedding='word2vec'):
        if self.dimension_embedding == 'word_embedding':
            if size_params == 'small':
                if self.max_features is None:
                    self.max_features = self.flags_parameters.w2v_max_features
                if self.maxlen is None:
                    self.maxlen = self.flags_parameters.w2v_maxlen
            else:
                if self.max_features is None:
                    self.max_features = self.flags_parameters.w2v_max_features  # 100000
                if self.maxlen is None:
                    self.maxlen = self.flags_parameters.w2v_maxlen  # 350
        self.method_embedding = method_embedding

    def preprocessing_fit_transform(self, x, size_params='small', method_embedding='word2vec'):
        """ Fit preprocessing and transform x according to embedding method and dimension embedding
            1st step : initialize some fixed params needed for embedding method
            2nd step : Build a Word2Vec scratch model or use a pre-trained Word2Vec model/weights
            3rd step:
                - word dimension embedding : tensorflow tokenization + get word matrix embedding with Word2Vec method
                - document dimension embedding : get document vectors with Word2Vec method
        Args:
            x (Dataframe) need to have column column_text
            size_params ('small' or 'big') size of parameters range for optimization
            method_embedding (str) 'word2vec' if want to use a scratch model else a path for a pre-trained model/weights
        Return:
            - word dimension embedding : x_preprocessed (dict)
            - document dimension embedding : document_embedding (array) a matrix of document vectors
        """
        self.init_params(size_params, method_embedding)

        if isinstance(x, list):
            x_preprocessed = x
        else:
            if isinstance(self.column_text, int) and self.column_text not in x.columns:
                col = self.column_text
            else:
                col = list(x.columns).index(self.column_text)
            x_preprocessed = list(x.iloc[:, col])

        # build gensim scratch model
        if self.method_embedding.lower() == 'word2vec':
            if self.apply_logs:
                dir_word2vec = os.path.join(self.flags_parameters.outdir, 'Word2Vec')
            elif self.apply_mlflow:
                dir_word2vec = os.path.join(self.path_mlflow, self.experiment_id, 'Word2Vec')
            if not os.path.exists(dir_word2vec):
                os.makedirs(dir_word2vec, exist_ok=True)
                logger.info(
                    "Build Word2Vec model from scratch with train set and size_vector={}, window={}, epochs={} ...".format(
                        self.flags_parameters.w2v_size_vector, self.flags_parameters.w2v_window,
                        self.flags_parameters.w2v_epochs))
                build_word2vec_model(x_preprocessed, output_dir=dir_word2vec,
                                     size_vector=self.flags_parameters.w2v_size_vector,
                                     window=self.flags_parameters.w2v_window, epochs=self.flags_parameters.w2v_epochs)
                logger.info("Save Word2Vec model in '{}'".format(dir_word2vec))
                self.method_embedding = os.path.join(dir_word2vec, "word2vec.wordvectors")
            else:
                self.method_embedding = os.path.join(dir_word2vec, "word2vec.wordvectors")
                logger.info("Load Word2Vec scratch model from path : {}".format(dir_word2vec))

        # build_embedding_matrix(self):
        try:
            try:
                self.embeddings_gensim_model = load_model(self.method_embedding)
                self.method = "model"
            except:
                self.embeddings_gensim_model = load_keyedvectors(self.method_embedding)
                self.method = "keyedvectors"
        except Exception:
            logger.critical("unknown path for Word2Vec weights : '{}'".format(self.method_embedding))

        if self.dimension_embedding == 'word_embedding':
            # Tokenization by tensorflow with vocab size = max_features
            if self.tokenizer is None:
                if self.apply_logs:
                    dir_tokenizer = os.path.join(self.flags_parameters.outdir, 'tokenizer.pickle')
                elif self.apply_mlflow:
                    dir_tokenizer = os.path.join(self.path_mlflow, self.experiment_id, 'tokenizer.pickle')
                if os.path.exists(dir_tokenizer):
                    with open(dir_tokenizer, 'rb') as handle:
                        self.tokenizer = pickle.load(handle)
                    logger.info("Load Tensorflow Tokenizer from past tokenization in {}".format(dir_tokenizer))
                else:
                    self.tokenizer = Tokenizer(num_words=self.max_features, lower=True, oov_token="<unk>")
                    self.tokenizer.fit_on_texts(x_preprocessed)

                    # Save Tokenizer :
                    if self.apply_logs:
                        path_tokenizer = os.path.join(self.flags_parameters.outdir, 'tokenizer.pickle')
                        with open(path_tokenizer, 'wb') as handle:
                            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    if self.apply_mlflow:
                        path_tokenizer = os.path.join(self.path_mlflow, self.experiment_id, 'tokenizer.pickle')
                        with open(path_tokenizer, 'wb') as handle:
                            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


            tok = self.tokenizer.texts_to_sequences(x_preprocessed)

            self.word_index = self.tokenizer.word_index
            self.vocab_idx_word = {idx: word for word, idx in self.tokenizer.word_index.items()}
            self.length_word_index = len(self.word_index)

            tok = pad_sequences(tok, maxlen=self.maxlen, padding='post')

            x_preprocessed = {"tok": tok}

            self.embedding_matrix = build_embedding_matrix_from_gensim_model(self.word_index, self.embeddings_gensim_model, self.method)
            self.embed_size = self.embedding_matrix.shape[1]

            return x_preprocessed

        else:
            document_embedding = build_embedding_documents_from_gensim_model(x_preprocessed,
                                                                             self.embeddings_gensim_model, self.method,
                                                                             self.flags_parameters.language_text)
            return document_embedding

    def preprocessing_transform(self, x):
        """ Transform x data according to latest fit preprocessing
        Args:
            x (Dataframe) need to have column column_text
        Return:
            - word dimension embedding : x_preprocessed (dict)
            - document dimension embedding : document_embedding (array) a matrix of document vectors
        """
        if isinstance(x, list):
            x_preprocessed = x
        else:
            if isinstance(self.column_text, int) and self.column_text not in x.columns:
                col = self.column_text
            else:
                col = list(x.columns).index(self.column_text)
            x_preprocessed = list(x.iloc[:, col])

        if self.dimension_embedding == 'word_embedding':
            tok = self.tokenizer.texts_to_sequences(x_preprocessed)
            tok = pad_sequences(tok, maxlen=self.maxlen, padding='post')
            x_preprocessed = {"tok": tok}
            return x_preprocessed
        else:
            try:
                _ = self.embeddings_gensim_model
                _ = self.method
            except:
                try:
                    try:
                        self.embeddings_gensim_model = load_model(self.method_embedding)
                        self.method = "model"
                    except:
                        self.embeddings_gensim_model = load_keyedvectors(self.method_embedding)
                        self.method = "keyedvectors"
                except Exception:
                    logger.critical("unknown path for Word2Vec weights : '{}'".format(self.method_embedding))

            document_embedding = build_embedding_documents_from_gensim_model(x_preprocessed,
                                                                             self.embeddings_gensim_model, self.method,
                                                                             self.flags_parameters.language_text)
            return document_embedding

    def load_tokenizer(self, outdir):
        try:
            with open(os.path.join(outdir, 'tokenizer.pickle'), 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        except:
            logger.warning("tokenizer.pickle is not provided in {}".format(outdir))

    def model(self):

        if self.dimension_embedding == 'word_embedding':
            token = tf.keras.layers.Input(shape=(self.maxlen,), name="tok")
            inp = {"tok": token}

            # Embedding
            if self.embedding_matrix is not None:
                x = Embedding(self.length_word_index + 1, self.embed_size, weights=[self.embedding_matrix], trainable=True)(token)
            else:
                x = Embedding(self.length_word_index + 1, self.embed_size, trainable=True)(token)

            return x, inp


#################
# Help function : Get Word2Vec pre-training-weight and attention head
#################

def load_model(embed_dir):
    """ Load a full gensim model
    Args:
        embed_dir (str) path of gensim model
        model are often separated in several files but it only need the path of one file
        all files need to be in the same directory
    Return:
        model (gensim model)
    """
    # need to have gensim model + syn0.npy + syn1neg.npy
    model = gensim.models.Word2Vec.load(embed_dir)
    return model


def load_keyedvectors(embed_dir):
    """ Load a word vector gensim model : the model have only the option to give vector of a string
    Args:
        embed_dir (str) path of word vector gensim model
        model are often separated in several files but it only need the path of one file
        all files need to be in the same directory
    Return:
        embedding_index (word vector gensim model)
    """
    # need to have file : word2vec.wordvectors
    embedding_index = gensim.models.KeyedVectors.load(embed_dir)
    return embedding_index


def get_vect(word, model, method):
    """ Obtain the vector of a word with gensim model according to method
    Args:
        word (str)
        model (word vector gensim model or full gensim model)
        method ('model' or 'vector')
    Return:
        vector (array)
    """
    if method == "model":
        try:
            return model.wv[word]
        except KeyError:
            return None
    else:
        try:
            return model[word]
        except KeyError:
            return None


def build_embedding_matrix_from_gensim_model(word_index, model, method="model", lower=True, verbose=True):
    """ Create a word vector for each word in dictionary word_index with a gensim model
    Args:
        word_index (dict) dictionary word:index got from tensorflow tokenization
        model (word vector gensim model or full gensim model)
        method ('model' or 'vector')
        lower (Boolean) lower each word of embedding matrix
        verbose (Boolean) show iteration progression in word_index
    Return:
         embedding_matrix (array) matrix of word vectors
    """
    embedding_matrix = None
    for word, i in tqdm(word_index.items(), disable=not verbose):
        if lower:
            word = word.lower()
        embedding_vector = get_vect(word, model, method)
        if embedding_matrix is None and embedding_vector is not None:
            embedding_matrix = np.zeros((len(word_index) + 1, embedding_vector.shape[0]))
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def build_embedding_documents_from_gensim_model(documents, model, method="model", language_text="fr", lower=True, verbose=True):
    """ Create a document vector for each document in documents with a gensim model
        and concatenate to get an embedding matrix
    Args:
        documents (List[str])
        model (word vector gensim model or full gensim model)
        method ('model' or 'vector')
        lower (Boolean) lower each word of embedding matrix
        verbose (Boolean) show iteration progression in word_index
    Return:
        embedding_documents (array) matrix of document vectors
    """
    embedding_documents = None

    # add option document vector = average weighted by tf-idf
    if language_text == 'fr':
        stopwords = fr_stop
    else:
        stopwords = en_stop

    vectorizer = TfidfVectorizer(stop_words=stopwords, ngram_range=(1, 1))
    vectorizer.fit([d.lower() for d in documents])
    voc_dict_weight = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

    for i, doc in tqdm(enumerate(documents), disable=not verbose):
        if lower:
            doc = doc.lower()
        doc = re.sub('[%s]' % re.escape(string.punctuation), ' ', doc)
        try:
            doc_split = doc.split(' ')
            embedding_vector = []
            for word in doc_split:
                vect_word = get_vect(word, model, method)
                if vect_word is None:
                    continue
                try:
                    weight_word = voc_dict_weight[word.lower()]
                    vect_word = vect_word * weight_word
                    embedding_vector.append(vect_word)
                except:
                    pass

            embedding_vector = [i for i in embedding_vector if i is not None]
            if len(embedding_vector) < 1:
                embedding_vector = None
            else:
                embedding_vector = sum(embedding_vector)
            if embedding_documents is None and embedding_vector is not None:
                embedding_documents = np.zeros((len(documents), embedding_vector.shape[0]))
        except:
            embedding_vector = None
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_documents[i] = embedding_vector
    return embedding_documents