from ...features.embeddings.base_embedding import Base_Embedding
import numpy as np
from hyperopt import hp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import transformers
from transformers import AutoTokenizer, TFAutoModel, AutoConfig, pipeline, \
                        CamembertTokenizer, FlaubertTokenizer, XLMRobertaTokenizer, \
                        RobertaTokenizer, BertTokenizer, TFCamembertModel, TFFlaubertModel, TFXLMRobertaModel, \
                        TFRobertaModel, TFBertModel
from sentence_transformers import SentenceTransformer

#import transformers
#import tokenizers

import tensorflow as tf

from ...utils.logging import get_logger

logger = get_logger(__name__)


class TransformerNLP(Base_Embedding):
    """ Base_Embedding class with TransformerNLP embedding method from Huggingface library """

    def __init__(self, flags_parameters, column_text, dimension_embedding):
        Base_Embedding.__init__(self, flags_parameters, column_text, dimension_embedding)
        self.name_model = 'transformer'
        self.maxlen = None
        self.tokenizer = None

    def hyper_params(self, size_params='small'):

        self.parameters = dict()
        self.parameters['learning_rate'] = hp.choice('learning_rate', self.flags_parameters.tr_learning_rate)
        return self.parameters

    def init_params(self, size_params='small', method_embedding='camembert'):
        if self.maxlen is None:
            if size_params == 'small':
                self.maxlen = self.flags_parameters.tr_maxlen
            else:
                self.maxlen = self.flags_parameters.tr_maxlen  # 350

        language = self.flags_parameters.language_text

        if method_embedding.lower() in ['bert']:
            method_embedding = 'bert-base-uncased'
        elif method_embedding.lower() in ['camembert']:
            method_embedding = 'jplu/tf-camembert-base'
        elif method_embedding.lower() in ['flaubert']:
            method_embedding = 'jplu/tf-flaubert-base-uncased'
        elif method_embedding.lower() in ['xlm-roberta']:
            method_embedding = 'jplu/tf-xlm-roberta-base'
        elif method_embedding.lower() in ['roberta']:
            method_embedding = 'roberta-base'
        elif method_embedding.lower() in ['sentence-bert']:
            method_embedding = "sentence-transformers/distiluse-base-multilingual-cased-v2" # paraphrase-xlm-r-multilingual-v1
        elif method_embedding.lower() in ['zero-shot']:
            method_embedding = "joeddav/xlm-roberta-large-xnli"

        self.method_embedding = method_embedding

        if self.tokenizer is None:
            # Instantiate tokenizer
            if "sentence-transformers" not in self.method_embedding and "nli" not in self.method_embedding:
                if "camembert" in self.method_embedding.lower():
                    self.tokenizer = CamembertTokenizer.from_pretrained(self.method_embedding)
                elif "flaubert" in self.method_embedding.lower():
                    self.tokenizer = FlaubertTokenizer.from_pretrained(self.method_embedding)
                elif "xlm-roberta" in self.method_embedding.lower():
                    self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.method_embedding)
                elif "roberta" in self.method_embedding.lower():
                    self.tokenizer = RobertaTokenizer.from_pretrained(self.method_embedding)
                elif "bert" in self.method_embedding.lower():
                    self.tokenizer = BertTokenizer.from_pretrained(self.method_embedding)
                else:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(self.method_embedding)  # model_name_or_path
                    except:
                        logger.critical("unknown embedding method name : '{}'".format(self.method_embedding))
            else:
                self.tokenizer = None

    def preprocessing_fit_transform(self, x, size_params='small', method_embedding='CamemBERT'):
        """ Fit preprocessing and transform x according to embedding method and dimension embedding
            1st step : initialize some fixed params and tokenizer needed for embedding method
            2nd step : Transformer Tokenization
            3rd step:
                - word dimension embedding : no more preprocessing to do
                - document dimension embedding : get document vectors with Transformer pre-trained model
        Args:
            x (Dataframe) need to have column column_text
            size_params ('small' or 'big') size of parameters range for optimization
            method_embedding (str) name of a Transformer model
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

        if "sentence-transformers" not in self.method_embedding and "nli" not in self.method_embedding:
            ct = len(x_preprocessed)
            # INPUTS
            if 'roberta' in self.method_embedding.lower() or 'camembert' in self.method_embedding.lower():
                ids = np.ones((ct, self.maxlen), dtype='int32')
            else:
                ids = np.zeros((ct, self.maxlen), dtype='int32')
            att = np.zeros((ct, self.maxlen), dtype='int32')
            tok = np.zeros((ct, self.maxlen), dtype='int32')

            for k in range(ct):
                text = "  " + " ".join(x_preprocessed[k].split())

                #if self.method_embedding.lower() == 'roberta':
                #    enc = self.tokenizer.encode(text)
                #else:
                enc = self.tokenizer.encode(text, max_length=self.maxlen, truncation=True)

                # CREATE BERT INPUTS
                if self.method_embedding.lower() == 'roberta':
                    ids[k, :len(enc.ids)] = enc.ids[:self.maxlen]
                    att[k, :len(enc.ids)] = 1
                else:
                    ids[k, :len(enc)] = enc
                    att[k, :len(enc)] = 1

            x_preprocessed = [ids, att, tok]
        else:
            x_preprocessed = x_preprocessed

        if self.dimension_embedding == 'word_embedding':
            return x_preprocessed
        else:
            if "sentence-transformers" in self.method_embedding:
                document_embedding = self.model_sentence_embedding(x_preprocessed)
            elif "nli" in self.method_embedding:
                x_preprocessed = [' '.join(text.split()[:self.maxlen]) for text in x_preprocessed]
                return x_preprocessed
            else:
                model_extractor = self.model_extract_document_embedding()
                document_embedding = model_extractor.predict(x_preprocessed)
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

        if "sentence-transformers" not in self.method_embedding and "nli" not in self.method_embedding:
            ct = len(x_preprocessed)
            # INPUTS
            if 'roberta' in self.method_embedding.lower() or 'camembert' in self.method_embedding.lower():
                ids = np.ones((ct, self.maxlen), dtype='int32')
            else:
                ids = np.zeros((ct, self.maxlen), dtype='int32')
            att = np.zeros((ct, self.maxlen), dtype='int32')
            tok = np.zeros((ct, self.maxlen), dtype='int32')

            for k in range(ct):
                text = "  " + " ".join(x_preprocessed[k].split())

                if self.method_embedding.lower() == 'roberta':
                    enc = self.tokenizer.encode(text)
                else:
                    enc = self.tokenizer.encode(text, max_length=self.maxlen, truncation=True)

                # CREATE BERT INPUTS
                if self.method_embedding.lower() == 'roberta':
                    ids[k, :len(enc.ids)] = enc.ids[:self.maxlen]
                    att[k, :len(enc.ids)] = 1
                else:
                    ids[k, :len(enc)] = enc
                    att[k, :len(enc)] = 1

            x_preprocessed = [ids, att, tok]
        else:
            x_preprocessed = x_preprocessed

        if self.dimension_embedding == 'word_embedding':
            return x_preprocessed
        else:
            if "sentence-transformers" in self.method_embedding:
                document_embedding = self.model_sentence_embedding(x_preprocessed)
            elif "nli" in self.method_embedding:
                x_preprocessed = [' '.join(text.split()[:self.maxlen]) for text in x_preprocessed]
                return x_preprocessed
            else:
                model_extractor = self.model_extract_document_embedding()
                document_embedding = model_extractor.predict(x_preprocessed)
            return document_embedding

    def model_extract_document_embedding(self):
        """ Create a Tensorflow model which extract [CLS] token output of the Transformer model
            [CLS] token output can be used as a document vector
        Return:
            model (Tensorflow model)
        """
        input_ids = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="ids")
        attention_mask = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="att")
        token = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="tok")

        try:
            if "camembert" in self.method_embedding.lower():
                self.auto_model = TFCamembertModel.from_pretrained(self.method_embedding)
            elif "flaubert" in self.method_embedding.lower():
                self.auto_model = TFFlaubertModel.from_pretrained(self.method_embedding)
            elif "xlm-roberta" in self.method_embedding.lower():
                self.auto_model = TFXLMRobertaModel.from_pretrained(self.method_embedding)
            elif "roberta" in self.method_embedding.lower():
                self.auto_model = TFRobertaModel.from_pretrained(self.method_embedding)
            elif "bert" in self.method_embedding.lower():
                self.auto_model = TFBertModel.from_pretrained(self.method_embedding)
            else:
                config = AutoConfig.from_pretrained(self.method_embedding)
                self.auto_model = TFAutoModel.from_pretrained(self.method_embedding, config=config)
            x = self.auto_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        except:
            logger.critical("unknown embedding method name : '{}'".format(self.method_embedding))

        # word vectors shape : (None, maxlen, 768)
        x = x[0]
        cls_token = x[:, 0, :]

        model = tf.keras.models.Model(inputs=[input_ids, attention_mask, token], outputs=cls_token)
        return model

    def model_sentence_embedding(self, encoded_input):
        model = SentenceTransformer(self.method_embedding)

        size_batch = 200
        n_batch = len(encoded_input) // size_batch
        for i in tqdm(range(n_batch + 1)):
            x = encoded_input[i * size_batch: (i + 1) * size_batch]
            if len(x) > 0:
                # Compute token embeddings
                subset_sentence_embeddings = model.encode(x)
                if i == 0:
                    sentence_embeddings = subset_sentence_embeddings
                else:
                    sentence_embeddings = np.concatenate((sentence_embeddings, subset_sentence_embeddings), axis=0)
        return sentence_embeddings

    def model_zero_shot(self):
        classifier = pipeline("zero-shot-classification", model=self.method_embedding, device=0)
        return classifier

    def load_tokenizer(self, outdir):
        # Instantiate tokenizer
        if "sentence-transformers" not in self.method_embedding and "nli" not in self.method_embedding:
            if "camembert" in self.method_embedding.lower():
                self.tokenizer = CamembertTokenizer.from_pretrained(self.method_embedding)
            elif "flaubert" in self.method_embedding.lower():
                self.tokenizer = FlaubertTokenizer.from_pretrained(self.method_embedding)
            elif "xlm-roberta" in self.method_embedding.lower():
                self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.method_embedding)
            elif "roberta" in self.method_embedding.lower():
                self.tokenizer = RobertaTokenizer.from_pretrained(self.method_embedding)
            elif "bert" in self.method_embedding.lower():
                self.tokenizer = BertTokenizer.from_pretrained(self.method_embedding)
            else:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.method_embedding)  # model_name_or_path
                except:
                    logger.critical("unknown embedding method name : '{}'".format(self.method_embedding))
        else:
            self.tokenizer = None

    def model(self):
        input_ids = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="ids")
        attention_mask = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="att")
        token = tf.keras.layers.Input(shape=(self.maxlen,), dtype=tf.int32, name="tok")

        inp = [input_ids, attention_mask, token]

        try:
            if "camembert" in self.method_embedding.lower():
                self.auto_model = TFCamembertModel.from_pretrained(self.method_embedding)
            elif "flaubert" in self.method_embedding.lower():
                self.auto_model = TFFlaubertModel.from_pretrained(self.method_embedding)
            elif "xlm-roberta" in self.method_embedding.lower():
                self.auto_model = TFXLMRobertaModel.from_pretrained(self.method_embedding)
            elif "roberta" in self.method_embedding.lower():
                self.auto_model = TFRobertaModel.from_pretrained(self.method_embedding)
            elif "bert" in self.method_embedding.lower():
                self.auto_model = TFBertModel.from_pretrained(self.method_embedding)
            else:
                config = AutoConfig.from_pretrained(self.method_embedding)
                try:
                    self.auto_model = TFAutoModel.from_config(config)
                except:
                    self.auto_model = TFAutoModel.from_pretrained(self.method_embedding, config=config)
            x = self.auto_model(input_ids, attention_mask=attention_mask, token_type_ids=token)
        except:
            logger.critical("unknown embedding method name : '{}'".format(self.method_embedding))

        # word vectors shape : (None, maxlen, 768)
        x = x[0] # last hidden layers
        return x, inp
