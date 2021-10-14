# AutoNLP

AutoNLP is an automated training and deployment of NLP models

## Requirements

Python 3.8 or later with all requirements.txt dependencies installed. To install run:
```python
$ pip install -r requirements.txt
```

## Minimum codes for classification

Setting up:

```python
from autonlp.autonlp import AutoNLP
from autonlp.flags import Flags

parameters_to_update = {
    "path_data": "data/FinancialPhraseBank.csv",
    "column_text": "text_fr",
    "target": "sentiment",
    "language_text": "fr",
    "objective": 'multi-class',
    "embedding": {"tf": 1, "tf-idf": 2, "word2vec": None, "fasttext": None, "doc2vec": None, "transformer": None},

    "classifier": {"Naive_Bayes": [1], "Logistic_Regression": [2], "SGD_Classifier": [1],
                   "XGBoost": [], "Global_Average": [], "Attention": [], "BiRNN": [], "BiRNN_Attention": [],
                   "biLSTM": [], "BiLSTM_Attention": [], "biGRU": [], "BiGRU_Attention": []},
    "scoring": 'f1',
    "method_embedding": {'Word2vec': 'Word2Vec',
                         'Fasttext': 'FastText',
                         'Doc2Vec': 'Doc2Vec',
                         'Transformer': 'CamemBERT',
                         'spacy': [('all', False)]}
}
# update parameters :
flags = Flags().update(parameters_to_update)
autonlp = AutoNLP(flags)
```
Preprocessing, split train/test and Training + Validation:
```python
autonlp.data_preprocessing()
autonlp.train()
# validation leaderboard :
leaderboard_val = autonlp.get_leaderboard(dataset='val')
```
Prediction on test set for all models:
```python
autonlp.leader_predict()
# get prediction :
df_prediction = autonlp.dataframe_predictions
# test leaderboard :
leaderboard_test = autonlp.get_leaderboard(dataset='test')
```
Save a model for deployment:
```python
autonlp.launch_to_model_deployment('tf-idf+Logistic_Regression')
```
Deploy the model with the combination API / Dashboard / Docker
```python
# Run docker compose (backend : autonlp + deployment model + api / frontend : Streamlit Dashboard)
!docker-compose up -d --build
# Open the Dashboard in http://host.docker.internal:8501
```


## Usage examples

To find out how to work with AutoNLP:

- [Tutorial 1: autonlp-classification.ipynb](notebooks/autonlp-classification.ipynb) pipeline using Hyperparameters
  Optimization, cross-validation and prediction for models : 'tf-idf+SGD_Classifier', 'tf-Logistic_Regression', 'tf-Naive_Bayes' and use manual logs option.

If you want to apply AutoNLP in several steps, each step can be used multiple times: (manual logs option)

1. [autonlp-optimization.ipynb](notebooks/autonlp-optimization.ipynb) apply only hyperparameters
   optimization, keep a history of each model tested (hyperparameters + score) and save in a json file
   "models_best_parameters.json" hyperparameters with best score for each model.
   
2. [autonlp-validation.ipynb](notebooks/autonlp-validation.ipynb) use previous logs,
   apply validation or cross-validation, save model and compute metric scores.
   
3. [autonlp-prediction.ipynb](notebooks/autonlp-prediction.ipynb) apply prediction on all saved models.

- [Tutorial 2: autonlp-unsupervised.ipynb](notebooks/autonlp-unsupervised.ipynb) pipeline using unsupervised methods: 
  "tf+LDA", "tf-idf+NMF_Kullback", "fasttext+Similarity_voc" and use manual logs option.
  
- [Tutorial 3: autonlp-zero-shot-classification.ipynb](notebooks/autonlp-zero-shot-classification.ipynb) pipeline using
  zero shot classification and use manual logs option.


## Parameters - Quick overview

Parameters are instantiated with flags.py :  

- objective : specify target objective
```python
list_possible_objective = ['binary', 'multi-class', 'regression']
```
    For 'binary', only labels 0 and 1 are possible.
    For 'multi-class', if labels are numerics, labels must be in the range 0 to the number of labels.
- embedding: vectorization methods
```python
embedding= {"tf": 1, "tf-idf": None, "word2vec": 2, "fasttext": None, "doc2vec": None, "transformer": 3}
```
- classifier: classification methods / numbers are linked to embedding methods
```python
classifier = {"Naive_Bayes": [1], "Logistic_Regression": [], "SGD_Classifier": [],
              "XGBoost": [1], "Global_Average": [3], "Attention": [2], "BiRNN": [], "BiRNN_Attention": [],
              "biLSTM": [], "BiLSTM_Attention": [], "biGRU": [], "BiGRU_Attention": []}
```
- regressor: regression methods
```python
regressor = {"SGD_Regressor": [], "XGBoost": [], 
             "Global_Average": [], "Attention": [], "BiRNN": [], "BiRNN_Attention": [],
             "biLSTM": [], "BiLSTM_Attention": [], "biGRU": [], "BiGRU_Attention": []}
```
- clustering: clustering methods / unsupervised method / zero-shot classification
```python
clustering = {"NMF_frobenius": [], "NMF_kullback": [], "LDA": [1],
              "hdbscan": [], "ACP_hdbscan": [], "UMAP_hdbscan": [],
              "kmeans": [], "ACP_kmeans": [2], "UMAP_kmeans": [2],
              "agglomerativeclustering": [], "ACP_agglomerativeclustering": [], "UMAP_agglomerativeclustering": [],
              "Similarity_voc": [1,2], "zero_shot": [3]}
```
- method_embedding : information about embedding method
```python
# default:
method_embedding = {'Word2Vec': 'Word2Vec',
                    'FastText': 'FastText',
                    'Doc2Vec': 'Doc2Vec',
                    'Transformer': 'CamemBERT',
                    'spacy': [('all', False), (['ADJ', 'NOUN', 'VERB', 'DET'], False),
                              (['ADJ', 'NOUN', 'VERB', 'DET'], True)]}
```
    For 'Word2Vec', 'FastText' and 'Doc2Vec', you can create a gensim model from scratch by writing
    the name of the model as embedding method or you can use pre-trained model/weights by indicating the path.
    
    For 'Transformer', you have the choice between these pre-trained models : 'BERT', 'RoBERTa',
    'CamemBERT', 'FlauBERT', 'XLM-RoBERTa', 'sentence-BERT' and 'zero-shot' / or write the path from huggingface library
    
    For 'Spacy', it doesn't indicate an embedding method but the preprocessing step for
    tf and tf-idf embedding method. You can choose several preprocessing methods in the tuple
    format (keep_pos_tag, lemmatize). keep_pos_tag can be 'all' for no pos_tag else list of tags to keeps.
    lemmatize is boolean to know if apply lemmatization by Spacy model.

- path_data_validation : write a path to use your own validation set instead of using cross-validation
- apply_logs : if True, use a manual logs to track and save model
- apply_mlflow : if True, use MLflow Tracking
- scoring : metric optimized during optimization
```python
binary_posssible_scoring = ['accuracy','f1','recall','precision','roc_auc']
multi_class_posssible_scoring = ['accuracy','f1','recall','precision']
regression_posssible_scoring = ['mse','explained_variance','r2']
```
- apply_optimization : if True, apply Hyperparameters Optimization else load models parameters from path
indicated in flags.path_models_parameters
- apply_validation : if True, apply validation / cross-validation and save models

## Documentation

- [Preprocessing](autonlp/features/README.md)
- [Embeddings](autonlp/features/embeddings/README.md)
- [Classifiers](autonlp/models/classifier/README.md)
- [UML](AutoNLP%20UML.png)
- [Classes](autonlp/README.md)

## Things to improve

- In class Preprocessing_NLP : Need to download Spacy model at each instantiation of AutoNLP, not practical for inference

- add model BART / BARThez

- better way to score prediction for unsupervised methods

- add an option of time series prediction

- data engineering part