  # Importing libraries
import numpy as np
import pandas as pd
import nltk
# nltk.download('punkt')
import re
# nltk.download('stopwords')
from nltk.corpus import stopwords
# stop_words = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  
le=WordNetLemmatizer()
import logging
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
tqdm.pandas(desc="progress bar!")
import scipy.stats as stats
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances

from collections import Counter
from operator import itemgetter

from ML_pipeline import dataset
from ML_pipeline import pre_processing
from ML_pipeline import vectorizing_dataset
from ML_pipeline import topic_modeling
from ML_pipeline import predict_topic
from ML_pipeline import lsa_model
from ML_pipeline import predict_lsa
from ML_pipeline import utils
from ML_pipeline import tuning_lda


print('script started')
# Reading the dataset
train_documents, test_documents          = dataset.read_data("E:/PROJECTpro/PROJECTS/project_2_topic_modelling/Topic_modeling/input/documents.csv")

# Text Preprocessing 

## New column having the cleaned sentences
train_documents['clean_document']        = train_documents['document'].progress_apply(lambda x: pre_processing.clean_documents(x)[0])
test_documents['clean_document']         = test_documents['document'].progress_apply(lambda x: pre_processing.clean_documents(x)[0])
## New column having the cleaned tokens
train_documents['clean_token']           = train_documents['document'].progress_apply(lambda x: pre_processing.clean_documents(x)[1])
test_documents['clean_token']            = test_documents['document'].progress_apply(lambda x: pre_processing.clean_documents(x)[1])

# train_documents.to_csv('../output/train_documents.csv', index = False)
# test_documents.to_csv('../output/test_documents.csv', index = False)


# Transforming dataset into

## Count Vectorizer
count_vect, count_vect_text              = vectorizing_dataset.transform_dataset(train_documents, 'clean_document', 'count')
count_vectorized_test                    = count_vect.transform(test_documents['clean_document'])
## TFIDF Vectorizer

tfidf_vect, tfidf_vect_text              = vectorizing_dataset.transform_dataset(train_documents, 'clean_token', 'tfidf')
tfidf_vectorized_test                    = tfidf_vect.transform(test_documents['clean_token'])

# Topic Modeling
## LSA
print("--------------LSA starts-------------------")
lsa_model, lsa_top                      = lsa_model.lsa_model( tfidf_vect_text , '../output/lsa_model_trained.pkl')
documet_topic_lsa                       = predict_lsa.topics_document(model_output= lsa_top, n_topics=10, data=train_documents)

lsa_keys                                = utils.get_keys(lsa_top)
lsa_categories, lsa_counts              = utils.keys_to_counts(lsa_keys)

print("----------------LSA ends--------------------")

## LDA
print("--------------LDA starts-------------------")
lda_model, lda_model_output              = topic_modeling.modeling(count_vect_text, 'count', model_path='../output/lda_trained.pkl')

'''
# Takes too much time. Run this if you have efficient computer CPU.
search_params = {'n_components': [10, 15, 20], 'learning_decay': [.5, .7, .9]}
best_lda_model = tuning_lda.tune_lda(search_params, count_vect_text, "../output/best_lda_model.pkl" )
'''
print("--------------LDA ends---------------------")
# ## NMF
print("--------------NMF starts---------------------")
nmf_model, nmf_model_output              = topic_modeling.modeling(tfidf_vect_text, 'tfidf', model_path='../output/nmf_trained.pkl')
print("--------------NMF ends---------------------")
# # # Predict topic

## LDA
topic_seris_lda                          = predict_topic.topic_document(lda_model, count_vectorized_test, 10)
## NMF
topic_seris_nmf                          = predict_topic.topic_document(nmf_model, tfidf_vectorized_test, 13)

# ## Exporting the dataset with the topic attached

test_documents['index'] = [i for i in range(len(test_documents))]
## LDA 
test_documents_lda                       = pd.merge(test_documents[['index','document']], topic_seris_lda, on = ['index'], how = 'left')
## NMF 
test_documents_nmf                       = pd.merge(test_documents[['index','document']], topic_seris_nmf, on = ['index'], how = 'left')



path = '../output'
# LDA
test_documents_lda[['document','dominant_topic']].to_csv(path+'/'+'test_lda_1.csv', index=False)
# NMF
test_documents_nmf[['document','dominant_topic']].to_csv(path+'/'+'test_nmf_1.csv', index=False)
print('script completed successfully')