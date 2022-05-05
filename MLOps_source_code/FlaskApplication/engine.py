  # Importing libraries
import numpy as np
import boto3
from io import StringIO
import os
import pandas as pd
import nltk
import joblib
# nltk.download('punkt')
import re
import datetime
# nltk.download('stopwords')
from nltk.corpus import stopwords
# nltk.download('wordnet')
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
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import euclidean_distances

from collections import Counter
from operator import itemgetter
import traceback

from ML_pipeline import dataset
from ML_pipeline import pre_processing
from ML_pipeline import vectorizing_dataset
from ML_pipeline import topic_modeling
from ML_pipeline import predict_topic
from ML_pipeline import lsa_model as lsa_model_main
from ML_pipeline import predict_lsa
from ML_pipeline import utils
from ML_pipeline import tuning_lda


s3_resource = boto3.resource('s3', region_name='us-east-1')
s3_client = boto3.client('s3', region_name='us-east-1')


def start_process(train_documents, test_documents, path):
    train_documents['clean_document'] = train_documents['document'].progress_apply(
        lambda x: pre_processing.clean_documents(x)[0])
    test_documents['clean_document'] = test_documents['document'].progress_apply(
        lambda x: pre_processing.clean_documents(x)[0])
    # New column having the cleaned tokens
    train_documents['clean_token'] = train_documents['document'].progress_apply(
        lambda x: pre_processing.clean_documents(x)[1])
    test_documents['clean_token'] = test_documents['document'].progress_apply(
        lambda x: pre_processing.clean_documents(x)[1])

    # train_documents.to_csv('output/train_documents.csv', index = False)
    # test_documents.to_csv('output/test_documents.csv', index = False)

    # Transforming dataset into

    # Count Vectorizer
    count_vect, count_vect_text = vectorizing_dataset.transform_dataset(
        train_documents, 'clean_document', 'count')
    count_vectorized_test = count_vect.transform(
        test_documents['clean_document'])
    # TFIDF Vectorizer

    tfidf_vect, tfidf_vect_text = vectorizing_dataset.transform_dataset(
        train_documents, 'clean_token', 'tfidf')
    tfidf_vectorized_test = tfidf_vect.transform(test_documents['clean_token'])

    # Topic Modeling
    # LSA
    print("--------------LSA starts-------------------")
    lsa_model, lsa_top = lsa_model_main.lsa_model(
        tfidf_vect_text, 'output/lsa_model_trained.pkl')
    documet_topic_lsa = predict_lsa.topics_document(
        model_output=lsa_top, n_topics=10, data=train_documents)

    lsa_keys = utils.get_keys(lsa_top)
    lsa_categories, lsa_counts = utils.keys_to_counts(lsa_keys)

    print("----------------LSA ends--------------------")

    # LDA
    print("--------------LDA starts-------------------")
    lda_model, lda_model_output = topic_modeling.modeling(
        count_vect_text, 'count', model_path='output/lda_trained.pkl')

    '''
	# Takes too much time. Run this if you have efficient computer CPU.
	search_params = {'n_components': [10, 15, 20], 'learning_decay': [.5, .7, .9]}
	best_lda_model = tuning_lda.tune_lda(search_params, count_vect_text, "output/best_lda_model.pkl" )
	'''
    print("--------------LDA ends---------------------")
    # ## NMF
    print("--------------NMF starts---------------------")
    nmf_model, nmf_model_output = topic_modeling.modeling(
        tfidf_vect_text, 'tfidf', model_path='output/nmf_trained.pkl')

    print("--------------NMF ends---------------------")
    # # # Predict topic

    # LDA
    topic_seris_lda = predict_topic.topic_document(
        lda_model, count_vectorized_test, 10)
    # NMF
    topic_seris_nmf = predict_topic.topic_document(
        nmf_model, tfidf_vectorized_test, 13)

    # ## Exporting the dataset with the topic attached

    test_documents['index'] = [i for i in range(len(test_documents))]
    # LDA
    test_documents_lda = pd.merge(
        test_documents[['index', 'document']], topic_seris_lda, on=['index'], how='left')
    # NMF
    test_documents_nmf = pd.merge(
        test_documents[['index', 'document']], topic_seris_nmf, on=['index'], how='left')

    print(test_documents_lda.head(2))
    print(test_documents_nmf.head(2))
    BUCKET = os.environ.get("BUCKET_NAME")
    
    # Upload the result csvs to the S3 output path
    csv_buffer = StringIO()
    test_documents_lda[['document','dominant_topic']].to_csv(csv_buffer)
    s3_resource.Object(BUCKET, path+'/'+'test_lda.csv').put(Body=csv_buffer.getvalue())

    csv_buffer = StringIO()
    test_documents_nmf[['document','dominant_topic']].to_csv(csv_buffer)
    s3_resource.Object(BUCKET, path+'/'+'test_nmf.csv').put(Body=csv_buffer.getvalue())

    print('script completed successfully')


def pre_process(s3_path):
    # Dowloadin the CSV input files from the S3 and converting them as Dataframes
    response = s3_client.get_object(
        Bucket=os.environ.get("BUCKET_NAME"), Key=s3_path)
    print("CONTENT TYPE: " + response['ContentType'])
    print(response)
    TESTDATA = StringIO(response['Body'].read().decode("utf-8"))
    dataset = pd.read_csv(TESTDATA)
    dataset = dataset.sample(frac=1.0)
    train_documents = dataset[:int(len(dataset)*0.9)]
    test_documents = dataset[int(len(dataset)*0.9):]
    return train_documents, test_documents


def convert_documents(s3_path, path):
    try:
        train_documents, test_documents = pre_process(s3_path)
        start_process(train_documents, test_documents, path)
        return 200, {"status": "Document processed successfully"}
    except Exception as error:
        print("Error details => ", error)
        # Incase of error, the error details will be uploaded to the s3 output path i.e path parameter
        content=f"processing failed, details => {str(error)},\n traceback = >{str(traceback.format_exc())}"
        s3_resource.Object(os.environ.get("BUCKET_NAME"), path+'/error/error_details.txt').put(Body=content)
