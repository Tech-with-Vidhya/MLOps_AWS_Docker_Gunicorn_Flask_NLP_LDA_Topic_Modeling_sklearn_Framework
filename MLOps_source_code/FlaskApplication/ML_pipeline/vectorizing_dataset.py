import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


"""
This function is used to read a dataframe, column name and type of vectorizer 
and return vectorizer object with the transformed dataset
Arguments - 
    Output : object of vectorizer, transformed dataset
"""
def transform_dataset(data, col, type_vect):

    try:
        if type_vect == 'tfidf':
            tfidf_vect        = TfidfVectorizer(
                                    min_df=3,
                                    max_df=0.85,
                                    max_features=4000,
                                    ngram_range=(1, 2),
                                    preprocessor=' '.join
                                    )
            tfidf_vect_text   = tfidf_vect.fit_transform(data[col])

            return tfidf_vect, tfidf_vect_text

        elif type_vect == 'count':
            count_vect = CountVectorizer(
                                    analyzer='word',       
                                    min_df=10,# minimum reqd occurences of a word 
                                    stop_words='english',             # remove stop words
                                    lowercase=True,                   # convert all words to lowercase
                                    token_pattern='[a-zA-Z0-9]{3,}'
                                    )  # num chars > 3
            count_vect_text = count_vect.fit_transform(data[col])

            return count_vect, count_vect_text

    except Exception as e:
        logger.error(f"Error occured in transforming dataset: {e}", exc_info = True)
        pass  