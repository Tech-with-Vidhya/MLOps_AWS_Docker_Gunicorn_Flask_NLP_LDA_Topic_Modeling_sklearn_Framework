from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
import logging
import warnings
import joblib
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


"""
This function is used to read a transformed_data, model type, type of vectorizer
and return model object with the array of topics for each document
Arguments - 
    Output : object of model, array of topics
"""
def modeling(transformed_data, vect_type, model_path):

    try:

        if vect_type == 'count':
            model = LatentDirichletAllocation(max_iter=10, learning_method='online', learning_offset=50.,random_state=0, learning_decay = 0.7, n_components = 10)
            model_output = model.fit(transformed_data)
            joblib.dump(model, model_path)
            print(f'LDA Model saved in {model_path}')
            return model, model_output
        
        elif vect_type =='tfidf':
            model = NMF(n_components=13, random_state=43, init='nndsvd')
            model_output = model.fit(transformed_data)
            joblib.dump(model, model_path)
            print(f'NMF Model saved in {model_path}')

            return model, model_output

    except Exception as e:
        logger.error(f"Error occured in topic modeling: {e}", exc_info = True)
        pass