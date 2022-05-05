import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


"""
This function returns a dataframe for each document having topic weightages
    and the dominant topic for each doc. 
Arguments - 
    Output : dataframe having dominant topic
"""
def topic_document(model, data,n_topics):
    
    try:
        #predict topics for each document
        model_output = model.transform(data)
        # column names
        topicnames = ["Topic" + str(i) for i in range(n_topics)]
        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(model_output, 2), columns=topicnames)
        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1)
        df_document_topic["dominant_topic"] = dominant_topic
        df_document_topic['index'] = [i for i in range(len(df_document_topic))]
        
        return df_document_topic[['index','dominant_topic']]

    except Exception as e:
        logger.error(f"Error occured in predicting the topic: {e}", exc_info = True)
        pass