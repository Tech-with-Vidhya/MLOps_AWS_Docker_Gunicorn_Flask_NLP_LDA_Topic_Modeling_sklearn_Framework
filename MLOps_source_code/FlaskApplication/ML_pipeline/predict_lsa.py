import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

def topics_document(model_output, n_topics , data):
    '''
    returns a dataframe for each document having topic weightages
    and the dominant topic for each doc. 
    '''
    
    # column names
    topicnames = ["Topic" + str(i) for i in range(n_topics)]
    # index names
    docnames = ["Doc" + str(i) for i in range(len(data))]
    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(model_output, 2), columns=topicnames, index=docnames)
    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic["dominant_topic"] = dominant_topic
    
    return df_document_topic