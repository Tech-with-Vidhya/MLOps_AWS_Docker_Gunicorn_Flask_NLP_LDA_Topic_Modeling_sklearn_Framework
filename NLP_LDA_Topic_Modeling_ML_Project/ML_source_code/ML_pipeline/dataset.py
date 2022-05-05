import pandas as pd
import numpy as np 
import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# data_path = "C:/ProjectPro/Topic_modeling/input/documents.csv"

"""
This function is used to read data file from a certain path
Arguments - 
    Output : training documents, test documents
"""
def read_data(data_path):
    
    try:

        dataset = pd.read_csv(data_path)
        dataset = dataset.sample(frac=1.0)
        train_documents = dataset[:int(len(dataset)*0.9)]
        test_documents  = dataset[int(len(dataset)*0.9):]
        
        return train_documents, test_documents
        
    except Exception as e:
        logger.error(f"Error occured in reading data files: {e}", exc_info = True)
        pass  
