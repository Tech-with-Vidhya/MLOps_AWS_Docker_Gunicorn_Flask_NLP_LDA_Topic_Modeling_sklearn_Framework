import pandas as pd 
import numpy as np 
import joblib
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
def lsa_model(vect_text, model_path):
    lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)
    lsa_top   = lsa_model.fit_transform(vect_text)
    joblib.dump(lsa_model, model_path)
    print(f'LSA Model saved in {model_path}')
    return lsa_model,lsa_top