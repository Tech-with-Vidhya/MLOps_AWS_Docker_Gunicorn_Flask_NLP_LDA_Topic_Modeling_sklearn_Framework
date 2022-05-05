from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import joblib

def tune_lda(search_params, data_vectorized, model_path):
    # Init the Model
    lda = LatentDirichletAllocation(max_iter=5, learning_method='online', learning_offset=50.,random_state=0)
    # Init Grid Search Class
    model = GridSearchCV(lda, param_grid=search_params)
    # Do the Grid Search
    model.fit(data_vectorized)
    # Best Model
    best_lda_model = model.best_estimator_
    # Model Parameters
    print("Best Model's Params: ", model.best_params_)
    # Log Likelihood Score
    print("Best Log Likelihood Score: ", model.best_score_)
    # Perplexity
    print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))
    joblib.dump(best_lda_model, model_path)
    print(f"Best LDA model is saved in {model_path}")
    return best_lda_model