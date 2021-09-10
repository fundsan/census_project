from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from aequitas.group import Group
from aequitas.preprocessing import preprocess_input_df
import aequitas as aq
from aequitas.plotting import Plot
import matplotlib.pyplot as plt
# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    gs_clf=GridSearchCV(SGDClassifier(),param_grid={'max_iter':(500,1000)})
    gs_clf.fit(X_train, y_train)
    return gs_clf


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta

def slice_inference(model,X, df, cat_feats, slice_feats='all'):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    slice_feats : list-like, string
        feature columns to get slices
        
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    
    if slice_feats=='all':
        slice_cols =list(X.columns)
    else:
        slice_cols = list(slice_feats)
    df['label_value'] = df['salary'].values

    df['score']=inference(model,X)
    # double-check that categorical columns are of type 'string'
    df[cat_feats] = df[cat_feats].astype(str)
    
    df, _ = preprocess_input_df(df[slice_feats+['score']+['label_value']])
    g = Group()
    xtab, _ = g.get_crosstabs(df)
    df[slice_cols]
    attr_xtab=xtab[xtab['attribute_name'].isin(slice_cols)]
    
    
    aqp = Plot()
    fig=aqp.plot_group_metric_all(attr_xtab, ncols=3,show_figure=False, min_group_size=.01)
    plt.savefig(os.path.abspath(os.getcwd())+'/images/slice_performance_output.png')
    return attr_xtab
    

def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]

    return model.predict(X)
