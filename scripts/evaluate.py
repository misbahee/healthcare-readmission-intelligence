import pandas as pd
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_classifier(
    name: str, 
    pipeline: Any, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_test: pd.Series, 
) -> dict:
    """
    Fits a pipeline and returns a dictionary of evaluation metrics.
    """
    pipeline.fit(X_train, y_train)
    
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    try:
        y_test_prob = pipeline.predict_proba(X_test)
    except AttributeError:
        y_test_prob = pipeline.decision_function(X_test)
    
    row = {'Model': name}
    
    for split, y_true, y_pred in [('Train', y_train, y_train_pred), 
                                  ('Test', y_test, y_test_pred)]:
        
        row[f'{split} Accuracy'] = round(accuracy_score(y_true, y_pred), 4)
        row[f'{split} Precision'] = round(precision_score(y_true, y_pred, zero_division=0), 4)
        row[f'{split} Recall'] = round(recall_score(y_true, y_pred, zero_division=0), 4)
        row[f'{split} F1'] = round(f1_score(y_true, y_pred, zero_division=0), 4)
    
    row['_pipeline'] = pipeline
    row['_test_pred'] = y_test_pred
    row['_test_prob'] = y_test_prob
    
    return row
