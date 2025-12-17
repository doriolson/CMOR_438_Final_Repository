import pytest
import numpy as np
from rice_ml.supervised_learning.ensemble_methods import VotingClassifier, LogisticRegression, DecisionTreeClassifier

def test_voting_ensemble_logic():
    # Create simple linear data
    X = np.array([[1, 2], [2, 3], [8, 8], [9, 10]])
    y = np.array([0, 0, 1, 1])
    
    lr = LogisticRegression()
    dt = DecisionTreeClassifier(max_depth=2)
    
    ensemble = VotingClassifier(estimators=[('lr', lr), ('dt', dt)])
    ensemble.fit(X, y)
    
    preds = ensemble.predict(X)
    assert preds.shape == (4,)
    assert np.all(preds == y)

def test_ensemble_probability_shape():
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)
    
    ensemble = VotingClassifier(estimators=[
        ('lr', LogisticRegression()), 
        ('dt', DecisionTreeClassifier())
    ])
    ensemble.fit(X, y)
    
    probs = ensemble.predict_proba(X)
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)