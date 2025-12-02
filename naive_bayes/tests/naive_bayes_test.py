
"""
Tests for Naive Bayes implementation
"""

import pytest
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import the functions to test
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
from naive_bayes.naive_bayes import train_algorithm, predict


# Simple fixture with a tiny binary dataset
@pytest.fixture
def simple_binary_data():
    X = [[1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0]]  # 4 samples, 3 features
    Y = [1, 1, 0, 0]  # labels
    return X, Y


def test_priors_match_class_distribution(simple_binary_data):
    """
    Priors should match the fraction of positive/negative labels.
    """
    # get the tiny dataset
    X, Y = simple_binary_data
    # train the model (returns model and class priors)
    _, prior_pos, prior_neg = train_algorithm(X, Y)
    # the two priors should add up to 1.0
    assert pytest.approx(prior_pos + prior_neg, abs=1e-10) == 1.0
    # prior for positive should equal fraction of positive labels
    assert pytest.approx(prior_pos, abs=1e-10) == sum(Y) / len(Y)


def test_predict_returns_binary(simple_binary_data):
    """
    Predict must return either 0 or 1 for each input sample.
    """
    # load data and train quickly
    X, Y = simple_binary_data
    model, prior_pos, prior_neg = train_algorithm(X, Y)
    # each prediction should be a 0 or 1 (no weird values)
    for sample in X:
        p = predict(model, prior_pos, prior_neg, sample)
        assert p in (0, 1)


# ----------------------
# scikit-learn comparison
# ----------------------

@pytest.fixture
def sklearn_dataset():
    # set the random seed so the test is repeatable
    np.random.seed(42)
    # generate a synthetic classification problem (continuous features)
    # create a synthetic classification problem with these settings:
    # - n_samples: how many data points (rows) we want
    # - n_features: total number of features (columns) produced
    # - n_informative: how many features actually carry signal for the label
    # - n_redundant: features that are linear combos of informative ones (here none)
    # - n_classes: number of target classes (2 for binary classification)
    # - random_state: seed so results are reproducible across runs
    X, Y = make_classification(
        n_samples=200,      
        n_features=15,      
        n_informative=10,   
        n_redundant=0,      
        n_classes=2,        
        random_state=42,    
    )
    # casual binarization: turn continuous features into 0/1 by thresholding at the median
    # (this makes the data compatible with BernoulliNB and our simple implementation)
    X = (X > np.median(X, axis=0)).astype(int)
    # return features and labels
    return X, Y


def test_predictions_match_sklearn(sklearn_dataset):
    """
    Compare overall accuracy with scikit-learn's BernoulliNB.
    """
    # prepare train/test split
    X, Y = sklearn_dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    # train our simple NB
    our_model, our_prior_pos, our_prior_neg = train_algorithm(X_train, Y_train)
    # train sklearn's BernoulliNB for comparison
    sklearn_model = BernoulliNB(alpha=1.0)
    sklearn_model.fit(X_train, Y_train)
    # make predictions with both
    our_predictions = [predict(our_model, our_prior_pos, our_prior_neg, x) for x in X_test]
    sklearn_predictions = sklearn_model.predict(X_test)
    # calculate accuracy
    acc_ours = sum(p == y for p, y in zip(our_predictions, Y_test)) / len(Y_test)
    acc_sklearn = sum(p == y for p, y in zip(sklearn_predictions, Y_test)) / len(Y_test)
    # allow a little slack because implementations differ slightly
    assert abs(acc_ours - acc_sklearn) < 0.1


def test_priors_match_sklearn(sklearn_dataset):
    """Check that our computed priors match sklearn's class priors.

    Casual: we pull class priors from sklearn (they store log-priors),
    exponentiate them and compare to our simple counts.
    """
    # split and train
    X, Y = sklearn_dataset
    X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.3, random_state=42)
    _, our_prior_pos, our_prior_neg = train_algorithm(X_train, Y_train)
    # sklearn's priors come as log-probabilities, so we exp() them
    sklearn_model = BernoulliNB(alpha=1.0)
    sklearn_model.fit(X_train, Y_train)
    sklearn_prior_pos = np.exp(sklearn_model.class_log_prior_[1])
    sklearn_prior_neg = np.exp(sklearn_model.class_log_prior_[0])
    # compare with tiny tolerance
    assert pytest.approx(our_prior_pos, abs=1e-10) == sklearn_prior_pos
    assert pytest.approx(our_prior_neg, abs=1e-10) == sklearn_prior_neg

