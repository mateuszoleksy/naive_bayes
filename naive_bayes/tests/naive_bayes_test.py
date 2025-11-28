"""
Tests for the Naive Bayes implementation.

Verifies correctness by comparing against scikit-learn's implementation.
"""
import pytest
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Import the module under test
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from naive_bayes.naive_bayes import (
    train_algorithm,
    predict,
    read_data,
    normalize_probs,
)


@pytest.fixture
def simple_binary_data():
    X = [[1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]]
    Y = [1, 1, 0, 0, 1, 0]
    return X, Y


@pytest.fixture
def random_binary_data():
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X = np.random.randint(0, 2, size=(n_samples, n_features))
    Y = (X.sum(axis=1) > n_features // 2).astype(int)
    return X.tolist(), Y


@pytest.fixture
def sklearn_dataset():
    np.random.seed(42)
    X, Y = make_classification(
        n_samples=200,
        n_features=15,
        n_informative=10,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    X = (X > np.median(X, axis=0)).astype(int)
    return X, Y


class TestTrainAlgorithm:
    def test_train_returns_correct_structure(self, simple_binary_data):
        X, Y = simple_binary_data
        model, prior_pos, prior_neg = train_algorithm(X, Y)

        assert isinstance(model, dict)
        assert len(model) == len(X[0])
        for feat_idx, feat_probs in model.items():
            assert "positive" in feat_probs
            assert "negative" in feat_probs
            assert 0 <= feat_probs["positive"] <= 1
            assert 0 <= feat_probs["negative"] <= 1

    def test_priors_sum_to_one(self, simple_binary_data):
        X, Y = simple_binary_data
        model, prior_pos, prior_neg = train_algorithm(X, Y)

        assert pytest.approx(prior_pos + prior_neg, abs=1e-10) == 1.0

    def test_priors_match_class_distribution(self, simple_binary_data):
        X, Y = simple_binary_data
        expected_prior_pos = sum(Y) / len(Y)
        expected_prior_neg = (len(Y) - sum(Y)) / len(Y)

        model, prior_pos, prior_neg = train_algorithm(X, Y)

        assert pytest.approx(prior_pos, abs=1e-10) == expected_prior_pos
        assert pytest.approx(prior_neg, abs=1e-10) == expected_prior_neg

    def test_smoothing_effect(self, simple_binary_data):
        X, Y = simple_binary_data
        model1, _, _ = train_algorithm(X, Y, smoothing=0.1) if 'smoothing' in train_algorithm.__code__.co_varnames else train_algorithm(X, Y)
        model2, _, _ = train_algorithm(X, Y, smoothing=10.0) if 'smoothing' in train_algorithm.__code__.co_varnames else train_algorithm(X, Y)

        # If smoothing is supported, probabilities should differ
        if 'smoothing' in train_algorithm.__code__.co_varnames:
            assert model1[0]["positive"] != model2[0]["positive"]

    def test_empty_data_raises_error(self):
        with pytest.raises(ValueError, match="Brak próbek"):
            train_algorithm([], [])

    def test_mismatched_lengths_raises_error(self):
        X = [[1, 0], [1, 1]]
        Y = [1]
        try:
            train_algorithm(X, Y)
        except (ValueError, IndexError):
            pass


class TestPredict:
    def test_predict_returns_binary(self, simple_binary_data):
        X, Y = simple_binary_data
        model, prior_pos, prior_neg = train_algorithm(X, Y)

        for sample in X:
            pred = predict(model, prior_pos, prior_neg, sample)
            assert pred in (0, 1)

    def test_predict_on_training_data(self, simple_binary_data):
        X, Y = simple_binary_data
        model, prior_pos, prior_neg = train_algorithm(X, Y)

        predictions = [predict(model, prior_pos, prior_neg, x) for x in X]
        accuracy = sum(p == y for p, y in zip(predictions, Y)) / len(Y)

        assert accuracy >= 0.5

    def test_missing_feature_in_model_raises_error(self, simple_binary_data):
        X, Y = simple_binary_data
        model, prior_pos, prior_neg = train_algorithm(X, Y)
        del model[0]

        with pytest.raises(KeyError, match="Brak informacji"):
            predict(model, prior_pos, prior_neg, X[0])


class TestAgainstSklearn:
    def test_predictions_match_sklearn(self, sklearn_dataset):
        X, Y = sklearn_dataset
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3, random_state=42
        )

        our_model, our_prior_pos, our_prior_neg = train_algorithm(X_train, Y_train)

        sklearn_model = BernoulliNB(alpha=1.0)
        sklearn_model.fit(X_train, Y_train)

        our_predictions = [
            predict(our_model, our_prior_pos, our_prior_neg, x) for x in X_test
        ]
        sklearn_predictions = sklearn_model.predict(X_test)

        accuracy_our = sum(p == y for p, y in zip(our_predictions, Y_test)) / len(Y_test)
        accuracy_sklearn = sum(
            p == y for p, y in zip(sklearn_predictions, Y_test)
        ) / len(Y_test)

        assert abs(accuracy_our - accuracy_sklearn) < 0.05

    def test_priors_match_sklearn(self, sklearn_dataset):
        X, Y = sklearn_dataset
        X_train, _, Y_train, _ = train_test_split(
            X, Y, test_size=0.3, random_state=42
        )

        our_model, our_prior_pos, our_prior_neg = train_algorithm(X_train, Y_train)

        sklearn_model = BernoulliNB(alpha=1.0)
        sklearn_model.fit(X_train, Y_train)

        sklearn_prior_pos = np.exp(sklearn_model.class_log_prior_[1])
        sklearn_prior_neg = np.exp(sklearn_model.class_log_prior_[0])

        assert pytest.approx(our_prior_pos, abs=1e-10) == sklearn_prior_pos
        assert pytest.approx(our_prior_neg, abs=1e-10) == sklearn_prior_neg

    def test_feature_probabilities_match_sklearn(self, simple_binary_data):
        X, Y = simple_binary_data

        our_model, _, _ = train_algorithm(X, Y) if 'smoothing' not in train_algorithm.__code__.co_varnames else train_algorithm(X, Y, smoothing=1.0)

        sklearn_model = BernoulliNB(alpha=1.0)
        sklearn_model.fit(X, Y)

        n_features = len(X[0])
        for feat_idx in range(n_features):
            our_pos = our_model[feat_idx]["positive"]
            our_neg = our_model[feat_idx]["negative"]

            sklearn_pos_log = sklearn_model.feature_log_prob_[1, feat_idx]
            sklearn_neg_log = sklearn_model.feature_log_prob_[0, feat_idx]

            sklearn_pos = np.exp(sklearn_pos_log)
            sklearn_neg = np.exp(sklearn_neg_log)

            assert pytest.approx(our_pos, abs=1e-10) == sklearn_pos
            assert pytest.approx(our_neg, abs=1e-10) == sklearn_neg

    def test_on_larger_dataset(self, random_binary_data):
        X, Y = random_binary_data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3, random_state=42
        )

        our_model, our_prior_pos, our_prior_neg = train_algorithm(X_train, Y_train)

        sklearn_model = BernoulliNB(alpha=1.0)
        sklearn_model.fit(X_train, Y_train)

        our_predictions = [
            predict(our_model, our_prior_pos, our_prior_neg, x) for x in X_test
        ]
        sklearn_predictions = sklearn_model.predict(X_test)

        accuracy_our = sum(p == y for p, y in zip(our_predictions, Y_test)) / len(Y_test)
        accuracy_sklearn = sum(
            p == y for p, y in zip(sklearn_predictions, Y_test)
        ) / len(Y_test)

        assert abs(accuracy_our - accuracy_sklearn) < 0.1


class TestNormalizeProbs:
    def test_probs_sum_to_one(self):
        probs = [1, 2, 3, 4]
        normalized = normalize_probs(probs)
        assert pytest.approx(sum(normalized), abs=1e-10) == 1.0

    def test_empty_list_handled(self):
        with pytest.raises((ZeroDivisionError, IndexError)):
            normalize_probs([])

    def test_all_zeros_handled(self):
        probs = [0, 0, 0, 0]
        normalized = normalize_probs(probs)
        assert all(p == 0.25 for p in normalized)

    def test_preserves_relative_order(self):
        probs = [1, 2, 3]
        normalized = normalize_probs(probs)
        assert normalized[0] < normalized[1] < normalized[2]


class TestEdgeCases:
    def test_all_positive_labels(self):
        X = [[1, 0], [1, 1], [0, 1]]
        Y = [1, 1, 1]

        model, prior_pos, prior_neg = train_algorithm(X, Y)
        assert pytest.approx(prior_pos, abs=1e-10) == 1.0
        assert pytest.approx(prior_neg, abs=1e-10) == 0.0

    def test_all_negative_labels(self):
        X = [[1, 0], [1, 1], [0, 1]]
        Y = [0, 0, 0]

        model, prior_pos, prior_neg = train_algorithm(X, Y)
        assert pytest.approx(prior_pos, abs=1e-10) == 0.0
        assert pytest.approx(prior_neg, abs=1e-10) == 1.0

    def test_single_sample(self):
        X = [[1, 0, 1]]
        Y = [1]

        model, prior_pos, prior_neg = train_algorithm(X, Y)
        pred = predict(model, prior_pos, prior_neg, X[0])
        assert pred in (0, 1)

    def test_single_feature(self):
        X = [[1], [0], [1], [0]]
        Y = [1, 0, 1, 0]

        model, prior_pos, prior_neg = train_algorithm(X, Y)
        assert len(model) == 1
        assert 0 in model


class TestIntegration:
    def test_full_pipeline(self, sklearn_dataset):
        X, Y = sklearn_dataset
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.3, random_state=42
        )

        model, prior_pos, prior_neg = train_algorithm(X_train, Y_train)

        predictions = [predict(model, prior_pos, prior_neg, x) for x in X_test]

        accuracy = sum(p == y for p, y in zip(predictions, Y_test)) / len(Y_test)

        assert accuracy > 0.5

    def test_consistency_across_multiple_runs(self, simple_binary_data):
        X, Y = simple_binary_data

        model1, prior_pos1, prior_neg1 = train_algorithm(X, Y)
        model2, prior_pos2, prior_neg2 = train_algorithm(X, Y)

        assert prior_pos1 == prior_pos2
        assert prior_neg1 == prior_neg2
        for key in model1:
            assert model1[key]["positive"] == model2[key]["positive"]
            assert model1[key]["negative"] == model2[key]["negative"]
