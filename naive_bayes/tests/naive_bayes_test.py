import time
import numpy as np
import sys
from pathlib import Path

# Scikit-learn imports for comparison
from sklearn.naive_bayes import BernoulliNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Add the src directory to the system path to allow importing the custom implementation
project_root = Path(__file__).parent.parent 
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from naive_bayes.naive_bayes import train_algorithm, predict

def check_accuracy():
    # This function compares the accuracy of the custom implementation against Scikit-Learn.
    # It generates a dataset, splits it into training and testing sets,
    # and ensures the custom model's predictions match the library's performance.
    print("\n" + "="*50)
    print("TEST 1: ACCURACY COMPARISON (200 Samples)")
    print("="*50)

    print("[*] Generating dataset...")
    # Generate 200 samples with 15 features, where 10 are informative
    X, Y = make_classification(
        n_samples=200, 
        n_features=15, 
        n_informative=10, 
        random_state=42
    )
    # Binarize data (set to 0 or 1 based on median) for Bernoulli Naive Bayes
    X = (X > np.median(X, axis=0)).astype(int)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    print(f"[*] Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

    # Train and predict with the custom implementation
    print("\n[Custom Implementation]")
    model, p_pos, p_neg = train_algorithm(X_train, Y_train)
    my_preds = [predict(model, p_pos, p_neg, row) for row in X_test]
    my_acc = accuracy_score(Y_test, my_preds)
    print(f" -> Accuracy: {my_acc:.4f} ({my_acc*100:.2f}%)")

    # Train and predict with the reference Scikit-Learn
    print("\n[Scikit-Learn Implementation]")
    clf = BernoulliNB(alpha=1.0)
    clf.fit(X_train, Y_train)
    sk_preds = clf.predict(X_test)
    sk_acc = accuracy_score(Y_test, sk_preds)
    print(f" -> Accuracy: {sk_acc:.4f} ({sk_acc*100:.2f}%)")

    # verify that the accuracy is within a 5% margin of error
    if abs(my_acc - sk_acc) < 0.05:
        print("\n[SUCCESS] Results match!")
    else:
        print("\n[WARNING] Results differ significantly.")


def check_speed():
    # This function benchmarks the execution time of the custom implementation and the Scikit-Learn implementation.
    print("\n" + "="*50)
    print("TEST 2: EXECUTION TIME (5000 Samples)")
    print("="*50)
    
    X_large, Y_large = make_classification(n_samples=5000, n_features=20, random_state=42)
    X_large = (X_large > np.median(X_large, axis=0)).astype(int)
    
    print(f"[*] Benchmarking with {len(X_large)} samples...")

    # Measure time for custom implementation (training + prediction loop)
    start = time.time()
    model, p1, p2 = train_algorithm(X_large, Y_large)
    _ = [predict(model, p1, p2, x) for x in X_large]
    end = time.time()
    my_time = end - start

    # Measure time for Scikit-Learn implementation (training + vectorized prediction)
    start = time.time()
    clf = BernoulliNB(alpha=1.0)
    clf.fit(X_large, Y_large)
    _ = clf.predict(X_large)
    end = time.time()
    sk_time = end - start

    print(f"\nCustom Time:      {my_time:.5f} seconds")
    print(f"Scikit-Learn Time: {sk_time:.5f} seconds")
    
    print("-" * 30)
    if sk_time > 0:
        ratio = my_time / sk_time
        print(f"RESULT: Scikit-Learn is {ratio:.1f}x faster.")
    print("-" * 30)


def check_laplace_smoothing():
    # This function tests if the algorithm correctly handles unseen features using Laplace Smoothing.
    # Without smoothing, probabilities would be 0, potentially causing math errors or incorrect predictions.
    print("\n" + "="*50)
    print("TEST 3: LAPLACE SMOOTHING (The 'Zero Prob' Trap)")
    print("="*50)
    
    print("[*] Training on mutually exclusive features...")
    # Create a dataset where Class 0 only has Feature 1, and Class 1 only has Feature 0
    X_train = [
        [0, 1], [0, 1], [0, 1], # Class 0 samples
        [1, 0], [1, 0], [1, 0]  # Class 1 samples
    ]
    Y_train = [0, 0, 0, 1, 1, 1]

    try:
        model, p_pos, p_neg = train_algorithm(X_train, Y_train)
    except Exception as e:
        print(f"[FAIL] Training crashed! Likely failed to handle 0 counts.\nError: {e}")
        return

    # Create a 'Trap' sample [1, 1]. Both features are technically "impossible" for their respective classes
    # based on the training data alone.
    trap_sample = [1, 1]
    
    print(f"[*] Predicting on trap sample {trap_sample}...")
    
    try:
        prediction = predict(model, p_pos, p_neg, trap_sample)
        
        # Check internal model values for exact 0.0s (which implies no smoothing)
        smoothing_verified = False
        if isinstance(model, dict):
            zeros_found = any(v == 0.0 for v in model.values())
            if zeros_found:
                print("\n[WARNING] Found exact 0.0 probabilities in the model.")
                print("Likely missing Laplace Smoothing (alpha + 1).")
            else:
                smoothing_verified = True
        
        if prediction in [0, 1]:
            print(f" -> Prediction: Class {prediction}")
            if smoothing_verified:
                 print("[SUCCESS] Laplace smoothing works.")
            else:
                 print("[PASS] The code ran without crashing (Basic check passed).")
        else:
            print(f"[FAIL] Prediction returned weird value: {prediction}")

    except ValueError as e:
            print(f"\n[FAIL] crashed with ValueError: {e}")
    except Exception as e:
        print(f"\n[FAIL] crashed with error: {e}")


if __name__ == "__main__":
    check_accuracy()
    check_speed()
    check_laplace_smoothing()