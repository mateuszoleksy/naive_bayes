# Naive Bayes Classifier
import math

''' So there is a problem with encoding the bigger features like length, the use the 50/50'''

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False
    
# Funkcja normalizacji etykiet Y (zachowana z Twojego szablonu)
def normalize_probs(probs):
    total = sum(probs)
    if total == 0:
        return [1.0 / len(probs)] * len(probs)
    return [p / total for p in probs]

def normalize_feature_by_max_half(X):
    """
    Normalizuje wszystkie cechy X do formatu binarnego (0/1).
    Wymaga, aby X było macierzą. Oblicza globalne maksimum i stosuje próg Max/2.
    """
    if not X:
        return []

    all_values = [v for row in X for v in row]
    if not all_values:
        return X
    
    max_val = max(all_values)
    if max_val <= 1.0:
        return [[int(v) for v in row] for row in X]

    threshold = max_val / 2.0
    normalized_X = []
    for row in X:
        normalized_row = [1 if val >= threshold else 0 for val in row]
        normalized_X.append(normalized_row)
        
    return normalized_X

def read_data(x_path="input_x.txt", y_path="input_y.txt"):
    """
    Wczytuje X i Y. Dane X są normalizowane binarnie (0/1) za pomocą progu Max/2.
    """
    feature_names = None
    X = []
    with open(x_path, "r", encoding="utf-8") as fx:
        lines = [line.strip() for line in fx if line.strip() != ""]
        if not lines:
            raise ValueError("Plik X jest pusty.")
            
        first_tokens = [tok.strip() for tok in lines[0].split(",")]
        data_lines = lines
        
        if not all(is_number(tok) for tok in first_tokens):
            feature_names = first_tokens
            data_lines = lines[1:]
        
        for ln in data_lines:
            toks = [tok.strip() for tok in ln.split(",")]
            if not all(is_number(tok) for tok in toks):
                raise ValueError(f"Nie wszystkie wartości cech są liczbami w wierszu: {ln}")
            X.append([float(tok) for tok in toks])

    X = normalize_feature_by_max_half(X)

    Y = []
    with open(y_path, "r", encoding="utf-8") as fy:
        for line in fy:
            line = line.strip()
            if line == "":
                continue
            tok = line.split(",")[0].strip()
            if not is_number(tok):
                raise ValueError(f"Etykieta musi być liczbą (0/1), znaleziono: {tok}")
            val = float(tok)
            if val not in (0.0, 1.0):
                raise ValueError(f"Etykiety muszą być 0 lub 1. Znaleziono: {val}")
            Y.append(int(val))
            
    if len(X) != len(Y):
        raise ValueError(f"Liczba wierszy X ({len(X)}) nie zgadza się z liczbą etykiet Y ({len(Y)}).")

    return X, Y, feature_names

def train_algorithm(X, Y, smoothing=1.0):
    """
    Trenuje NB na danych binarnych cechach (0/1).
    Zwraca: (model, prior_pos, prior_neg)
    model: dict mapping feature_index -> {"positive": P(feature=1|class=1), "negative": P(feature=1|class=0)}
    """
    n_samples = len(Y)
    if n_samples == 0:
        raise ValueError("Brak próbek do trenowania.")
    n_features = len(X[0])

    # liczba pozytywnych i negatywnych przykładów
    num_pos = sum(Y)
    num_neg = n_samples - num_pos
    prior_pos = num_pos / n_samples
    prior_neg = num_neg / n_samples

    model = {}
    for j in range(n_features):
        count_pos = 0
        count_neg = 0
        for i in range(n_samples):
            val = X[i][j]
            if val == 1 or val == 1.0:
                if Y[i] == 1:
                    count_pos += 1
                else:
                    count_neg += 1
        # Laplace smoothing (binary feature): (count + alpha) / (N_class + 2*alpha)
        if num_pos > 0:
            pos_prob = (count_pos + smoothing) / (num_pos + 2 * smoothing)
        else:
            pos_prob = 0.5

        if num_neg > 0:
            neg_prob = (count_neg + smoothing) / (num_neg + 2 * smoothing)
        else:
            neg_prob = 0.5

        model[j] = {"positive": pos_prob, "negative": neg_prob}

    return model, prior_pos, prior_neg

def predict(model, prior_pos, prior_neg, x_test):
    """
    x_test: pojedynczy wektor cech (lista wartości 0/1 długości n_features)
    Zwraca 1 lub 0 (przewidywana klasa).
    Używa log-probabilities, aby uniknąć underflow.
    """
    if prior_pos == 0:
        return 0
    if prior_neg == 0:
        return 1

    pos_log = math.log(prior_pos)
    neg_log = math.log(prior_neg)

    for j, val in enumerate(x_test):
        feat = model.get(j)
        if feat is None:
            raise KeyError(f"Brak informacji o cesze {j} w modelu.")
        p_pos = feat["positive"]
        p_neg = feat["negative"]
        # zabezpieczenie przed log(0)
        p_pos = max(min(p_pos, 1 - 1e-12), 1e-12)
        p_neg = max(min(p_neg, 1 - 1e-12), 1e-12)

        if val == 1 or val == 1.0:
            pos_log += math.log(p_pos)
            neg_log += math.log(p_neg)
        else:
            pos_log += math.log(1 - p_pos)
            neg_log += math.log(1 - p_neg)

    return 1 if pos_log > neg_log else 0

if __name__ == "__main__":
    # przykład użycia; jeśli pliki nie istnieją, możesz zamiast tego zdefiniować mały zestaw danych tutaj:
    try:
        X, Y, feature_names = read_data()
    except Exception as e:
        print("Błąd przy wczytywaniu danych:", e)
        # przykładowy zbiór
        X = [
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
            [0, 1, 0]
        ]
        Y = [1, 1, 0, 0]
        feature_names = None
        print("Używam przykładowego zbioru danych.")

    model, prior_pos, prior_neg = train_algorithm(X, Y, smoothing=1.0)
    print("Prior positive:", prior_pos, "Prior negative:", prior_neg)
    print("Model (feature -> {positive, negative}):")
    for k, v in model.items():
        name = f"feat_{k}" if not feature_names else feature_names[k]
        print(f" {name}: {v}")

    # test predykcji na pierwszym wierszu
    pred = predict(model, prior_pos, prior_neg, X[0])
    print("Predykcja dla pierwszego przykładu:", pred, "prawdziwa etykieta:", Y[0])
