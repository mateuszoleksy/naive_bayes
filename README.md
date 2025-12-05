# naive_bayes

A lightweight Naive Bayes classifier example in Python — a simple implementation for binary features (0/1).

Purpose: educational code demonstrating a Naive Bayes classifier with Laplace smoothing and basic feature normalization.

Quick start
- Source module: `src/naive_bayes/naive_bayes.py`.
- Example input files: `src/naive_bayes/input_x.txt` and `src/naive_bayes/input_y.txt`. Run the script from `src/naive_bayes` or copy those files into your working directory.

Installation (local / development)
- Create a virtual environment and install dependencies if needed:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Alternatively, install the package in editable mode (helps imports during development):

```powershell
pip install -e .
```

Running
- Run directly without installation by changing to the module directory:

```powershell
cd src\naive_bayes
python naive_bayes.py
```

- If you installed the package (`pip install -e .`), run as a module:

```powershell
python -m naive_bayes.naive_bayes
```

Note: The script expects `input_x.txt` and `input_y.txt` in the current working directory. If they are missing, the script will fall back to a built-in example dataset.

What the code does
- `read_data(x_path='input_x.txt', y_path='input_y.txt')` — reads feature matrix X and label vector Y. If the first row of the X file contains non-numeric values, it is treated as a `feature_names` header.
- `normalize_feature_by_max_half(X)` — binarizes features using a threshold of `max(X)/2`.
- `train_algorithm(X, Y, smoothing=1.0)` — trains the Naive Bayes model, returning the model structure and class priors.
- `predict(model, prior_pos, prior_neg, x_test)` — predicts class 0 or 1 using log-probabilities.

Running the tests
- Tests are located in the `tests/` directory and use `pytest`.

```powershell
# install pytest if needed
pip install pytest
# run tests
pytest -q
```

You can also use `tox` if configured for the project:

```powershell
tox
```

Repository structure (key files)
- `src/naive_bayes/naive_bayes.py`: classifier implementation with a sample `if __name__ == '__main__'` block.
- `src/naive_bayes/input_x.txt`, `src/naive_bayes/input_y.txt`: example input files (if present).
- `tests/`: unit tests (`pytest`).
- `setup.py`, `pyproject.toml`: packaging configuration.

Usage example
```python
from naive_bayes.naive_bayes import read_data, train_algorithm, predict

X, Y, feature_names = read_data('input_x.txt', 'input_y.txt')
model, prior_pos, prior_neg = train_algorithm(X, Y)
print(predict(model, prior_pos, prior_neg, X[0]))
```

Contributing
- Report issues and open pull requests.
- Follow PEP8 and add tests for new features.

License
- The project includes a `LICENSE` file in the repository root — follow its terms.