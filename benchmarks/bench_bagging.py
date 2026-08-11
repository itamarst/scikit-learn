"""
Benchmark ``BaggingClassifer`` and ``BaggingRegressor`` with parallelism.
"""

import numpy as np

from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split

n_samples = 100_000
dim = 10
n_classes = 10
X = np.random.randn(n_samples, dim)
y = np.random.randint(0, n_classes, (n_samples,))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.99, random_state=42
)
clf = BaggingClassifier(n_jobs=-1)
clf.fit(X_train, y_train).predict(X_test)
clf = BaggingRegressor(n_jobs=-1)
clf.fit(X_train, y_train).predict(X_test)
