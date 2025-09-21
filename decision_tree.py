import numpy as np
from typing import Self

"""
This is a suggested template and you do not need to follow it. You can change any part of it to fit your needs.
There are some helper functions that might be useful to implement first.
At the end there is some test code that you can use to test your implementation on synthetic data by running this file.
"""

def count(y: np.ndarray) -> np.ndarray:
    """
    Count unique values in y and return the proportions of each class sorted by label in ascending order.
    Example:
        count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])) -> np.array([0.2, 0.3, 0.4, 0.1])
    """
    _ , counts = np.unique(y, return_counts=True)
    return counts / counts.sum()

def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """
    return 1 - np.sum(count(y)**2)

def entropy(y: np.ndarray) -> float:
    """
    Return the entropy of a given NumPy array y.
    """
    probs = count(y)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def split(x: np.ndarray, value: float) -> np.ndarray:
    """
    Return a boolean mask for the elements of x satisfying x <= value.
    Example:
        split(np.array([1, 2, 3, 4, 5, 2]), 3) -> np.array([True, True, True, False, False, True])
    """
    return x <= value

def most_common(y: np.ndarray) -> int:
    """
    Return the most common element in y.
    Example:
        most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])) -> 4
    """
    return np.bincount(y).argmax()

def impurity(criterion: str, y: np.ndarray) -> float:
    """
    Return the impurity of y using the specified criterion.
    """
    if criterion == "gini":
        return gini_index(y)
    return entropy(y)

def best_split(X: np.ndarray, y: np.ndarray, criterion: str, feature_indices) -> tuple[int, float]:
    """
    Given a NumPy array X of features and a NumPy array y of integer labels,
    return the index of the feature and the threshold value to split on that maximizes the information gain.
    """
    n_samples, n_features = X.shape
    if feature_indices is None:
        feature_indices = range(n_features)

    best_ig = -1
    best_feature = None
    best_threshold = None

    current_impurity = impurity(criterion, y)

    for i in feature_indices:
        thresholds = np.unique(X[:, i])
        for t in thresholds:
            mask = X[:, i] <= t
            left, right = y[mask], y[~mask]
            if len(left) == 0 or len(right) == 0:
                continue
            ig = current_impurity - (len(left)/n_samples*impurity(criterion, left) + len(right)/n_samples*impurity(criterion, right))
            
            if ig > best_ig:
                best_ig = ig
                best_feature = i
                best_threshold = t

    return best_feature, best_threshold

class Node:
    """
    A class to represent a node in a decision tree.
    If value != None, then it is a leaf node and predicts that value, otherwise it is an internal node (or root).
    The attribute feature is the index of the feature to split on, threshold is the value to split at,
    and left and right are the left and right child nodes.
    """
    
    def __init__(
        self,
        feature: int = 0,
        threshold: float = 0.0,
        left: int | Self | None = None,
        right: int | Self | None = None,
        value: int | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        """
        Return True if the node is a leaf node.
        """
        return self.value is not None

class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        criterion: str = "entropy",
        max_features: str | None = None,
        random_state: int | None = None,
    ) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def get_params(self, deep=True):
        """
        Return the parameters of the DecisionTree instance as a dictionary.
        """
        return {"criterion": self.criterion, "max_depth": self.max_depth, "random_state": self.random_state}

    def set_params(self, **params):
        """
        Set the parameters of the DecisionTree instance using keyword arguments.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def features_subset(self, n_features: int) -> np.ndarray:
        """
        Select a random subset of feature indices based on max_features.
        """
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features
        return self.rng.choice(
            n_features, size=max_features, replace=False
        )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        """
        Fit the DecisionTree model on the training data X and labels y.
        """
        self.root = self._fit(X, y, 0)

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth
    ):
        
        """
        This functions learns a decision tree given (continuous) features X and (integer) labels y.
        """

        if len(np.unique(y)) == 1:
            return Node(value=y[0])
        
        if X.shape[0] == 0:
            return Node(value= most_common(y))

        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value= most_common(y))

        feature_indices = self.features_subset(X.shape[1])

        best_feature, best_threshold = best_split(X, y, self.criterion, feature_indices)

        if best_feature is None:
            return Node(value= most_common(y))

        mask = split(X[:, best_feature], best_threshold)
        left_node = self._fit(X[mask], y[mask], depth + 1)
        right_node = self._fit(X[~mask], y[~mask], depth + 1)

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_node,
            right=right_node,
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        return np.array([self._predict(x, self.root) for x in X])

    def _predict(self, x: np.ndarray, node: Node) -> int:
        """
        Given a single data point x and a decision tree node, return the predicted integer label.
        """
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)

if __name__ == "__main__":
    # Test the DecisionTree class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 0

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    # Expect the training accuracy to be 1.0 when max_depth=None
    rf = DecisionTree(max_depth=None, criterion="entropy")
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")