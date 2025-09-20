import numpy as np
from decision_tree import DecisionTree, most_common


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "entropy",
        max_features: None | str = "sqrt",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth, "criterion": self.criterion, "max_features": self.max_features}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):

        n_samples, n_features = X.shape
        self.trees = []

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            tree = DecisionTree(max_depth=self.max_depth, criterion=self.criterion)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)


    def predict(self, X: np.ndarray) -> np.ndarray:
        all_prediction = []
        for tree in self.trees:
            predictions = tree.predict(X)
            all_prediction.append(predictions)
        all_prediction = np.array(all_prediction)

        return np.array([most_common(pred) for pred in all_prediction.T])

if __name__ == "__main__":
    # Test the RandomForest class on a synthetic dataset
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

    rf = RandomForest(
        n_estimators=20, max_depth=5, criterion="gini", max_features="sqrt"
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
