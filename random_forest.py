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

    def features_subset(self, n_features: int) -> np.ndarray:
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            max_features = int(np.log2(n_features))
        else:
            max_features = n_features

        feature_indices = np.random.choice(
            n_features, size=max_features, replace=False
        )
        return feature_indices

    def fit(self, X: np.ndarray, y: np.ndarray):

        n_samples, n_features = X.shape
        
        self.trees = []

        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            feature_indices = self.features_subset(n_features)
            X_sample = X_sample[:, feature_indices]

            tree = DecisionTree(max_depth=self.max_depth, criterion=self.criterion)
            tree.fit(X_sample, y_sample)
            self.trees.append((tree, feature_indices))


    def predict(self, X: np.ndarray) -> np.ndarray:
        
        all_prediction = []
        for i in range(self.n_estimators):
            tree, feature_indices = self.trees[i]
            X_subset = X[:, feature_indices]

            predictions = DecisionTree.predict_tree(tree, X_subset)
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
        n_estimators=20, max_depth=5, criterion="gini", max_features="one"
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
