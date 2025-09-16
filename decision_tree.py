import numpy as np
from typing import Self

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

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
    proportions = counts / counts.sum()
    return proportions
    
print( "count:", count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])))


def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """

    1 - np.sum(count(y)**2)

def entropy(y: np.ndarray) -> float:
    """
    Return the entropy of a given NumPy array y.
    """
    return -np.sum(count(y)*np.log2(count(y)))
    return -np.sum(count(y)*np.log2(count(y)))

print("entropy:", entropy(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])))

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
    value, counts = np.unique(y, return_counts=True)
    return value[np.argmax(counts)]

print("most common:", most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 3])))

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
        # Return True iff the node is a leaf node
        return self.value is not None


class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        criterion: str = "entropy",
    ) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):

        ## finn beste split
        ## lag to nye lister med splitten
        ## kall fit rekursivt på hver av listene og lag en node
        """
        This functions learns a decision tree given (continuous) features X and (integer) labels y.
        """

        #if all datapoints have the same label, return a leaf node with that label.
        if len(np.unique(y)) == 1:
            return Node(value=self.count(y))
        
        #If all data points have identical feature values, return a leaf node with the most common label.
        if np.all(X == X[0]):
            return Node(value=self.count(y))
        

        #split on mean or median on a datapoint with most information gain into to arraylists

        
    

   

        #Otherwise, choose a feature that maximizes the information gain, split the data based on
        #the value of the feature, and add a branch for each subset of data. For each branch, call the
        #algorithm recursively for the data points belonging to the particular branch.
    
    
    
    


    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        raise NotImplementedError(
            "Implement this function"
        )  # Remove this line when you implement the function


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




print("dette er Thone, din hacker")

clf = DecisionTreeClassifier().fit(X, y)

tree_text = tree.export_text(clf, feature_names=["f1", "f2", "f3"])
print(tree_text)