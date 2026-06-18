"""
Decision Tree Classifier

The Node dataclass and DecisionTreeClassifier are provided. Implement
fit, predict, and _grow.

More on the CART algorithm this loosely follows:

    https://en.wikipedia.org/wiki/Decision_tree_learning
"""

from collections.abc import Callable
from dataclasses import dataclass

from .impurity import entropy, gini
from .split import best_split


@dataclass
class Node:
    """
    A single node of the decision tree.

    Internal nodes store the split used and references to their children.
    Leaves store only a prediction. Every node stores n_samples, the
    number of training samples that reached it during fit.

    Numeric split  : feature <= threshold  (threshold is set, category_value
                     is None)
    Categorical split: feature == category_value  (category_value is set,
                     threshold is None)
    """

    prediction: int | None = None
    n_samples: int = 0
    feature: int | None = None
    threshold: float | None = None
    category_value: str | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTreeClassifier:
    """
    A classification tree built by greedy, top-down information-gain
    maximization (CART-style).

    Hyperparameters:
        max_depth         -- maximum depth of the tree; None means grow
                             until another stopping condition fires
        min_samples_split -- a node with fewer than this many samples
                             becomes a leaf without searching for a split
        criterion         -- "gini" or "entropy"
        feature_types     -- list of "numeric" or "categorical" strings,
                             one per feature; None treats all as numeric
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
        feature_types: list[str] | None = None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(
                f"criterion must be 'gini' or 'entropy', got {criterion!r}"
            )
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be >= 1 or None")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.feature_types = feature_types
        self._impurity: Callable[[list[int]], float] = (
            gini if criterion == "gini" else entropy
        )
        self.root: Node | None = None

    def fit(self, X: list[list], y: list[int]) -> "DecisionTreeClassifier":
        """
        Builds the tree by recursively splitting starting from the root.

        Raises ValueError if X or y is empty, or if their lengths differ.

        Arguments:
            X -- feature vectors to train on
            y -- class labels, one per row in X

        Returns:
            DecisionTreeClassifier -- self, to allow method chaining
        """
        # ------ WRITE YOUR CODE HERE ------
        if not X or not y:
            raise ValueError("X and y cannot be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        # print("grow send")
        self.root = self._grow(X, y, depth=0)
        return self

    def predict(
        self,
        X: list[tuple],
        stop_depth: int | None = None,
        stop_below: int | None = None,
    ) -> list[int]:
        """
        Predicts a class label for each sample in X.

        Raises RuntimeError if the tree has not been fitted yet.

        Arguments:
            X          -- feature vectors to classify
            stop_depth -- if given, stops traversal at this depth and
                          uses the prediction stored at that node.
                          Useful for simulating a tree trained with a
                          smaller max_depth without retraining.
            stop_below -- if given, stops traversal at any node whose
                          n_samples is less than this value. Useful for
                          simulating a tree trained with a larger
                          min_samples_split without retraining.

        Returns:
            list[int] -- predicted class label for each sample
        """
        # ------ WRITE YOUR CODE HERE ------
        if self.root is None:
            raise RuntimeError("Tree has not been fitted")

        preds = []
        for x in X:
            _, pred = self._predict_one(x, self.root, stop_depth, stop_below)
            preds.append(pred)
        return preds

    def predict_with_depth(
        self,
        X: list[tuple],
        stop_depth: int | None = None,
        stop_below: int | None = None,
    ) -> list[tuple[int, int]]:
        """
        Predicts a class label and traversal depth for each sample in X.

        Returns a list of (prediction, depth) pairs, where depth is the
        number of edges traversed from the root to the decision node.

        Accepts the same stop_depth and stop_below arguments as predict.
        """
        # ------ WRITE YOUR CODE HERE ------
        if self.root is None:
            raise RuntimeError("Tree has not been fitted")

        return [
            self._predict_one(x, self.root, stop_depth, stop_below) for x in X
        ]

    @staticmethod
    def _predict_one(
        x: tuple,
        node: Node,
        stop_depth: int | None = None,
        stop_below: int | None = None,
    ) -> tuple[int, int]:
        """
        Walks one sample down the tree and returns its prediction and
        the depth of the node where traversal stopped.

        Arguments:
            x          -- a single feature vector
            node       -- the node to start traversal from
            stop_depth -- stop at this depth if given; see predict
            stop_below -- stop at nodes with fewer samples if given;
                          see predict

        Returns:
            tuple[int, int] -- (predicted class label, traversal depth)
        """

        # ------ WRITE YOUR CODE HERE ------
        depth = 0
        current = node

        while True:
            """if current.prediction is None:
                print("None")"""
            # stopping conditions
            if current.is_leaf:
                return current.prediction, depth

            if stop_depth is not None and depth >= stop_depth:
                return current.prediction, depth

            if stop_below is not None and current.n_samples < stop_below:
                return current.prediction, depth

            # numeric split
            if current.threshold is not None:
                if x[current.feature] <= current.threshold:
                    current = current.left
                else:
                    current = current.right

            # categorical split
            else:
                if x[current.feature] == current.category_value:
                    current = current.left
                else:
                    current = current.right

            depth += 1

    @staticmethod
    def _majority(self, y):
        # sin Counter
        # pero algo como:
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        print("MAJORITY CALLED WITH y =", y)
        return max(counts, key=counts.get)

    def _grow(self, X: list[list], y: list[int], depth: int) -> Node:
        """
        Recursively builds a subtree and returns its root Node.

        Returns a leaf node (prediction = majority class) if any stopping
        condition holds: the node is pure, fewer than min_samples_split
        samples remain, max_depth is reached, or no split improves impurity.
        Otherwise splits on the best feature and recurses.

        For numeric features the split is feature <= threshold; for
        categorical features it is feature == category_value.

        Arguments:
            X     -- feature vectors at this node
            y     -- labels at this node
            depth -- depth of this node (root is 0)

        Returns:
            Node -- root of the subtree grown from (X, y)
        """

        # ------ WRITE YOUR CODE HERE ------
        node = Node(n_samples=len(y))
        # print("in grow")
        """if node.prediction is None:
            print("None")"""
        # stopping: pure node
        if len(set(y)) == 1:
            node.prediction = y[0]
            return node

        # stopping: too few samples
        if len(y) < self.min_samples_split:
            node.prediction = self._majority(y)
            return node

        # stopping: max depth reached
        if self.max_depth is not None and depth >= self.max_depth:
            node.prediction = self._majority(y)
            return node

        # find best split
        split = best_split(X, y, self._impurity, self.feature_types)
        if split is None:
            node.prediction = self._majority(y)
            return node

        feature, value, gain = split
        node.feature = feature

        # numeric split
        if isinstance(value, float):
            node.threshold = value
            X_left, y_left = [], []
            X_right, y_right = [], []

            for xi, yi in zip(X, y):
                if xi[feature] <= value:
                    X_left.append(xi)
                    y_left.append(yi)
                else:
                    X_right.append(xi)
                    y_right.append(yi)

        # categorical split
        else:
            node.category_value = value
            X_left, y_left = [], []
            X_right, y_right = [], []

            for xi, yi in zip(X, y):
                if xi[feature] == value:
                    X_left.append(xi)
                    y_left.append(yi)
                else:
                    X_right.append(xi)
                    y_right.append(yi)

        print("SPLIT at depth", depth)
        print("  y_left =", y_left)
        print("  y_right =", y_right)

        # recursive children
        node.left = self._grow(X_left, y_left, depth + 1)
        node.right = self._grow(X_right, y_right, depth + 1)

        return node
