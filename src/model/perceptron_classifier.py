import json
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, confusion_matrix


class PerceptronClassifier:

    def __init__(self, threshold=0.5, learning_rate=0.1, max_iter=100):
        """
        Simple Perceptron classifier. The threshold of the
        output decision can be set based on the class values
        in Y, and the learning rate can be adjusted. The
        weights will update row-by-row in order of the data
        X.

        :param threshold:       Threshold activation for the Perceptron.
        :param learning_rate:   Influence given to new weights over
                                previous weights.
        :param max_iter:        Maximum iterations for training.
        """
        # The weights provided by the train() function
        self.weights = None

        # The length of the weights array (dimensionality of data)
        self.dimensionality = None

        self.features = None

        self.threshold = threshold
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def _check_inputs(self, x, y):
        if x is None or len(x) == 0:
            print("[ ERR ] No data provided: Either empty set or None was provided.")
            return None, None

        if len(x) != len(y):
            print("[ ERR ] X and Y are not the same length")
            return None, None

        if isinstance(x, pd.DataFrame):
            if not self.features:
                self.features = x.columns.tolist()

            x = x.values

        if isinstance(y, pd.DataFrame):
            y = y.values

        return x, y

    def _is_training_error(self, x, y, w):
        errors = 0
        pred = [int(np.dot(x_i, w) >= self.threshold) for x_i in x]
        for i in range(len(pred)):
            if pred[i] != y[i]:
                errors += 1

        print("\tTraining error:", errors)
        return errors > 0

    def fit(self, x, y):
        """
        Fits the classifier on the X, Y training data set.
        If X and Y are not linearly separable, the weight
        output will not be meaningful.
        Sets internal property, `self.weights`.

        :param x: Training data. Can be either 2D array or DataFrame
                  If a DataFrame is provided `self.features` will be
                  populated with the columns in X.
        :param y: Labels for training data X.
        :return: `None` Sets internal property `self.weights`.
        """

        # Validate inputs
        x, y = self._check_inputs(x, y)
        if x is None or y is None:
            return

        print("[ INF ] Fitting Perceptron with", len(x[0]), "dimensionality")

        # The weights array, in dimensionality of data
        # The initial values will be zero
        w = len(x[0]) * [0]

        for i in range(self.max_iter):
            print("\tIter:", i)
            for j in range(len(x)):
                # X and Y for the jth sample
                x_j = x[j]
                d_j = y[j]

                # The model output at time t for sample j
                y_jt = int(np.dot(w, x_j) >= self.threshold)

                if d_j - y_jt == 0:
                    continue

                # The new weight vector for t + 1
                w = [w[i] + (0.01 * (d_j - y_jt) * x_j[i]) for i in range(len(w))]

            if not self._is_training_error(x, y, w):
                break

        # Assign the final weights to this object instance
        self.weights = w
        self.dimensionality = len(x[0])

    def predict(self, x, weights=None):
        """
        Make predictions Y on provided X.

        :param x:           Unlabelled input data. Can be
                            either DataFrame or 2D array.
        :param weights:     Optional weights vector.
                            `self.weights` used by default.
        :return:            Labels vector Y for X.
        """
        y = []

        if weights is None:
            weights = self.weights

        if isinstance(x, pd.DataFrame):
            x = x.values

        for x_i in x:
            y.append(int(np.dot(x_i, weights) >= self.threshold))

        return y

    def validate(self, val_x, val_y):
        """
        Prints the F1 and Confusion Matrix for
        classifier.

        :param val_x: Validation data X.
        :param val_y: Validation data Y.
        :return: `None`
        """
        if self.weights is None:
            print("[ ERR ] Could not validate model: Not fitted")
            return

        val_x, val_y = self._check_inputs(val_x, val_y)
        if val_x is None or val_y is None:
            return

        y_pred = self.predict(val_x)
        print("F1 Score:", f1_score(val_y, y_pred))
        print(confusion_matrix(val_y, y_pred))

    def save_weights(self, filename='perceptron_weights.json'):
        if self.weights is None:
            print("[ ERR ] Cannot save weights: Not fitted")
            return

        with open(filename, 'w') as fp:
            json.dump(self.weights, fp, indent=4)

    def save_features(self, filename='feature_weights.csv'):
        if self.weights is None or self.features is None:
            print("[ ERR ] Cannot save features: Not fitted or no features")
            return

        df = pd.DataFrame()
        df['feature'] = self.features
        df['weight'] = self.weights
        df.to_csv(filename, index=False)

    def load_weights(self, filename='doc/perceptron_weights.json'):
        with open(filename, 'r') as fp:
            w = json.load(fp)
            self.weights = w


class AveragedPerceptronClassifier(PerceptronClassifier):
    def __init__(self, threshold=0.5, learning_rate=0.1, max_iter=100):
        PerceptronClassifier.__init__(self,
                                      threshold=threshold,
                                      learning_rate=learning_rate,
                                      max_iter=max_iter)

    def fit(self, x, y):
        """
        Fits the classifier on the X, Y training data set.
        If X and Y are not linearly separable, the weight
        output will not be meaningful.
        Sets internal property, `self.weights`.
        Final weight vector is the average of all considered
        weight vectors.

        :param x: Training data. Can be either 2D array or DataFrame
                  If a DataFrame is provided `self.features` will be
                  populated with the columns in X.
        :param y: Labels for training data X.
        :return: `None` Sets internal property `self.weights`.
        """

        # Validate inputs
        x, y = self._check_inputs(x, y)
        if x is None or y is None:
            return

        print("[ INF ] Fitting Perceptron with", len(x[0]), "dimensionality")

        # The weights array, in dimensionality of data
        # The initial values will be zero
        w = len(x[0]) * [0]

        # Average accumulator
        acc = w
        count = 0

        for i in range(self.max_iter):
            print("\tIter:", i)
            for j in range(len(x)):
                # X and Y for the jth sample
                x_j = x[j]
                d_j = y[j]

                # The model output at time t for sample j
                y_jt = int(np.dot(w, x_j) >= self.threshold)

                if d_j - y_jt == 0:
                    continue

                # The new weight vector for t + 1
                w = [w[i] + (0.01 * (d_j - y_jt) * x_j[i]) for i in range(len(w))]
                acc = [acc[i] + w[i] for i in range(len(acc))]
                count += 1

            if not self._is_training_error(x, y, w):
                break

        # Assign the final weights to this object instance
        self.weights = [x / count for x in acc]
        self.dimensionality = len(x[0])
