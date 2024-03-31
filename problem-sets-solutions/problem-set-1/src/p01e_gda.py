import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    # *** START CODE HERE ***
    model = GDA()
    model.fit(x_train, y_train)
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self):
        self.phi = 0
        self.mu = None
        self.sigma = None

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m, n = x.shape
        self.mu = np.zeros((2, n))
        # *** START CODE HERE ***
        self.phi = sum(1 for i in y if int(i) == 1) / m
        self.mu[0] = sum(x[i] for i in range(0, m) if int(y[i]) == 0) / sum(1 for i in y if int(i) == 0)
        self.mu[1] = sum(x[i] for i in range(0, m) if int(y[i]) == 1) / sum(1 for i in y if int(i) == 1)
        self.sigma = sum(((x[i] - self.mu[int(y[i])]) @ (x[i] - self.mu[int(y[i])])).T for i in range(0, m)) / m
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        
        # *** END CODE HERE
