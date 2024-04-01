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
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    model = GDA()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_eval)
    y_pred_label = (y_pred > 0.5).astype(int)
    np.savetxt(pred_path, y_pred_label, fmt="%d")
    util.plot(x_eval, y_eval, model.theta, 'output/p01e_eval_ds2.png')
    util.plot(x_train, y_train, model.theta, 'output/p01e_train_ds2.png')
    util.plot(x_eval, y_pred_label, model.theta, 'output/p01e_pred_ds2.png')


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m, n = x.shape
        self.theta = np.zeros(n+1)

        y_1 = sum(y == 1)
        phi = y_1 / m
        mu_0 = np.sum(x[y == 0], axis=0) / (m - y_1)
        mu_1 = np.sum(x[y == 1], axis=0) / y_1
        sigma = ((x[y == 0] - mu_0).T @ (x[y == 0] - mu_0) + (x[y == 1] - mu_1).T @ (x[y == 1] - mu_1)) / m

        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = ((mu_0 + mu_1) @ sigma_inv @ (mu_0 - mu_1)) / 2 + np.log((1-phi) / phi)

        self.theta[1:] = sigma_inv @ (mu_1 - mu_0)

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return 1 / (1 + np.exp(-(x @ self.theta)))