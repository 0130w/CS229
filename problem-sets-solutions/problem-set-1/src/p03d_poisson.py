import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    model = PoissonRegression(step_size=lr)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred, fmt="%d")
    util.plt.figure()
    util.plt.plot(y_eval, y_pred, 'bx')
    util.plt.xlabel('true labels')
    util.plt.ylabel('predict labels')
    util.plt.savefig('output/p03d.png')


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m, n = x.shape
        self.theta = np.zeros(n)
        while True:
            old_theta = np.copy(self.theta)
            h_x = np.exp(x @ self.theta)
            self.theta = self.theta + self.step_size * (x.T @ (y - h_x)) / m
            if np.linalg.norm(self.theta - old_theta, ord=1) < self.eps:
                break

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        m, n = x.shape
        return np.exp(x @ self.theta.T) 