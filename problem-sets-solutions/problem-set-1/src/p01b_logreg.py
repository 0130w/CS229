import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_test, y_test = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)

    util.plot(x_train, y_train, model.theta, 'output/p01b_train_ds2.png')
    util.plot(x_test, y_test, model.theta, 'output/p01b_eval_ds2.png')
    y_pred_labels = (model.predict(x_test) > 0.5).astype(int)
    util.plot(x_test, y_pred_labels, model.theta, 'output/p01b_pred_ds2.png')
    np.savetxt(pred_path, y_pred_labels, fmt="%d")
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n) # 1 * n vector
        while True:

            theta_old = np.copy(self.theta)

            h_x = 1 / (1 + np.exp(- x @ self.theta.T))
            J_theta_grad = x.T @ (h_x - y) / m  # n * 1 vector
            hessian = (x.T * h_x * (1 - h_x)) @ x / m

            self.theta = self.theta - np.linalg.inv(hessian) @ J_theta_grad

            if np.linalg.norm(self.theta - theta_old, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(- x @ self.theta))
        # *** END CODE HERE ***
