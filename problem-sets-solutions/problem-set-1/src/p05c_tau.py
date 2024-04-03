import matplotlib.pyplot as plt
import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    # *** START CODE HERE ***
    # Fit a LWR model
    target_tau = None
    mse_value = None
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau=tau)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_eval)
        # Get MSE value on the training set
        mse_train = np.mean((y_pred - y_eval)**2)
        if mse_value is None or mse_train < mse_value:
            mse_value = mse_train
            target_tau = tau
    print(f'MSE with tau = {target_tau} : {mse_value}')
    model = LocallyWeightedLinearRegression(tau=target_tau)
    model.fit(x_train, y_train)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    y_pred = model.predict(x_test)
    plt.figure()
    plt.plot(x_train, y_train, 'bx', linewidth=2)
    plt.plot(x_test, y_pred, 'ro', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f'output/p05c_lwr_tau{target_tau}_test.png')
    np.savetxt(pred_path, y_pred)
    # *** END CODE HERE ***
