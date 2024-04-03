import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)
    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    model = LogisticRegression()
    model.fit(x_train, t_train)
    t_pred = model.predict(x_test)
    util.plot(x_test, t_test, model.theta, save_path='output/p02c_posonly.png')
    # Make sure to save outputs to pred_path_c
    np.savetxt(pred_path_c, t_pred > 0.5, fmt='%d')
    # Part (d): Train on y-labels and test on true labels
    model = LogisticRegression()
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    util.plot(x_test, y_test, model.theta, save_path='output/p02d_posonly.png')
    # Make sure to save outputs to pred_path_d
    np.savetxt(pred_path_d, y_pred > 0.5, fmt='%d')
    # Part (e): Apply correction factor using validation set and test on true labels
    model = LogisticRegression()
    model.fit(x_train, y_train)
    alpha = np.mean(model.predict(x_valid[y_valid == 1]))
    correction = 1 + np.log(2 / alpha - 1) / model.theta[0]
    util.plot(x_test, t_test, model.theta, save_path='output/p02e_posonly.png', correction=correction)
    t_pred = model.predict(x_test) / alpha
    # Plot and use np.savetxt to save outputs to pred_path_e
    np.savetxt(pred_path_e, t_pred > 0.5, fmt='%d')
    # *** END CODER HERE
