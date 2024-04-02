import p01b_logreg
import p01e_gda
import numpy as np
import util
import matplotlib.pyplot as plt

def main(train_path, correction=1.0):
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    gda_model = p01e_gda.GDA()
    gda_model.fit(x_train, y_train)
    new_x = np.zeros((x_train.shape[0], x_train.shape[1] + 1), dtype=x_train.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x_train
    x_train = new_x
    log_reg_model = p01b_logreg.LogisticRegression()
    log_reg_model.fit(x_train, y_train)
    plt.figure()
    plt.plot(x_train[y_train == 1, -2], x_train[y_train == 1, -1], 'bx', label='Positive')
    plt.plot(x_train[y_train == 0, -2], x_train[y_train == 0, -1], 'go', label='Negative')
    margin1 = (max(x_train[:, -2]) - min(x_train[:, -2])) * 0.2
    margin2 = (max(x_train[:, -1]) - min(x_train[:, -1])) * 0.2
    x1 = np.arange(min(x_train[:, -2]) - margin1, max(x_train[:, -2]) + margin1, 0.01)
    x2 = -(log_reg_model.theta[0] / log_reg_model.theta[2] * correction + log_reg_model.theta[1] / log_reg_model.theta[2] * x1)
    plt.plot(x1, x2, c='red', label='Logistic Regression', linewidth=2)
    x2 = -(gda_model.theta[0] / gda_model.theta[2] * correction + gda_model.theta[1] / gda_model.theta[2] * x1)
    plt.plot(x1, x2, c='yellow', label='GDA', linewidth=2)
    plt.xlim(x_train[:, -2].min() - margin1, x_train[:, -2].max() + margin1)
    plt.ylim(x_train[:, -1].min() - margin2, x_train[:, -1].max() + margin2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig('output/p01g_plot.png')