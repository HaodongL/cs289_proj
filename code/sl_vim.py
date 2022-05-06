
import numpy as np
from sklearn.model_selection import KFold
from sl_task import sl_task
from learners import initialize_sl, square_error_loss, binomial_loglik_loss


def importance(X, Y, stack, meta, family, K1 = 5, K2 = 10):
    # define parameters
    folds = list(KFold(n_splits = K1).split(Y))
    n = X.shape[0]
    p = X.shape[1]
    sl_list = []
    if family == 'Gaussian':
        loss_f = square_error_loss
    elif family == 'Binomial':
        loss_f = binomial_loglik_loss

    # fit a sl on each training set
    for k in range(K1):
        idx_t = folds[k][0]
        idx_v = folds[k][1]
        X_t = X[idx_t]
        Y_t = Y[idx_t]
        X_v = X[idx_v]
        Y_v = Y[idx_v]

        current_task = sl_task(X_t, Y_t, family, K2)
        current_sl = initialize_sl(current_task, stack, meta)
        current_sl.train()
        sl_list.append(current_sl)

    # permute each col, predit on each v set
    cv_risks = np.zeros(p + 1)

    for j in range(p + 1):
        X_c = np.copy(X)
        if j != p:
            X_c[:,j] = np.random.permutation(X[:,j])
        cv_preds = np.zeros(n)

        for i in range(K1):
            idx_v = folds[i][1]
            X_v = X_c[idx_v]
            Y_v = Y[idx_v]

            current_sl = sl_list[i]
            preds = current_sl.predict(X_v)
            cv_preds[idx_v] = preds

        cv_risks[j] = loss_f(cv_preds, Y.ravel())
        
    # sort feature index by risk differences
    cv_risk_true = cv_risks[p + 1]
    cv_risk_perm = cv_risks[0:(p + 1)]
    diff_risk = np.absolute(cv_risk_perm - cv_risk_true)

    out = np.zeros((p, 2))
    out[:, 0] = np.arange(p)
    out[:, 1] = diff_risk
    out = out[out[:, 1].argsort()]
    return out

















