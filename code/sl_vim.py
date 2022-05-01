
import numpy as np
import code.sl_task
from code.learners import initialize_sl, square_error_loss, binomial_loglik_loss


def importance(X, Y, stack, meta, family, K1 = 5, K2 = 10):
	# define parameters
	folds = list(KFold(n_splits = K1).split(Y))
	n = X.shape[0]
	p = X.shape[1]
	sl_list = []
	if (family == 'Gaussian'):
		loss_f = square_error_loss
	elif (family == 'Binomial'):
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
    cv_risks = np.zeros(p)

    for j in range(p):
        X_c = np.copy(X)
        X_c[:,j] = np.random.permutation(X[:,j])
        cv_preds = np.zeros(n)

        for i in range(K1):
            idx_v = folds[k][1]
            X_v = X[idx_v]
            Y_v = Y[idx_v]

            current_sl = sl_list[i]
            preds = current_sl.predict(X_v)
            cv_preds[idx_v] = preds

        cv_risks[j] = loss_f(cv_preds, Y.ravel())
        
    # sort feature index by cv_risks
    out = np.zeros(p, 2)
    out[:, 1] = np.arange(p)
    out[:, 2] = cv_risks
    out = out[[:, 2].argsort()]
    return out





















