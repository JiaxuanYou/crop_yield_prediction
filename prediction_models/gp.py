import numpy as np

from scipy.spatial.distance import cdist
from scipy.linalg           import solve

class GaussianProcess:
    """
    The crop yield Gaussian process
    """
    def __init__(self, sigma=1, r_loct=0.5, r_year=1.5, sigma_e=0.32, sigma_b=0.01):
        self.sigma = sigma
        self.r_loct = r_loct
        self.r_year = r_year
        self.sigma_e = sigma_e
        self.sigma_b = sigma_b

    @staticmethod
    def _normalize(x):
        x_mean  = np.mean(x, axis=0, keepdims=True)
        x_scale =  np.ptp(x, axis=0, keepdims=True)

        return (x - x_mean) / x_scale

    def run(self, feat_train, feat_tests, loct_train, loct_tests, year_train, year_tests,
            train_yield, model_weights, model_bias):
        """
        Gaussian process for linear regression (from Rasmussen 2006):

            We have the training dataset (X_1, y_1), and would like to predict
            values for y_2 from observed points X_2. Assume that y_1, y_2 are
            jointly Gaussian and come from the same distribution. Our model is

            Jointly Gaussian
            1. [y_1, y_2].T ~ N([μ_1, μ_2], [[Σ_11, Σ_12], [Σ_21, Σ_22]])

            Linear Model for Regression with Normally Distributed Residuals
            2.  y_i = f(X_i) where f(X) = H(X).β + ε, β ~ N(b, B), ε ~ N(0, K).

            The predictions given by our model is

            Conditional mean of posterior Gaussian Pr(μ_2 | μ_1)
            1.  y_2 = H(X_2).β + K(X_1, X_2)inv(K'(X_1, X_1))(y_1 - H(X_1).β)

            Tikhonov regularization for β ~ N(b, B)
            2.  β   = inv(inv(B) + H(X_1)inv(K'(X_1, X_1))H(X_1).T) *
                 (H(X_1)inv(K'(X_1, X_1)).y_1 + inv(B).b)

        Now in python:
        """

        n_train = feat_train.shape[0]
        n_tests = feat_tests.shape[0]

        normed_locts = self._normalize(np.concatenate((loct_train, loct_tests), axis=0))
        normed_years = self._normalize(np.concatenate((year_train, year_tests), axis=0))

        loct_train = normed_locts[:len(loct_train) ]
        loct_tests = normed_locts[ len(loct_train):]

        year_train = np.expand_dims(normed_years[:len(year_train) ], axis=1)
        year_tests = np.expand_dims(normed_years[ len(year_train):], axis=1)

        # Calculate squared euclidean distances between training points.
        ker_train  = -cdist(loct_train, loct_train, metric='sqeuclidean') / (self.r_loct ** 2)
        ker_train += -cdist(year_train, year_train, metric='sqeuclidean') / (self.r_year ** 2)

        # Obtain Gaussian kernel (RBF) by exponentiating and multiplying.
        ker_train  = (self.sigma   ** 2) *    np.exp(ker_train)

        # Add noise along diagonal since observations are noisy.
        ker_train += (self.sigma_e ** 2) * np.identity(n_train)

        # Calculate feature / basis matrix H with bias term appended.
        H_train = np.concatenate((feat_train, np.ones((n_train, 1))), axis=1)
        H_tests = np.concatenate((feat_tests, np.ones((n_tests, 1))), axis=1)

        y_train = np.expand_dims(train_yield, axis=1)

        # Calculate equations involving K inverse, a PD matrix due to RBF kernel.
        inv_K_H_train = solve(ker_train, H_train, assume_a="pos")
        inv_K_y_train = solve(ker_train, y_train, assume_a="pos")

        del ker_train


        # H_train has shape (n_train, n_feats).
        _, n_feats = H_train.shape

        # Calculate inverse of B, which is a diagonal matrix.
        B_inv = np.identity(n_feats) / self.sigma_b

        # "We choose b as the weight vector of the last layer of our deep models"
        b = np.concatenate((model_weights.transpose(1, 0), np.expand_dims(model_bias, 1)))

        # Solve for optimal coefficients (β) in equation 2.41 in Rasmussen (2006).
        beta = solve(

            # L-hand side
            B_inv + H_train.T.dot(inv_K_H_train),

            # R-hand side
            H_train.T.dot(inv_K_y_train) + B_inv.dot(b),

            # L-hand side is Hermitian
            assume_a = "pos"
        )

        # Calculate squared euclidean distances between testsing points and training points.
        ker_tests  = -cdist(loct_tests, loct_train, metric='sqeuclidean') / (self.r_loct ** 2)
        ker_tests += -cdist(year_tests, year_train, metric='sqeuclidean') / (self.r_year ** 2)

        # Obtain Gaussian kernel (RBF) by exponentiating and multiplying.
        ker_tests  = (self.sigma   ** 2) * np.exp(ker_tests)

        # We take the mean of g(X*) as our prediction, also from equation 2.41
        pred = H_tests.dot(beta) + ker_tests.dot(inv_K_y_train - inv_K_H_train.dot(beta))
        del ker_tests

        return pred
