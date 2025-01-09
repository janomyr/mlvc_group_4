import numpy as np
import scipy
import scipy.optimize as opt
from scipy.linalg import cho_solve, cholesky, solve_triangular
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared

###################### TO-DO #######################
########### Gaussian Process (30 Points) ###########
# NOTE: You may use the imported functions from scipy,
#       but you can also use another library if you want,
#       or implement the functions yourself.
# 1. Implement sampling (unconditioned): (2 Points)
#   --> See: sample_points(observation=None)
#   --> You may ignore the conditioned parameter for now
# 2. Update sampling (conditioned): (3 Points)
#   --> See: sample_points(observation=x) # x could be any number (for example 1.2)
# 3. Implement prediction based only on prior (8 Points)
#   --> See: GaussianProcess.predict()
# 4. Implement prediction based on posterior (8 Points)
#   --> See: GaussianProcess.predict()
#   --> And: GaussianProcess.fit(meta_parameter_search=False)
#   --> You may ignore the if neg_log_likelihood for now
# 5. Implement negative log likelihood type 2 (9 Points)
#   --> See: GaussianProcess.negative_log_likelihood_type_2()
#   --> And: GaussianProcess.fit(meta_parameter_search=True)


def sample_points(mean, cov, n, observation=None):
    """
    The function generates n samples from a multivariate normal distribution
    specified by the given mean and covariance matrix. If the observation
    parameter is set to a number, it generates samples
    conditioned on the observed number of the first variable.

    Args:
        mean (numpy.ndarray): Mean of the distribution
        cov (numpy.ndarray): Covariance matrix of the distribution
        n (int): Number of points to sample
        observation (int): the number to condition the second variable on

    Returns:
        numpy.ndarray: Sampled points
    """

    # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
    if observation:
        # conditioned on observation+
        mu_1 = mean[0]
        mu_2 = mean[1]

        sigma_11 = cov[0,0]
        sigma_12 = cov[0,1]
        sigma_21 = cov[1,0]
        sigma_22_inv = 1/cov[1,1]
        
        mean_c = [mu_1 + sigma_12 * sigma_22_inv * (observation - mu_2)]
        cov_c = sigma_11 - sigma_12 * sigma_22_inv * sigma_21

        x1 = np.ones((n))*observation
        x2 = np.random.normal(mean_c, np.sqrt(cov_c), n)

        sampled_points = np.stack((x1,x2),axis=0)
    else:
        # unconditioned
        sampled_points = np.random.multivariate_normal(mean, cov, n).T
    # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    return sampled_points


class MultivariateNormal:
    def __init__(self, mean, cov, seed=42):
        self.mean = mean
        self.covariance = cov
        self.seed = seed

        self.distr = scipy.stats.multivariate_normal(mean, cov, seed=seed)

    def pdf(self, X, Y):
        return self.distr.pdf(np.dstack((X, Y)))


class GaussianProcess:
    def __init__(self, length_scale=1.0, noise=1e-10, kernel=None, periodicity=1.0):
        # Hyperparameters
        self.length_scale = length_scale  # Hyperparameter for length scale
        self.periodicity = periodicity  # Hyperparameter for periodicity
        self.noise = noise  # Hyperparameter for noise

        # Training Data and Related Variables
        self.X_train = None  # Placeholder for training data (input features)
        self.y_train = None  # Corresponding labels for training data
        self.n_targets = None  # Number of targets or outputs

        # Kernel-related Variables
        self.K = None  # Kernel matrix
        self.alpha_ = None  # Alpha variable related to the kernel
        self.L_ = None  # Lower triangular matrix
        self.kernel = kernel  # Kernel function
        self.kernel_type = kernel  # Variable storing the type of kernel

        assert self.kernel_type in [
            "RBF",
            "RBF+Sine",
            "Sine+RBF",
            "Sine",
        ], "Invalid kernel type"

    def negative_log_likelihood_type_2(self, params):
        """
        Negative log likelihood function.

        Args:
            params (numpy.ndarray): The parameters to optimize

        Returns:
            float: The negative log likelihood
        """

        length_scale, noise, periodicity = params

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        pass
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def fit(self, X_train, y_train, meta_parameter_search=False):
        """
        Fit the Gaussian Process model to the training data.

        Parameters:
        - X_train: Input features for training (numpy array)
        - y_train: Target values for training (numpy array)
        """
        self.X_train = X_train
        self.y_train = y_train

        # Update hyperparameters
        if meta_parameter_search:
            print(
                f"Parameters before: Lengthscale: {self.length_scale}, Noise: {self.noise}, Periodicity: {self.periodicity}"
            )

            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            self.negative_log_likelihood_type_2
            self.length_scale, self.noise, self.periodicity = None, None, None
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

            print(
                f"Parameters after: Lengthscale: {self.length_scale}, Noise: {self.noise}, Periodicity: {self.periodicity}"
            )

        if self.kernel_type == "RBF+Sine" or self.kernel_type == "Sine+RBF":
            self.kernel = RBF(length_scale=self.length_scale) + ExpSineSquared(
                length_scale=self.length_scale, periodicity=self.periodicity
            )
        elif self.kernel_type == "RBF":
            self.kernel = RBF(length_scale=self.length_scale)
        elif self.kernel_type == "Sine":
            self.kernel = ExpSineSquared(
                length_scale=self.length_scale, periodicity=self.periodicity
            )

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        self.K = self.kernel(X_train) + self.noise * np.eye(len(X_train))
        self.L_ = cholesky(self.K, lower=True)
        self.alpha_ = cho_solve((self.L_, True),y_train)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def predict(self, X_test):
        """
        Make predictions on new data.

        Parameters:
        - X_test: Input features for prediction (numpy array)

        Returns:
        - mean: Predicted mean for each input point
        - std: Predicted standard deviation for each input point
        """

        if (
            not hasattr(self, "X_train") or self.X_train is None
        ):  # Unfitted;predict based on GP prior
            # If the GP is called unfitted, we need to set the kernel here
            if self.kernel_type == "RBF+Sine" or self.kernel_type == "Sine+RBF":
                self.kernel = RBF(length_scale=self.length_scale) + ExpSineSquared(
                    length_scale=self.length_scale, periodicity=self.periodicity
                )
            elif self.kernel_type == "RBF":
                self.kernel = RBF(length_scale=self.length_scale)
            elif self.kernel_type == "Sine":
                self.kernel = ExpSineSquared(
                    length_scale=self.length_scale, periodicity=self.periodicity
                )

            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            K_test_test = self.kernel(X_test) # Covariance matrix of test data
            mean_pred_distribution, std_pred_distribution, conv_pred_distribution = (
                np.zeros(X_test.shape[0]), # mean has to be 0
                np.sqrt(np.diag(K_test_test)), # std. der. of variance to itself 
                K_test_test,
            )
            y_mean_noisy, y_std_noisy, y_cov_noisy = mean_pred_distribution, std_pred_distribution, conv_pred_distribution
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

            return y_mean_noisy, y_std_noisy, y_cov_noisy

        else:  # Predict based on GP posterior
            # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
            K_test_test = self.kernel(X_test) # Covariance matrix of test data
            K_test_train = self.kernel(X_test, self.X_train) # Covariance matrix of test and train data
            v = solve_triangular(self.L_, K_test_train.T, lower=True) # solve for v for predictive varianace
            conv = K_test_test - np.matmul(v.T, v) # compute predictive variance
            mean_pred_distribution, std_pred_distribution, conv_pred_distribution = (
                np.matmul(K_test_train, self.alpha_), # according to literature should be K_test_train.T but dimensions do not match (?)
                np.sqrt(np.diag(conv)),
                conv,
            )
            # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

            return mean_pred_distribution, std_pred_distribution, conv_pred_distribution
