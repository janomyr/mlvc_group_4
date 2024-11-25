import cvxopt
import numpy as np

########### TO-DO ###########
# 1. Implement linear kernel
#   --> See: def linear_kernel(x1, x2):
# 2. Implement rbf kernel
#   --> See: def rbf_kernel(x1, x2):
# 3. Implement fit
#   --> See: def fit(self, X, y):
#   --> Add matrix Q, p, G, h, A, b and save the solution
# 4. Implement predict
#   --> See: def predict(self, X):


class SVM:
    """Implements the support vector machine"""

    def __init__(self, kernel="linear", sigma=0.25):
        """Initialize perceptron."""
        self.__alphas = None
        self.__targets = None
        self.__training_X = None
        self.__bias = None
        if kernel == "linear":
            self.__kernel = SVM.linear_kernel
        elif kernel == "rbf":
            self.__kernel = SVM.rbf_kernel
            self.__sigma = sigma
        else:
            raise ValueError("Invalid kernel")

    @staticmethod
    def linear_kernel(x1, x2):
        """
        Computes the linear kernel between two sets of vectors.

        Args:
            x1 (numpy.ndarray): A matrix of shape (n_samples_1, n_features) representing the first set of vectors.
            x2 (numpy.ndarray): A matrix of shape (n_samples_2, n_features) representing the second set of vectors.

        Returns:
            numpy.ndarray: A matrix of shape (n_samples_1, n_samples_2) representing the linear kernel between x1 and x2.
        """
        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Linear kernel means the inner products of the samples are the output.
        # Therefore, x2 has to be transposed in order to get the n_sample x n_sample matrix.
        return np.dot(x1,x2.T)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    @staticmethod
    def rbf_kernel(x1, x2, sigma):
        """
        Computes the radial basis function (RBF) kernel between two sets of vectors.

        Args:
            x1: A matrix of shape (n_samples_1, n_features) representing the first set of vectors.
            x2: A matrix of shape (n_samples_2, n_features) representing the second set of vectors.

        Returns:
            A matrix of shape (n_samples_1, n_samples_2) representing the RBF kernel between x1 and x2.
        """

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # RBF kernel means using the formula
        #           ||x1 - x2||²          
        #         - -----------
        #   K = e^   2*sigma²
        diff = x1[:, None,:] - x2[None,:,:]     # subtract each element of each vector from each other
        dist = np.linalg.norm(diff, axis=-1)**2 # norm the difference -> distance
        return np.exp(-dist / (2*(sigma**2)))
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

    def fit(self, X, y):
        """Training function.

        Args:
            X (numpy.ndarray): Inputs.
            y (numpy.ndarray): labels/target.

        Returns:
            None
        """
        # n_observations -> number of training examples
        # m_features -> number of features
        n_observations, m_features = X.shape
        self.__norm = max(np.linalg.norm(X, axis=1))
        X = X / self.__norm
        y = y.reshape((1, n_observations))

        # quadprog and cvx all want 64 bits
        X = X.astype(np.float64)
        y = y.astype(np.float64)

        print("Computing kernel matrix...")
        if self.__kernel == SVM.linear_kernel:
            K = self.__kernel(X, X)
        elif self.__kernel == SVM.rbf_kernel:
            K = self.__kernel(X, X, self.__sigma)
        print("Done.")

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # SEE: https://cvxopt.org/examples/tutorial/qp.html and https://cvxopt.org/userguide/coneprog.html#quadratic-programming and http://www.seas.ucla.edu/~vandenbe/publications/mlbook.pdf
        Q = cvxopt.matrix(np.outer(y,y) * K)
        p = cvxopt.matrix(-1 * np.ones(n_observations))
        G = cvxopt.matrix(-1 * np.eye(n_observations))
        h = cvxopt.matrix(np.zeros((n_observations)))
        A = cvxopt.matrix(y)
        b = cvxopt.matrix(0.0)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        cvxopt.solvers.options["show_progress"] = False
        solution = cvxopt.solvers.qp(Q, p, G, h, A, b)

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****
        # Save the solution
        self.__alphas = np.ravel(solution['x'])
        sv_ind = np.where(self.__alphas > 1e-5)[0]  # Look for support vector indices
        y_sv = y[0][sv_ind]                         # y-values for support vectors
        self.__bias = np.mean(y_sv - np.sum((y*self.__alphas)*K[:][sv_ind], axis=1))
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****

        self.__targets = y
        self.__training_X = X

    def predict(self, X):
        """Prediction function.

        Args:
            X (numpy.ndarray): Inputs.

        Returns:
            Class label of X
        """

        X = X / self.__norm

        # *****BEGINNING OF YOUR CODE (DO NOT DELETE THIS LINE)*****     
        if self.__kernel == SVM.linear_kernel:
            K = self.__kernel(X, self.__training_X)
            return np.sign(np.sum(((self.__alphas * self.__targets)*K), axis=1) + self.__bias)
        elif self.__kernel == SVM.rbf_kernel:
            K = self.__kernel(X, self.__training_X, self.__sigma)
            return np.sign(np.sum(((self.__alphas * self.__targets)*K), axis=1) + self.__bias)
        # *****END OF YOUR CODE (DO NOT DELETE THIS LINE)*****
