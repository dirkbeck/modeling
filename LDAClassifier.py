from numpy.linalg import inv,pinv
from numpy.linalg import eig
import numpy as np

class LDAClassifier:

    def __init__(self, projection_dim=None):
        self.projection_dim = projection_dim
        self.W = None  # weights
        self.g_means, self.g_covariance, self.priors = None, None, None

    def fit(self, X):
        means_k = self.__compute_means(X)

        Sks = []
        for class_i, m in means_k.items():
            sub = np.subtract(X[class_i], m)
            Sks.append(np.matmul(np.transpose(sub), sub))

        Sks = np.asarray(Sks)
        Sw = np.sum(Sks, axis=0)  # shape = (D,D)

        Nk = {}
        sum_ = 0
        for class_id, data in X.items():
            Nk[class_id] = data.shape[0]
            sum_ += np.sum(data, axis=0)

        self.N = sum(list(Nk.values()))

        # m is the mean of the total data set
        m = sum_ / self.N

        SB = []
        for class_id, mean_class_i in means_k.items():
            sub_ = mean_class_i - m
            SB.append(np.multiply(Nk[class_id], np.outer(sub_, sub_.T)))

        # between class covariance matrix shape = (D,D). D = input vector dimensions
        SB = np.sum(SB, axis=0)  # sum of K (# of classes) matrices

        matrix = np.matmul(pinv(Sw), SB)
        # find eigen values and eigen-vectors pairs for this multiplication
        eigen_values, eigen_vectors = eig(matrix)

        eiglist = [(eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values))]

        # sort the eigvals in decreasing order
        eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)

        # take the first num_dims eigvectors
        self.W = np.array([eiglist[i][1] for i in range(self.projection_dim)])
        self.W = np.asarray(self.W).T

        # get parameter of the Gaussian distribution
        self.g_means, self.g_covariance, self.priors = self.gaussian(X)

    # Returns the parameters of the Gaussian distributions
    def gaussian(self, X):
        means = {}
        covariance = {}
        priors = {}  # p(Ck)
        for class_id, values in X.items():
            proj = np.dot(values, self.W)
            means[class_id] = np.mean(proj, axis=0)
            covariance[class_id] = np.cov(proj, rowvar=False)
            priors[class_id] = values.shape[0] / self.N
        return means, covariance, priors

    # model a multi-variate Gaussian distribution for each class’ likelihood distribution P(x|Ck)
    def gaussian_distribution(self, x, u, cov):
        scalar = (1. / ((2 * np.pi) ** (x.shape[0] / 2.))) * (1 / np.sqrt(np.linalg.det(cov)))
        x_sub_u = np.subtract(x, u)
        return scalar * np.exp(-np.dot(np.dot(x_sub_u, inv(cov)), x_sub_u.T) / 2.)

    def score(self, X, y):
        proj = self.project(X)
        gaussian_likelihoods = []
        classes = sorted(list(self.g_means.keys()))

        for x in proj:
            row = []
            for c in classes:  # number of classes
                # Compute the posterior P(Ck|x) prob of a class k given a point x
                res = self.priors[c] * self.gaussian_distribution(x, self.g_means[c], self.g_covariance[c])
                row.append(res)

            gaussian_likelihoods.append(row)

        # normalized
        likelihoods = gaussian_likelihoods / np.expand_dims(np.sum(gaussian_likelihoods, 1), axis=-1)

        # assign x to the class with the largest posterior probability
        predictions = np.argmax(gaussian_likelihoods, axis=1)
        return np.sum(predictions == y) / len(y), predictions, proj, likelihoods

    def project(self, X):
        return np.dot(X, self.W)

    def __compute_means(self, X):
        # Compute the means for each class k=1,2,3...K
        # If the dataset has K classes, then, self.means_k.shape = [# of records, K]
        means_k = {}
        for class_i, input_vectors in X.items():
            means_k[class_i] = np.mean(input_vectors, axis=0)
        return means_k

    def dimimportance(self, X, x):
        # find the value lost in classification at a given point when latent dimensions are removed
        # for results on all possible latent dimensions, set self.projection_dim to n classes - 1
        xproj = self.project(x.reshape(1,-1))
        meansproj,_,_ = self.gaussian(X)
        classes = sorted(list(self.g_means.keys()))
        res = np.zeros([len(classes),len(classes)-1])

        for c in range(2,classes):
            # res[c,:xproj.shape[1]] = self.priors[c] * self.gaussian_distribution(x, self.g_means[c], self.g_covariance[c])
            print(self.priors[c] * self.gaussian_distribution(x, self.g_means[c], self.g_covariance[c]))

        return res