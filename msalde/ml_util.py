import numpy as np
from sklearn.metrics import roc_curve


def create_gaussian_mixture(data, n_components):
    """
    Create a Gaussian Mixture Model from the given data.

    Parameters:
    data (array-like): The input data for fitting the Gaussian Mixture Model.
    n_components (int): The number of mixture components.

    Returns:
    GaussianMixture: Fitted Gaussian Mixture Model.
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=n_components, n_init=10, random_state=42)
    data = data.reshape(-1, 1)  # Reshape data to be a 2D array with one feature
    gmm.fit(data)
    x = np.linspace(data.min() - 1, data.max() + 1, 1000)
    logprob = gmm.score_samples(x.reshape(-1, 1))
    pdf = np.exp(logprob)

    # Find minimum between the peaks, threshold separating the two Gaussians
    means = np.sort(gmm.means_.flatten())
    between_means_mask = (x > means[0]) & (x < means[1])
    threshold = x[between_means_mask][np.argmin(pdf[between_means_mask])]

    return gmm, x, pdf, threshold


def calculate_optimal_youden_index(y_true, y_scores):
    """
    Compute the Youden's Index given sensitivity and specificity.

    Parameters:
    sensitivity (float): Sensitivity value.
    specificity (float): Specificity value.

    Returns:
    float: Youden's Index.
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    return optimal_idx, fpr, tpr, thresholds

