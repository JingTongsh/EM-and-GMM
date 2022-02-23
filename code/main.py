import numpy as np
import math
from sklearn.datasets import make_blobs
from sklearn.cluster import kmeans_plusplus
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def nearest_center(point, centers):
    """
    Args:
    - point: 1d array of shape (n_dim,)
    - centers: 2d array of shape (n_components, n_dim)
    
    Returns:
    - min_id: an integer
    """
    n_components = centers.shape[0]
    min_dist, min_id = 0.0, 0
    for k in range(n_components):
        dist = np.linalg.norm(point - centers[k, :])
        if k == 0 or dist < min_dist:
            min_dist = dist
            min_id = k

    return min_id


def assign_labels(data, centers):
    """
    Assign 'hard' labels; scalars, not one-hot
    
    Args:
    - data: 2d array of shape (n_data, n_dim)
    - centers: 2d array of shape (n_components, n_dim)
    
    Returns:
    - labels: 1d array of shape (n_data,)
    """
    n_data = data.shape[0]
    labels = np.zeros([n_data], dtype=int)
    for n in range(n_data):
        labels[n] = nearest_center(data[n, :], centers)

    return labels


def make_clusters(data, centers):
    """
    Args:
    - data: 2d array of shape (n_data, n_dim)
    - centers: 2d array of shape (n_components, n_dim)
    
    Returns:
    - clusters: list of lists
    """
    n_data = data.shape[0]
    n_components = centers.shape[0]
    clusters = []
    for _ in range(n_components):
        clusters.append([])

    for n in range(n_data):
        label = nearest_center(data[n, :], centers)
        clusters[label].append(data[n, :])

    return clusters


def show_process(data, centers, clusters, title):
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    n_components = centers.shape[0]
    plt.figure()
    if title == 'init':
        plt.scatter(data[:, 0], data[:, 1], s=5)
    else:
        for k in range(n_components):
            if len(clusters[k]) > 0:
                clst = np.array(clusters[k])
                plt.scatter(clst[:, 0], clst[:, 1], s=5, c=color[k])
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x')
    plt.title('my kmeans: ' + title)


def k_means(data, n_components: int, max_iter: int = 5):
    n_data, n_dim = data.shape
    # init
    centers = np.random.rand(n_components, n_dim)
    for d in range(n_dim):
        low, high = np.min(data[:, d]), np.max(data[:, d])
        centers[:, d] = centers[:, d] * (high - low) + low
    
    show_details = False
    if show_details:
        show_process(data, centers, None, 'init')

    # train
    for it in range(1, max_iter + 1):
        # E step
        clusters = make_clusters(data, centers)
        if show_details:
            show_process(data, centers, clusters, 'E{}'.format(it))

        # M step
        for k in range(n_components):
            if len(clusters[k]) > 0:
                clst = np.array(clusters[k])
                centers[k, :] = np.mean(clst, axis=0)
            else:
                centers[k, :] = np.random.rand(n_dim)
                for d in range(n_dim):
                    low, high = np.min(data[:, d]), np.max(data[:, d])
                    centers[k, d] = centers[k, d] * (high - low) + low
        if show_details:
            show_process(data, centers, clusters, 'M{}'.format(it))

    labels = assign_labels(data, centers)

    return centers, labels


def gaussian_density(data, mu, cov):
    n_data = data.shape[0]
    n_dim = mu.shape[0]
    densities = np.zeros([n_data])
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)

    for n in range(n_data):
        diff = data[n, :] - mu
        tmp1 = np.exp(diff.T.dot(inv).dot(diff) / (-2))
        tmp2 = ((2 * math.pi) ** n_dim) * det
        densities[n] = tmp1 / math.sqrt(tmp2)

    return densities


def plot_ellipse(mu, cov, ax, n_std=3.0):
    """
    ref:
    <https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html>
    """

    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor='r')

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x, mean_y = mu

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def gmm_em(data, n_components: int = 2, max_iter: int = 10):
    assert data.ndim == 2
    assert n_components > 0
    assert max_iter > 0
    n_data, n_dim = data.shape

    # initialize centers
    centers, _ = k_means(data, n_components, 5)

    # initialize covariances
    init_clusters = make_clusters(data, centers)
    covariances = np.zeros([n_components, n_dim, n_dim])
    for k in range(n_components):
        clst = np.array(init_clusters[k])
        covariances[k, :, :] = np.cov(clst[:, 0], clst[:, 1])

    # initialize mixture coefficients
    mix_coefficients = np.zeros([n_components])
    for k in range(n_components):
        mix_coefficients[k] = len(init_clusters[k]) / n_data

    # initialize probabilities
    x_given_parameters = np.zeros([n_data, n_components])
    responsibilities = np.zeros([n_data, n_components])

    for it in range(1, max_iter + 2):
        # E step: update posterior distribution
        for k in range(n_components):
            x_given_parameters[:, k] = gaussian_density(data, centers[k, :], covariances[k, :, :])
            responsibilities[:, k] = x_given_parameters[:, k] * mix_coefficients[k]
        norm_factor = np.sum(responsibilities, axis=1)
        for n in range(n_data):
            if norm_factor[n] != 0:
                responsibilities[n, :] /= norm_factor[n]

        if it == max_iter + 1:
            break

        # M step: update parameters
        n_effective = np.sum(responsibilities, axis=0)
        for k in range(n_components):
            centers[k, :] = data.T.dot(responsibilities[:, k]) / n_effective[k]
            diff = data - centers[k, :]
            covariances[k, :, :] = diff.T.dot(np.diag(responsibilities[:, k])).dot(diff)
        mix_coefficients = n_effective / n_data

    max_r = np.max(responsibilities, axis=1)
    prediction = np.zeros([n_data], dtype=int)
    for n in range(n_data):
        for k in range(n_components):
            if responsibilities[n, k] == max_r[n]:
                prediction[n] = k
                break

    return prediction


def main():
    # Generate sample data
    n_samples = 5000
    n_components = 5

    x_data, y_true = make_blobs(n_samples=n_samples, centers=n_components, cluster_std=0.60, random_state=0)
    x_data = x_data[:, ::-1]  # 对调X[:, 0]和X[:, 1]

    plt.figure()
    plt.scatter(x_data[:, 0], x_data[:, 1], s=5, c=y_true, cmap='viridis')
    plt.title('ground truth')

    centers_1, _ = kmeans_plusplus(x_data, n_clusters=5, random_state=0)
    labels_1 = assign_labels(x_data, centers_1)

    plt.figure()
    plt.scatter(x_data[:, 0], x_data[:, 1], s=5, c=labels_1, cmap='viridis')
    plt.scatter(centers_1[:, 0], centers_1[:, 1], c='r', marker='x')
    plt.title('sklearn kmeans++')

    centers_2, labels_2 = k_means(x_data, n_components=5)

    plt.figure()
    plt.scatter(x_data[:, 0], x_data[:, 1], s=5, c=labels_2, cmap='viridis')
    plt.scatter(centers_2[:, 0], centers_2[:, 1], c='r', marker='x')
    plt.title('my kmeans')
    
    gmm = GMM(n_components=5).fit(x_data)  # 指定聚类中心个数为4
    labels = gmm.predict(x_data)

    plt.figure()
    plt.scatter(x_data[:, 0], x_data[:, 1], c=labels, s=5, cmap='viridis')
    plt.title('sklearn gmm')

    prediction = gmm_em(x_data, n_components=5)

    plt.figure()
    plt.scatter(x_data[:, 0], x_data[:, 1], c=prediction, s=5, cmap='viridis')
    plt.title('my gmm')
    plt.show()


if __name__ == '__main__':
    main()
