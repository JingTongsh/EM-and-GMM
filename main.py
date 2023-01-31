import numpy as np
import math
from sklearn.datasets import make_blobs
from sklearn.cluster import kmeans_plusplus
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
import os


def nearest_center(point: np.ndarray, centers: np.ndarray) -> int:
    """
    Args:
    - point: 1D array of shape (n_dim)
    - centers: 2D array of shape (n_components, n_dim)
    
    Returns:
    - center_id: int
    """
    assert point.ndim == 1 and centers.ndim == 2 and point.shape[0] == centers.shape[1], f'{point.shape}, {centers.shape}'
    center_id = np.linalg.norm(point - centers, axis=1).argmin()

    return center_id


def assign_labels(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Assign 'hard' labels in scalar form, not one-hot
    
    Args:
    - data: 2D array of shape (n_data, n_dim)
    - centers: 2D array of shape (n_components, n_dim)
    
    Returns:
    - labels: 1D int array of shape (n_data)
    """
    assert data.ndim == 2 and centers.ndim == 2 and data.shape[1] == centers.shape[1], f'{data.shape}, {centers.shape}'
    n_data = data.shape[0]
    labels = np.zeros([n_data], dtype=int)
    for n in range(n_data):
        labels[n] = nearest_center(data[n, :], centers)

    return labels


def show_kmeans(ax, data: np.ndarray, centers: np.ndarray, labels: np.ndarray, title: str):
    if centers is None or labels is None:
        ax.scatter(data[:, 0], data[:, 1], s=5)
    else:
        ax.scatter(data[:, 0], data[:, 1], s=5, c=labels, cmap='viridis')
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x')
    ax.title.set_text('my kmeans: ' + title)


def k_means(data: np.ndarray, n_components: int, max_iter: int = 5, verbose: bool = False) -> tuple:
    assert data.ndim == 2, f'{data.shape}'
    n_dim = data.shape[1]
    # initialize randomly
    centers = np.random.rand(n_components, n_dim)
    for d in range(n_dim):
        low, high = np.min(data[:, d]), np.max(data[:, d])
        centers[:, d] = centers[:, d] * (high - low) + low
    
    if verbose:
        fig, axs = plt.subplots(2, max_iter + 1, figsize=(20, 10))
        show_kmeans(axs[0, 0], data, centers=None, labels=None, title='observed data')
        show_kmeans(axs[1, 0], data, centers, labels=None, title='init')

    # train
    for it in range(1, max_iter + 1):
        # E step
        prediction = assign_labels(data, centers)
        if verbose:
            show_kmeans(axs[0, it], data, centers, prediction, f'E{it}')

        # M step
        for k in range(n_components):
            cluster_mask = prediction == k
            if cluster_mask.sum() > 0:
                cluster = data[np.ix_(cluster_mask)]
                centers[k, :] = cluster.mean(axis=0)
            else:
                centers[k, :] = np.random.rand(n_dim)
                for d in range(n_dim):
                    low, high = np.min(data[:, d]), np.max(data[:, d])
                    centers[k, d] = centers[k, d] * (high - low) + low
        if verbose:
            show_kmeans(axs[1, it], data, centers, prediction, f'M{it}')

    labels = assign_labels(data, centers)

    if verbose:
        fig.savefig('results/kmeans_visualize.png')
        plt.show()

    return (centers, labels)


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


def gmm_assign_labels(data: np.ndarray, centers: np.ndarray, covariances: np.ndarray, mix_coefficients: np.ndarray) -> np.ndarray:
    n_data = data.shape[0]
    n_components = centers.shape[0]
    density_component = np.zeros([n_data, n_components])
    posterior = np.zeros([n_data, n_components])
    for k in range(n_components):
        density_component[:, k] = gaussian_density(data, centers[k, :], covariances[k, :, :])
        posterior[:, k] = density_component[:, k] * mix_coefficients[k]
    norm_factor = posterior.sum(axis=1)
    for n in range(n_data):
        if norm_factor[n] != 0:
            posterior[n, :] /= norm_factor[n]

    prediction = posterior.argmax(axis=1)

    return prediction


def show_gmm(ax, data: np.ndarray, centers: np.ndarray, covariances: np.ndarray, posterior: np.ndarray, title: str):
    if posterior is not None:
        prediction = posterior.argmax(axis=1)
        ax.scatter(data[:, 0], data[:, 1], s=5, c=prediction, cmap='viridis')
    else:
        ax.scatter(data[:, 0], data[:, 1], s=5)
    if centers is not None:
        n_components = centers.shape[0]
        val, vec = np.linalg.eig(covariances)  # somtimes nan or inf
        offset = np.expand_dims(np.sqrt(val), axis=1) * vec * 3
        a = np.array([centers + offset[:, :, 0], centers - offset[:, :, 0]])
        b = np.array([centers + offset[:, :, 1], centers - offset[:, :, 1]])
        for k in range(n_components):
            ax.plot(a[:, k, 0], a[:, k, 1], color='red')
            ax.plot(b[:, k, 0], b[:, k, 1], color='red')
    ax.axis('equal')
    ax.title.set_text('my gmm: ' + title)


def gmm_em(data: np.ndarray, n_components: int, max_iter: int = 5, verbose: bool = False) -> tuple:
    assert data.ndim == 2
    assert n_components > 0
    assert max_iter > 0
    n_data, n_dim = data.shape
    
    # initialize parameters
    centers, labels = k_means(data, n_components, max_iter=5, verbose=False)
    centers = np.random.rand(n_components, n_dim)
    for d in range(n_dim):
        low, high = np.min(data[:, d]), np.max(data[:, d])
        centers[:, d] = centers[:, d] * (high - low) + low
    labels = assign_labels(data, centers)
    counts = np.zeros([n_components])

    covariances = np.zeros([n_components, n_dim, n_dim])
    for k in range(n_components):
        cluster_mask = labels == k
        counts[k] = cluster_mask.sum()
        cluster = data[np.ix_(cluster_mask)]
        covariances[k, :, :] = np.cov(cluster.T)

    mix_coefficients = np.array(counts) / n_data

    # initialize probabilities
    density_component = np.zeros([n_data, n_components])
    posterior = np.zeros([n_data, n_components])

    if verbose:
        fig, axs = plt.subplots(2, max_iter + 1, figsize=(20, 10))
        show_gmm(axs[0, 0], data, centers=None, covariances=None, posterior=None, title='observed data')
        show_gmm(axs[1, 0], data, centers, covariances, posterior=None, title='init')

    for it in range(1, max_iter + 1):
        # E step: update posterior distribution
        for k in range(n_components):
            density_component[:, k] = gaussian_density(data, centers[k, :], covariances[k, :, :])
            posterior[:, k] = density_component[:, k] * mix_coefficients[k]
        norm_factor = posterior.sum(axis=1)
        for n in range(n_data):
            if norm_factor[n] != 0:
                posterior[n, :] /= norm_factor[n]
        
        if verbose:
            show_gmm(axs[0, it], data, centers, covariances, posterior, title=f'E{it}')

        # M step: update parameters
        n_effective = posterior.sum(axis=0)
        centers = posterior.T.dot(data) / n_effective.reshape(-1, 1)
        for k in range(n_components):
            diff = data - centers[k, :]
            covariances[k, :, :] = diff.T.dot(np.diag(posterior[:, k])).dot(diff) / n_effective[k]
        mix_coefficients = n_effective / n_data

        if verbose:
            # print(f'iter {it}')
            # print('n_eff', n_effective)
            # print('center', centers)
            # print('covariance', covariances)
            # print('mix', mix_coefficients)
            # print('posterior', posterior)
            show_gmm(axs[1, it], data, centers, covariances, posterior, title=f'M{it}')
        
    prediction = posterior.argmax(axis=1)
    if verbose:
        fig.savefig('results/em_visualize.png')
        plt.show()

    return (centers, covariances, mix_coefficients, prediction)


def main():
    # Generate sample data
    n_samples = 5000
    n_train = 4000
    n_components_real = 5
    n_components = 5
    verbose = True
    res_dir = 'results'
    if verbose and not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    x_data, y_data = make_blobs(n_samples=n_samples, centers=n_components_real, cluster_std=0.60, random_state=2023)
    x_data = x_data[:, ::-1]  # 对调x_data[:, 0]和x_data[:, 1]
    
    # Split train and test data
    idx = np.linspace(0, n_samples, n_samples, endpoint=False, dtype=int)
    np.random.shuffle(idx)
    x_train = x_data[idx[0: n_train]]
    y_train = y_data[idx[0: n_train]]
    x_test = x_data[idx[n_train: n_samples]]
    y_test = y_data[idx[n_train: n_samples]]
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    
    pred_train, pred_test = {}, {}
    
    # sklearn kmeans++
    centers_1, _ = kmeans_plusplus(x_train, n_clusters=n_components, random_state=0)
    pred_train[0] = assign_labels(x_train, centers_1)
    pred_test[0] = assign_labels(x_test, centers_1)

    # my kmeans
    centers_2, pred_train[1] = k_means(x_train, n_components, max_iter=5, verbose=verbose)
    pred_test[1] = assign_labels(x_test, centers_2)
    
    # sklearn gmm
    gmm = GMM(n_components).fit(x_train)
    pred_train[2] = gmm.predict(x_train)
    pred_test[2] = gmm.predict(x_test)

    # my gmm
    centers_4, covariances_4, mix_coefficients_4, pred_train[3] = gmm_em(x_train, n_components, max_iter=5, verbose=verbose)
    pred_test[3] = gmm_assign_labels(x_test, centers_4, covariances_4, mix_coefficients_4)
    
    for k in range(2):
        fig, axs = plt.subplots(2, 3)
        if k == 0:
            x, y, pred = x_train, y_train, pred_train
            title = 'train'
            fig.suptitle('train set')
        else:
            x, y, pred = x_test, y_test, pred_test
            title = 'test'
            fig.suptitle('test set')

        axs[0, 0].scatter(x[:, 0], x[:, 1], s=5, c='#1f77b4')
        axs[0, 0].axis('equal')
        axs[0, 0].title.set_text('observed data')

        axs[1, 0].scatter(x[:, 0], x[:, 1], s=5, c=y, cmap='viridis')
        axs[1, 0].axis('equal')
        axs[1, 0].title.set_text('ground truth')

        axs[0, 1].scatter(x[:, 0], x[:, 1], s=5, c=pred[0], cmap='viridis')
        axs[0, 1].scatter(centers_1[:, 0], centers_1[:, 1], c='r', marker='x')
        axs[0, 1].axis('equal')
        axs[0, 1].title.set_text('sklearn kmeans++')

        axs[0, 2].scatter(x[:, 0], x[:, 1], s=5, c=pred[1], cmap='viridis')
        axs[0, 2].scatter(centers_2[:, 0], centers_2[:, 1], c='r', marker='x')
        axs[0, 2].axis('equal')
        axs[0, 2].title.set_text('my kmeans')

        axs[1, 1].scatter(x[:, 0], x[:, 1], c=pred[2], s=5, cmap='viridis')
        axs[1, 1].axis('equal')
        axs[1, 1].title.set_text('sklearn gmm')

        axs[1, 2].scatter(x[:, 0], x[:, 1], c=pred[3], s=5, cmap='viridis')
        axs[1, 2].axis('equal')
        axs[1, 2].title.set_text('my gmm')
        
        fig.tight_layout()
        fig.savefig(f'results/{title}.png')
        plt.show()


if __name__ == '__main__':
    main()
