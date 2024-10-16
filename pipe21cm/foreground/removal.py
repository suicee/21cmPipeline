from sklearn.decomposition import PCA


def pca_removal(data, n_components=5):
    """
    Remove foregrounds using PCA.

    Args:
        :data: np.ndarray. The data cube. Assuing the last dimension is frequency.
        :n_components: int. The number of components to remove in the PCA.

    Returns:
        :res: np.ndarray. The residual after foreground removal.
    """

    # assume last dimension is frequency
    d_flat = data.reshape(-1, data.shape[-1])

    pca = PCA(n_components=n_components)
    fg_est=pca.fit_transform(d_flat)
    fg_est=pca.inverse_transform(fg_est)

    fg_est=fg_est.reshape(data.shape)
    res=data-(fg_est)

    return res