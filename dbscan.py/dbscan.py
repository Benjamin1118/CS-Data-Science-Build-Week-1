import numpy as np
import warnings
from scipy import sparse

from ..base import BaseEstimator, ClusterMixin
from ..utils.validation import _check_sample_weight, _deprecate_positional_args
from ..neighbors import NearestNeighbors

from ._dbscan_inner import dbscan_inner


@_deprecate_positional_args
def dbscan(X, eps=0.5, *, min_samples=5, metric='minkowski',
           metric_params=None, algorithm='auto', leaf_size=30, p=2,
           sample_weight=None, n_jobs=None):

est = DBSCAN(eps=eps, min_samples=min_samples, metric=metric,
                 metric_params=metric_params, algorithm=algorithm,
                 leaf_size=leaf_size, p=p, n_jobs=n_jobs)
    est.fit(X, sample_weight=sample_weight)
    return est.core_sample_indices_, est.labels_

class DBSCAN(ClusterMixin, BaseEstimator):
     @_deprecate_positional_args
    def __init__(self, eps=0.5, *, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None):
         X = self._validate_data(X, accept_sparse='csr')

        if not self.eps > 0.0:
            raise ValueError("eps must be positive.")

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Calculate neighborhood for all samples. This leaves the original
        # point in, which needs to be considered later (i.e. point i is in the
        # neighborhood of point i. While True, its useless information)
        if self.metric == 'precomputed' and sparse.issparse(X):
            # set the diagonal to explicit values, as a point is its own
            # neighbor
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', sparse.SparseEfficiencyWarning)
                X.setdiag(X.diagonal())  # XXX: modifies X's internals in-place

        neighbors_model = NearestNeighbors(
            radius=self.eps, 
            algorithm=self.algorithm,
            leaf_size=self.leaf_size, 
            metric=self.metric,
            metric_params=self.metric_params, 
            p=self.p, 
            
            
            n_jobs=self.n_jobs)
        neighbors_model.fit(X)
        # This has worst case O(n^2) memory complexity
        neighborhoods = neighbors_model.radius_neighbors(X,
                                                         return_distance=False)

        if sample_weight is None:
            n_neighbors = np.array([len(neighbors)
                                    for neighbors in neighborhoods])
        else:
            n_neighbors = np.array([np.sum(sample_weight[neighbors])
                                    for neighbors in neighborhoods])

        # Initially, all samples are noise.
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        # A list of all core samples found.
        core_samples = np.asarray(n_neighbors >= self.min_samples,
                                  dtype=np.uint8)
        dbscan_inner(core_samples, neighborhoods, labels)

        self.core_sample_indices_ = np.where(core_samples)[0]
        self.labels_ = labels

        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
       
        self.fit(X, sample_weight=sample_weight)
        return self.labels_