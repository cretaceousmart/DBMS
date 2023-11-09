import numpy as np
import scipy.stats
import scipy.sparse
import scipy.misc
import scipy.special
import warnings
from externel_library.mir_eval_simple import mir_eval_simple_utils


def validate_structure(reference_intervals, reference_labels,
                       estimated_intervals, estimated_labels):
    """Checks that the input annotations to a structure estimation metric (i.e.
    one that takes in both segment boundaries and their labels) look like valid
    segment times and labels, and throws helpful errors if not.

    Parameters
    ----------
    reference_intervals : np.ndarray, shape=(n, 2)
        reference segment intervals, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.

    reference_labels : list, shape=(n,)
        reference segment labels, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.

    estimated_intervals : np.ndarray, shape=(m, 2)
        estimated segment intervals, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.

    estimated_labels : list, shape=(m,)
        estimated segment labels, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.

    """
    for (intervals, labels) in [(reference_intervals, reference_labels),
                                (estimated_intervals, estimated_labels)]:

        mir_eval_simple_utils.validate_intervals(intervals)
        if intervals.shape[0] != len(labels):
            raise ValueError('Number of intervals does not match number '
                             'of labels')

        # Check only when intervals are non-empty
        if intervals.size > 0:
            # Make sure intervals start at 0
            if not np.allclose(intervals.min(), 0.0):
                raise ValueError('Segment intervals do not start at 0')

    if reference_intervals.size == 0:
        warnings.warn("Reference intervals are empty.")
    if estimated_intervals.size == 0:
        warnings.warn("Estimated intervals are empty.")
    # Check only when intervals are non-empty
    if reference_intervals.size > 0 and estimated_intervals.size > 0:
        if not np.allclose(reference_intervals.max(),
                           estimated_intervals.max()):
            raise ValueError('End times do not match')

def _contingency_matrix(reference_indices, estimated_indices):
    """Computes the contingency matrix of a true labeling vs an estimated one.

    Parameters
    ----------
    reference_indices : np.ndarray
        Array of reference indices
    estimated_indices : np.ndarray
        Array of estimated indices

    Returns
    -------
    contingency_matrix : np.ndarray
        Contingency matrix, shape=(#reference indices, #estimated indices)
    .. note:: Based on sklearn.metrics.cluster.contingency_matrix

    """
    ref_classes, ref_class_idx = np.unique(reference_indices,
                                           return_inverse=True)
    est_classes, est_class_idx = np.unique(estimated_indices,
                                           return_inverse=True)
    n_ref_classes = ref_classes.shape[0]
    n_est_classes = est_classes.shape[0]
    # Using coo_matrix is faster than histogram2d
    return scipy.sparse.coo_matrix((np.ones(ref_class_idx.shape[0]),
                                    (ref_class_idx, est_class_idx)),
                                   shape=(n_ref_classes, n_est_classes),
                                   dtype=np.int64).toarray()

def nce(reference_intervals, reference_labels, estimated_intervals,
      estimated_labels, frame_size=0.1, beta=1.0, marginal=False):
    """Frame-clustering segmentation: normalized conditional entropy

    Computes cross-entropy of cluster assignment, normalized by the
    max-entropy.

    Examples
    --------
    >>> (ref_intervals,
    ...  ref_labels) = mir_eval.io.load_labeled_intervals('ref.lab')
    >>> (est_intervals,
    ...  est_labels) = mir_eval.io.load_labeled_intervals('est.lab')
    >>> # Trim or pad the estimate to match reference timing
    >>> (ref_intervals,
    ...  ref_labels) = mir_eval.util.adjust_intervals(ref_intervals,
    ...                                               ref_labels,
    ...                                               t_min=0)
    >>> (est_intervals,
    ...  est_labels) = mir_eval.util.adjust_intervals(
    ...     est_intervals, est_labels, t_min=0, t_max=ref_intervals.max())
    >>> S_over, S_under, S_F = mir_eval.structure.nce(ref_intervals,
    ...                                               ref_labels,
    ...                                               est_intervals,
    ...                                               est_labels)

    Parameters
    ----------
    reference_intervals : np.ndarray, shape=(n, 2)
        reference segment intervals, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.
    reference_labels : list, shape=(n,)
        reference segment labels, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.
    estimated_intervals : np.ndarray, shape=(m, 2)
        estimated segment intervals, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.
    estimated_labels : list, shape=(m,)
        estimated segment labels, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.
    frame_size : float > 0
        length (in seconds) of frames for clustering
        (Default value = 0.1)
    beta : float > 0
        beta for F-measure
        (Default value = 1.0)

    marginal : bool
        If `False`, normalize conditional entropy by uniform entropy.
        If `True`, normalize conditional entropy by the marginal entropy.
        (Default value = False)

    Returns
    -------
    S_over
        Over-clustering score:

        - For `marginal=False`, ``1 - H(y_est | y_ref) / log(|y_est|)``

        - For `marginal=True`, ``1 - H(y_est | y_ref) / H(y_est)``

        If `|y_est|==1`, then `S_over` will be 0.

    S_under
        Under-clustering score:

        - For `marginal=False`, ``1 - H(y_ref | y_est) / log(|y_ref|)``

        - For `marginal=True`, ``1 - H(y_ref | y_est) / H(y_ref)``

        If `|y_ref|==1`, then `S_under` will be 0.

    S_F
        F-measure for (S_over, S_under)

    """

    validate_structure(reference_intervals, reference_labels,
                        estimated_intervals, estimated_labels)

    # Check for empty annotations.  Don't need to check labels because
    # validate_structure makes sure they're the same size as intervals
    if reference_intervals.size == 0 or estimated_intervals.size == 0:
        return 0., 0., 0.

    # Generate the cluster labels
    y_ref = mir_eval_simple_utils.intervals_to_samples(reference_intervals,
                                        reference_labels,
                                        sample_size=frame_size)[-1]

    y_ref = mir_eval_simple_utils.index_labels(y_ref)[0]

    # Map to index space
    y_est = mir_eval_simple_utils.intervals_to_samples(estimated_intervals,
                                        estimated_labels,
                                        sample_size=frame_size)[-1]

    y_est = mir_eval_simple_utils.index_labels(y_est)[0]

    # Make the contingency table: shape = (n_ref, n_est)
    contingency = _contingency_matrix(y_ref, y_est).astype(float)

    # Normalize by the number of frames
    contingency = contingency / len(y_ref)

    # Compute the marginals
    p_est = contingency.sum(axis=0)
    p_ref = contingency.sum(axis=1)

    # H(true | prediction) = sum_j P[estimated = j] *
    # sum_i P[true = i | estimated = j] log P[true = i | estimated = j]
    # entropy sums over axis=0, which is true labels

    true_given_est = p_est.dot(scipy.stats.entropy(contingency, base=2))
    pred_given_ref = p_ref.dot(scipy.stats.entropy(contingency.T, base=2))

    if marginal:
        # Normalize conditional entropy by marginal entropy
        z_ref = scipy.stats.entropy(p_ref, base=2)
        z_est = scipy.stats.entropy(p_est, base=2)
    else:
        z_ref = np.log2(contingency.shape[0])
        z_est = np.log2(contingency.shape[1])

    score_under = 0.0
    if z_ref > 0:
        score_under = 1. - true_given_est / z_ref

    score_over = 0.0
    if z_est > 0:
        score_over = 1. - pred_given_ref / z_est

    f_measure = mir_eval_simple_utils.f_measure(score_over, score_under, beta=beta)

    return score_over, score_under, f_measure

def pairwise(reference_intervals, reference_labels,
             estimated_intervals, estimated_labels,
             frame_size=0.1, beta=1.0):
    """Frame-clustering segmentation evaluation by pair-wise agreement.

    Examples
    --------
    >>> (ref_intervals,
    ...  ref_labels) = mir_eval.io.load_labeled_intervals('ref.lab')
    >>> (est_intervals,
    ...  est_labels) = mir_eval.io.load_labeled_intervals('est.lab')
    >>> # Trim or pad the estimate to match reference timing
    >>> (ref_intervals,
    ...  ref_labels) = mir_eval.util.adjust_intervals(ref_intervals,
    ...                                               ref_labels,
    ...                                               t_min=0)
    >>> (est_intervals,
    ...  est_labels) = mir_eval.util.adjust_intervals(
    ...     est_intervals, est_labels, t_min=0, t_max=ref_intervals.max())
    >>> precision, recall, f = mir_eval.structure.pairwise(ref_intervals,
    ...                                                    ref_labels,
    ...                                                    est_intervals,
    ...                                                    est_labels)

    Parameters
    ----------
    reference_intervals : np.ndarray, shape=(n, 2)
        reference segment intervals, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.
    reference_labels : list, shape=(n,)
        reference segment labels, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.
    estimated_intervals : np.ndarray, shape=(m, 2)
        estimated segment intervals, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.
    estimated_labels : list, shape=(m,)
        estimated segment labels, in the format returned by
        :func:`mir_eval.io.load_labeled_intervals`.
    frame_size : float > 0
        length (in seconds) of frames for clustering
        (Default value = 0.1)
    beta : float > 0
        beta value for F-measure
        (Default value = 1.0)

    Returns
    -------
    precision : float > 0
        Precision of detecting whether frames belong in the same cluster
    recall : float > 0
        Recall of detecting whether frames belong in the same cluster
    f : float > 0
        F-measure of detecting whether frames belong in the same cluster

    """
    validate_structure(reference_intervals, reference_labels,
                       estimated_intervals, estimated_labels)

    # Check for empty annotations.  Don't need to check labels because
    # validate_structure makes sure they're the same size as intervals
    if reference_intervals.size == 0 or estimated_intervals.size == 0:
        return 0., 0., 0.

    # Generate the cluster labels
    y_ref = mir_eval_simple_utils.intervals_to_samples(reference_intervals,
                                      reference_labels,
                                      sample_size=frame_size)[-1]

    y_ref = mir_eval_simple_utils.index_labels(y_ref)[0]

    # Map to index space
    y_est = mir_eval_simple_utils.intervals_to_samples(estimated_intervals,
                                      estimated_labels,
                                      sample_size=frame_size)[-1]

    y_est = mir_eval_simple_utils.index_labels(y_est)[0]

    # Build the reference label agreement matrix
    agree_ref = np.equal.outer(y_ref, y_ref)
    # Count the unique pairs
    n_agree_ref = (agree_ref.sum() - len(y_ref)) / 2.0

    # Repeat for estimate
    agree_est = np.equal.outer(y_est, y_est)
    n_agree_est = (agree_est.sum() - len(y_est)) / 2.0

    # Find where they agree
    matches = np.logical_and(agree_ref, agree_est)
    n_matches = (matches.sum() - len(y_ref)) / 2.0

    precision = n_matches / n_agree_est
    recall = n_matches / n_agree_ref
    f_measure = mir_eval_simple_utils.f_measure(precision, recall, beta=beta)

    return precision, recall, f_measure