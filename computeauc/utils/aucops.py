import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats


## DeLong implementation
# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
# https://github.com/yandexdataschool/roc_comparison/blob/master/example.py
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float32)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float32)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float32)
    ty = np.empty([k, n], dtype=np.float32)
    tz = np.empty([k, m + n], dtype=np.float32)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    pos_idx = np.where(ground_truth == 1)[0]
    neg_idx = np.where(ground_truth == 0)[0]
    order = np.concatenate([pos_idx, neg_idx])
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)

    predictions_sorted_transposed = predictions[np.newaxis, order]

    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert (
        len(aucs) == 1
    ), "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)

    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[
        :, order
    ]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)


def compute_auc_ci(auc, auc_cov, alpha):

    # Compute CI
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)

    # Maximum AUC=1
    ci[ci > 1] = 1

    return ci


def bootstrap_mean_auc(label: np.ndarray, preds: np.ndarray, n_bootstraps, alpha):

    # Compute mean auc
    mean_auc = roc_auc_score(label, preds)

    # Seed for reproducibility
    rng_seed = 42
    rng = np.random.RandomState(rng_seed)

    # List of computed aucs
    bootstrap_scores = []

    # Number of samples
    samples = label.shape[0]

    # Loop
    for i in range(n_bootstraps):

        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, samples, samples)

        # We need at least one positive and one negative sample for ROC AUC to be defined: reject the sample
        if len(np.unique(label[indices])) < 2:
            continue
        else:
            score = roc_auc_score(label[indices], preds[indices])
            bootstrap_scores.append(score)

    # Sort scores
    bootstrap_scores = np.array(bootstrap_scores)

    # Compute lower and upper bounds
    lower = (1 - alpha) / 2 * 100
    upper = 100 - lower

    # Compute confidence
    confidence_lower = np.percentile(bootstrap_scores, lower)
    confidence_upper = np.percentile(bootstrap_scores, upper)

    return mean_auc, [confidence_lower, confidence_upper]


def auc_calculator(label: dict, preds: dict, labelnames: list, n_bootstraps: int = 1000, alpha: float = 0.95):

    # Create empty dictionary of aucs
    aucs = dict()

    # Iterate over keys from predictions
    for (model, part), _ in preds.items():

        # Print computing pair model, partition
        print(f"Computing: {model, part}", flush=True)

        # Load labels and preds for this model + partition
        part_label = label[part]
        model_part_preds = preds[(model, part)]

        # Iterate over labels
        for i, label_name in enumerate(labelnames):

            # Read columns into contiguous format
            local_label = part_label[:, i].numpy()
            local_preds = model_part_preds[:, i].numpy()

            auc, auc_cov = delong_roc_variance(local_label, local_preds)

            # Compute auc ci
            auc_ci = compute_auc_ci(auc=auc, auc_cov=auc_cov, alpha=alpha)

            # Introduce data in dict
            aucs[(model, part, label_name)] = [
                float(auc),
                float(auc_ci[0]),
                float(auc_ci[1]),
            ]

        # Compute mean auc
        auc, auc_ci = bootstrap_mean_auc(
            label=part_label.numpy(),
            preds=model_part_preds.numpy(),
            n_bootstraps=n_bootstraps,
            alpha=alpha,
        )

        # Introduce data in dict
        aucs[(model, part, "mean")] = [
            float(auc),
            float(auc_ci[0]),
            float(auc_ci[1]),
        ]

    return aucs
