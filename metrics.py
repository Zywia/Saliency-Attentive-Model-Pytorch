"""
Saliency metrics implementation based on matlab code from https://github.com/cvzoya/saliency
that was used in http://saliency.mit.edu/
"""

import numpy as np


# Helpers function

def normalize(x: np.ndarray) -> np.ndarray:
    """Normalize each value of the array to be in [0,1]"""
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# Metrics based on human fixation points

def auc_judd(sal_map: np.ndarray, fixations: np.ndarray) -> float:
    """
    Calculate the AUC metric of the a saliency map predicts human fixation points based on code
    from https://github.com/cvzoya/saliency/blob/master/code_forMetrics/AUC_Judd.m
    :param sal_map: 2d array with a saliency map returned by a model
    :param fixations: 2d array with of shape (N, 2) with human fixation points

    :return: float, AUC score
    """

    assert len(fixations.shape) == 2
    assert fixations.shape[1] == 2
    assert len(sal_map.shape) == 2

    n_pixels = sal_map.shape[0] * sal_map.shape[1]
    n_fixations = fixations.shape[0]

    # normalize saliency map
    sal_map = normalize(sal_map)

    # calculate saliency values at fixation points
    sal_at_fix = np.sort(sal_map[fixations[:, 0], fixations[:, 1]])[::-1]  # descending

    # prepare true-pos and false-pos lists
    area = [(0, 0)]
    for i, thres in enumerate(sal_at_fix):
        n_fix_above_th = i + 1
        tp = n_fix_above_th / n_fixations  # ratio sal_map values at fixation locations above threshold

        above_th = np.sum(sal_map >= thres)  # total number of sal_map values above threshold
        fp = (above_th - n_fix_above_th) / (n_pixels - n_fixations)  # ratio other sal map values above threshold

        area.append((tp, fp))

    area.append((1, 1))

    # calculate the actual area under the ROC curve
    area = np.stack(area)
    score = np.trapz(area[:, 0], area[:, 1])

    return score


def auc_borji(sal_map: np.ndarray, fixations: np.ndarray, n_splits: int = 100, step_size: float = 0.1) -> float:
    """
    Calculate the AUC metric of the a saliency map predicts human fixation points based on code
    from https://github.com/cvzoya/saliency/blob/master/code_forMetrics/AUC_Borji.m
    :param sal_map: 2d array with a saliency map returned by a model
    :param fixations: 2d array with of shape (N, 2) with human fixation points
    :param n_splits: int, how many random splits per fixation to sample
    :param step_size: float, step size for determining the ROC curve thresholds

    :return: float, AUC score
    """

    assert len(fixations.shape) == 2
    assert fixations.shape[1] == 2
    assert len(sal_map.shape) == 2
    assert 0. < step_size < 1.

    n_pixels = sal_map.shape[0] * sal_map.shape[1]
    n_fixations = fixations.shape[0]
    height = sal_map.shape[0]

    # normalize saliency map
    sal_map = normalize(sal_map)

    sal_at_fix = sal_map[fixations[:, 0], fixations[:, 1]]

    # for each fixation, sample n_splits random pixels form sal_map
    rand_pixel = np.random.randint(low=0, high=n_pixels, size=(n_splits, n_fixations))

    # prepare a list for scores
    aucs = []

    # for each random split
    for split_pixeles in rand_pixel:

        # get sal_map values at random pixels
        rand_sal_val = sal_map[split_pixeles % height, split_pixeles // height]

        # prepare thresholds
        max_thres = np.max(np.concatenate([sal_at_fix, rand_sal_val]))
        thresholds = np.arange(start=0, stop=max_thres, step=step_size)[::-1]  # descending

        # prepare true-pos and false-pos lists
        area = [(0, 0)]

        # calculate ROC curve for each threshold
        for thres in thresholds:

            tp = np.sum(sal_at_fix >= thres) / n_fixations
            fp = np.sum(rand_sal_val >= thres) / n_fixations

            assert 0 <= tp <= 1
            assert 0 <= fp <= 1

            area.append((tp, fp))

        area.append((1, 1))

        # calculate the actual area under the ROC curve
        area = np.stack(area)
        auc = np.trapz(area[:, 0], area[:, 1])

        aucs.append(auc)

    return np.mean(aucs)


def auc_shuffled(sal_map: np.ndarray,
                 fixations: np.ndarray,
                 other_fix: np.ndarray,
                 n_splits: int = 100,
                 step_size: float = 0.1) -> float:
    """
    Calculate the AUC metric of the a saliency map predicts human fixation points using a random other fixations
    as a comparison, based on code from https://github.com/cvzoya/saliency/blob/master/code_forMetrics/AUC_shuffled.m
    :param sal_map: 2d array with a saliency map returned by a model
    :param fixations: 2d array with of shape (N, 2) with human fixation points
    :param other_fix: 2d array of shape (N, 2) with fixations from M other random images (Borji uses M=10)
    :param n_splits: int, how many random splits to sample
    :param step_size: float, step size for determining the ROC curve thresholds

    :return: float, shuffeled AUC score
    """

    assert len(fixations.shape) == 2
    assert fixations.shape[1] == 2
    assert len(other_fix.shape) == 2
    assert other_fix.shape[1] == 2
    assert len(sal_map.shape) == 2

    n_fixations = fixations.shape[0]

    # normalize saliency map
    sal_map = normalize(sal_map)

    sal_at_fix = sal_map[fixations[:, 0], fixations[:, 1]]

    # for each split sample points from other fixations
    n_other_fix = min(n_fixations, other_fix.shape[0])
    rand_pixels = []
    for i in range(n_splits):
        # randomize the order of other_fix
        rand_other_fix = other_fix[np.random.permutation(other_fix.shape[0])]
        # sample pixel form the other fixations
        rand_ind = np.random.choice(rand_other_fix.shape[0], size=n_other_fix)
        rand_pixels.append(rand_other_fix[rand_ind])

    # prepare a list for scores
    aucs = []

    # for each random split
    for split_pixeles in rand_pixels:

        # get sal_map values at random pixels
        rand_sal_val = sal_map[split_pixeles[:, 0], split_pixeles[:, 1]]

        # prepare thresholds
        max_thres = np.max(np.concatenate([sal_at_fix, rand_sal_val]))
        thresholds = np.arange(start=0, stop=max_thres, step=step_size)[::-1]  # descending

        # prepare true-pos and false-pos lists
        area = [(0, 0)]

        # calculate ROC curve for each threshold
        for thres in thresholds:
            tp = np.sum(sal_at_fix >= thres) / n_fixations
            fp = np.sum(rand_sal_val >= thres) / n_other_fix

            area.append((tp, fp))

        area.append((1, 1))

        # calculate the actual area under the ROC curve
        area = np.stack(area)
        auc = np.trapz(area[:, 0], area[:, 1])

        aucs.append(auc)

    return np.mean(aucs)


def nss(sal_map: np.ndarray, fixations: np.ndarray) -> float:
    """
    Calculate the Normalized Scanpath Saliency between two different saliency maps as the mean
    value of the normalized saliency map at fixation locations, based on code
    from: https://github.com/cvzoya/saliency/blob/master/code_forMetrics/NSS.m
    :param sal_map: 2d array with a saliency map returned by a model
    :param fixations: 2d array with of shape (N, 2) with human fixation points

    :return: float, NSS score
    """

    assert len(fixations.shape) == 2
    assert fixations.shape[1] == 2
    assert len(sal_map.shape) == 2

    # normalize saliency map
    sal_map = normalize(sal_map)

    # calculate the NSS score
    score = np.mean(sal_map[fixations[:, 0], fixations[:, 1]])
    return score


def info_gain(sal_map: np.ndarray, fixations: np.ndarray, baseline_map: np.ndarray) -> float:
    """
    Calculate information-gain of the saliency map over a baseline map, based on code
    from https://github.com/cvzoya/saliency/blob/master/code_forMetrics/InfoGain.m
    :param sal_map: 2d array with a saliency map returned by a model
    :param fixations: 2d array with of shape (N, 2) with human fixation points
    :param baseline_map: 2d array with a baseline saliency map of the same shape as sal_map

    :return: float, IG score
    """

    assert len(fixations.shape) == 2
    assert fixations.shape[1] == 2
    assert len(sal_map.shape) == 2
    assert len(baseline_map.shape) == 2
    assert sal_map.shape == baseline_map.shape

    eps = np.finfo(float).eps

    # normalize and create approx prob dist from both saliency maps
    if np.any(sal_map > 0):
        sal_map = normalize(sal_map)
        sal_map = sal_map / sal_map.sum()

    if np.any(baseline_map > 0):
        baseline_map = normalize(baseline_map)
        baseline_map = baseline_map / baseline_map.sum()

    # get values at fixations from both maps
    sal_at_fix = sal_map[fixations[:, 0], fixations[:, 1]]
    baseline_at_fix = baseline_map[fixations[:, 0], fixations[:, 1]]

    # calculate information gain
    score = np.mean(np.log2(sal_at_fix + eps) - np.log2(baseline_at_fix + eps))  # eps to avoid log(0)

    return score


# Metrics based on comapring two saliency maps

def similarity(sal_map: np.ndarray, other_map: np.ndarray) -> float:
    """
    Calculate the similarity between two different saliency maps when viewed as distributions (equivalent
    to histogram intersection), based on code from:
    https://github.com/cvzoya/saliency/blob/master/code_forMetrics/similarity.m
    :param sal_map: 2d array with a saliency map returned by a model
    :param other_map: 2d array with a baseline saliency map of the same shape as sal_map

    :return: float, similarirt score
    """

    assert len(sal_map.shape) == 2
    assert len(other_map.shape) == 2
    assert sal_map.shape == other_map.shape

    # normalize and create a approx. prob distribution from both maps
    if np.any(sal_map > 0):  # avoid div by 0
        sal_map = normalize(sal_map)
        sal_map = sal_map / sal_map.sum()

    if np.any(other_map > 0):  # avoid div by 0
        other_map = normalize(other_map)
        other_map = other_map / other_map.sum()

    # calculate the similarity score
    score = np.sum(np.minimum(sal_map, other_map))

    return score


def cc(sal_map: np.ndarray, other_map: np.ndarray) -> float:
    """
    Calculate the linear correlation coefficient between two different saliency maps (also called
    Pearson's linear coefficient), based on https://github.com/cvzoya/saliency/blob/master/code_forMetrics/CC.m
    :param sal_map: 2d array with a saliency map returned by a model
    :param other_map: 2d array with a baseline saliency map of the same shape as sal_map

    :return: float, CC score
    """

    assert len(sal_map.shape) == 2
    assert len(other_map.shape) == 2
    assert sal_map.shape == other_map.shape

    # normalize both saliency maps
    sal_map = normalize(sal_map)
    other_map = normalize(other_map)

    # calculate the correlation coefficient
    sal_map = sal_map - np.mean(sal_map)
    other_map = other_map - np.mean(other_map)

    score = (sal_map * other_map).sum() / np.sqrt((sal_map ** 2).sum() * (other_map ** 2).sum())

    return score


def kl_div(sal_map: np.ndarray, other_map: np.ndarray) -> float:
    """
    Calculate the KL-divergence between two different saliency maps when viewed as distributions: it is
    a non-symmetric measure of the information lost when saliency map is used to estimate the other map.
    :param sal_map: 2d array with a saliency map returned by a model
    :param other_map: 2d array with a baseline saliency map of the same shape as sal_map

    :return: float, CC score
    """

    assert len(sal_map.shape) == 2
    assert len(other_map.shape) == 2
    assert sal_map.shape == other_map.shape

    eps = np.finfo(float).eps

    # normalize and create a approx. prob distribution from both maps
    if np.any(sal_map > 0):  # avoid div by 0
        sal_map = normalize(sal_map)
        sal_map = sal_map / sal_map.sum()

    if np.any(other_map > 0):  # avoid div by 0
        other_map = normalize(other_map)
        other_map = other_map / other_map.sum()

    # compute KL divergence
    score = np.sum(other_map * np.log(eps + (other_map / (sal_map + eps))))

    return -score
