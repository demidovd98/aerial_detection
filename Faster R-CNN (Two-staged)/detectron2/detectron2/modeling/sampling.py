# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.layers import nonzero_tuple

__all__ = ["subsample_labels","subsample_labels_biased"]


def subsample_labels(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx



def subsample_labels_biased(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.

    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.

    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    if num_pos==64:

        rare = [2,3,4,5,6,9,10,11,12]
        common = [7,1,13,14]
        freq = [0,8]

        positive_labels = labels[positive]

        pos_cpu = positive_labels.cpu()

        rare_mask = pos_cpu.apply_(lambda x: x in rare).bool()
        common_mask = pos_cpu.apply_(lambda x: x in common).bool()
        freq_mask = pos_cpu.apply_(lambda x: x in freq).bool()

        rare_positive = positive[rare_mask]
        common_positive = positive[common_mask]
        freq_positive = positive[freq_mask]

        pos_idx1 = rare_positive[torch.randperm(rare_positive.numel(), device=positive.device)[:24]] #24
        pos_idx2 = common_positive[torch.randperm(common_positive.numel(), device=positive.device)[:20]] #20
        pos_idx3 = freq_positive[torch.randperm(freq_positive.numel(), device=positive.device)[:20]] #20

        pos_idx = torch.cat((pos_idx1,pos_idx2,pos_idx3),0)

    else:
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        pos_idx = positive[perm1]

    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    neg_idx = negative[perm2]
    
    return pos_idx, neg_idx
