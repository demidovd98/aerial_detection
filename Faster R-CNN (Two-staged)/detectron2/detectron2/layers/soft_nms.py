import torch

from detectron2.structures import Boxes, RotatedBoxes, pairwise_iou, pairwise_iou_rotated

def compute_RDIoU(pred, target, eps=1e-7):
    r"""`Implementation of Distance-IoU Loss: Faster and Better
    Learning for Bounding Box Regression, https://arxiv.org/abs/1911.08287`_.
    Code is modified from https://github.com/Zzh-tju/DIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    ag = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

    cw = enclose_wh[:, 0]
    ch = enclose_wh[:, 1]

    c2 = cw ** 2 + ch ** 2 + eps

    b1_x1, b1_y1 = pred[:, 0], pred[:, 1]
    b1_x2, b1_y2 = pred[:, 2], pred[:, 3]
    b2_x1, b2_y1 = target[:, 0], target[:, 1]
    b2_x2, b2_y2 = target[:, 2], target[:, 3]

    left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
    right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
    rho2 = left + right

    # DIoU
    #dious = ious - rho2 / c2
    #loss = 1 - dious

    #loss = loss.sum()

    return rho2 / c2


def soft_nms(boxes, scores, idxs, method, gaussian_sigma, linear_threshold, prune_threshold):
    """
    Performs soft non-maximum suppression algorithm on axis aligned boxes
    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        method (str):
           one of ['gaussian', 'linear', 'hard']
           see paper for details. users encouraged not to use "hard", as this is the
           same nms available elsewhere in detectron2
        gaussian_sigma (float):
           parameter for Gaussian penalty function
        linear_threshold (float):
           iou threshold for applying linear decay. Nt from the paper
           re-used as threshold for standard "hard" nms
        prune_threshold (float):
           boxes with scores below this threshold are pruned at each iteration.
           Dramatically reduces computation time. Authors use values in [10e-4, 10e-2]
    Returns:
        tuple(Tensor, Tensor):
            [0]: int64 tensor with the indices of the elements that have been kept
            by Soft NMS, sorted in decreasing order of scores
            [1]: float tensor with the re-scored scores of the elements that were kept
"""
    return _soft_nms(
        Boxes,
        pairwise_iou,
        boxes,
        scores,
        idxs,
        method,
        gaussian_sigma,
        linear_threshold,
        prune_threshold,
    )


def soft_nms_rotated(boxes, scores, method, gaussian_sigma, linear_threshold, prune_threshold):
    """
    Performs soft non-maximum suppression algorithm on rotated boxes
    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        method (str):
           one of ['gaussian', 'linear', 'hard']
           see paper for details. users encouraged not to use "hard", as this is the
           same nms available elsewhere in detectron2
        gaussian_sigma (float):
           parameter for Gaussian penalty function
        linear_threshold (float):
           iou threshold for applying linear decay. Nt from the paper
           re-used as threshold for standard "hard" nms
        prune_threshold (float):
           boxes with scores below this threshold are pruned at each iteration.
           Dramatically reduces computation time. Authors use values in [10e-4, 10e-2]
    Returns:
        tuple(Tensor, Tensor):
            [0]: int64 tensor with the indices of the elements that have been kept
            by Soft NMS, sorted in decreasing order of scores
            [1]: float tensor with the re-scored scores of the elements that were kept    """
    return _soft_nms(
        RotatedBoxes,
        pairwise_iou_rotated,
        boxes,
        scores,
        method,
        gaussian_sigma,
        linear_threshold,
        prune_threshold,
    )


def batched_soft_nms(
    boxes, scores, idxs, method, gaussian_sigma, linear_threshold, prune_threshold
):
    """
    Performs soft non-maximum suppression in a batched fashion.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    Args:
        boxes (Tensor[N, 4]):
           boxes where NMS will be performed. They
           are expected to be in (x1, y1, x2, y2) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        method (str):
           one of ['gaussian', 'linear', 'hard']
           see paper for details. users encouraged not to use "hard", as this is the
           same nms available elsewhere in detectron2
        gaussian_sigma (float):
           parameter for Gaussian penalty function
        linear_threshold (float):
           iou threshold for applying linear decay. Nt from the paper
           re-used as threshold for standard "hard" nms
        prune_threshold (float):
           boxes with scores below this threshold are pruned at each iteration.
           Dramatically reduces computation time. Authors use values in [10e-4, 10e-2]
    Returns:
        tuple(Tensor, Tensor):
            [0]: int64 tensor with the indices of the elements that have been kept
            by Soft NMS, sorted in decreasing order of scores
            [1]: float tensor with the re-scored scores of the elements that were kept
    """
    if boxes.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=boxes.device),
            torch.empty((0,), dtype=torch.float32, device=scores.device),
        )
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    #max_coordinate = boxes.max()
    #offsets = idxs.to(boxes) * (max_coordinate + 1)
    #boxes_for_nms = boxes + offsets[:, None]
    return soft_nms(
        boxes, scores, idxs, method, gaussian_sigma, linear_threshold, prune_threshold
    )


def batched_soft_nms_rotated(
    boxes, scores, idxs, method, gaussian_sigma, linear_threshold, prune_threshold
):
    """
    Performs soft non-maximum suppression in a batched fashion on rotated bounding boxes.
    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.
    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        method (str):
           one of ['gaussian', 'linear', 'hard']
           see paper for details. users encouraged not to use "hard", as this is the
           same nms available elsewhere in detectron2
        gaussian_sigma (float):
           parameter for Gaussian penalty function
        linear_threshold (float):
           iou threshold for applying linear decay. Nt from the paper
           re-used as threshold for standard "hard" nms
        prune_threshold (float):
           boxes with scores below this threshold are pruned at each iteration.
           Dramatically reduces computation time. Authors use values in [10e-4, 10e-2]
    Returns:
        tuple(Tensor, Tensor):
            [0]: int64 tensor with the indices of the elements that have been kept
            by Soft NMS, sorted in decreasing order of scores
            [1]: float tensor with the re-scored scores of the elements that were kept
    """
    if boxes.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=boxes.device),
            torch.empty((0,), dtype=torch.float32, device=scores.device),
        )
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes[:, :2].max() + torch.norm(boxes[:, 2:4], 2, dim=1).max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes.clone()
    boxes_for_nms[:, :2] += offsets[:, None]
    return soft_nms_rotated(
        boxes_for_nms, scores, method, gaussian_sigma, linear_threshold, prune_threshold
    )


def _soft_nms(
    box_class,
    pairwise_iou_func,
    boxes,
    scores,
    idxs,
    method,
    gaussian_sigma,
    linear_threshold,
    prune_threshold,
):
    """
    Soft non-max suppression algorithm.
    Implementation of [Soft-NMS -- Improving Object Detection With One Line of Codec]
    (https://arxiv.org/abs/1704.04503)
    Args:
        box_class (cls): one of Box, RotatedBoxes
        pairwise_iou_func (func): one of pairwise_iou, pairwise_iou_rotated
        boxes (Tensor[N, ?]):
           boxes where NMS will be performed
           if Boxes, in (x1, y1, x2, y2) format
           if RotatedBoxes, in (x_ctr, y_ctr, width, height, angle_degrees) format
        scores (Tensor[N]):
           scores for each one of the boxes
        method (str):
           one of ['gaussian', 'linear', 'hard']
           see paper for details. users encouraged not to use "hard", as this is the
           same nms available elsewhere in detectron2
        gaussian_sigma (float):
           parameter for Gaussian penalty function
        linear_threshold (float):
           iou threshold for applying linear decay. Nt from the paper
           re-used as threshold for standard "hard" nms
        prune_threshold (float):
           boxes with scores below this threshold are pruned at each iteration.
           Dramatically reduces computation time. Authors use values in [10e-4, 10e-2]
    Returns:
        tuple(Tensor, Tensor):
            [0]: int64 tensor with the indices of the elements that have been kept
            by Soft NMS, sorted in decreasing order of scores
            [1]: float tensor with the re-scored scores of the elements that were kept
    """

    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes = boxes + offsets[:, None]
    idxs_orig = idxs

    boxes = boxes.clone()
    scores = scores.clone()
    idxs = torch.arange(scores.size()[0])

    idxs_out = []
    scores_out = []

    while scores.numel() > 0:
        top_idx = torch.argmax(scores)
        
        idxs_out.append(idxs[top_idx].item())
        scores_out.append(scores[top_idx].item())

        top_box = boxes[top_idx]
        ious = pairwise_iou_func(box_class(top_box.unsqueeze(0)), box_class(boxes))[0]

        if method == "linear":
            decay = torch.ones_like(ious)
            decay_mask = ious > linear_threshold
            decay[decay_mask] = 1 - ious[decay_mask]
        elif method == "gaussian":
            decay = torch.exp(-torch.pow(ious, 2) / gaussian_sigma)
        elif method == "hard":  # standard NMS
            decay = (ious < linear_threshold).float()
        elif method == 'diou':
            d = compute_RDIoU(top_box.unsqueeze(0), boxes)
            decay = ((ious - d) < linear_threshold).float()
        else:
            raise NotImplementedError("{} soft nms method not implemented.".format(method))

        scores *= decay
        keep = scores > prune_threshold
        keep[top_idx] = False

        boxes = boxes[keep]
        scores = scores[keep]
        idxs = idxs[keep]
        idxs_orig = idxs_orig[keep]

    return torch.tensor(idxs_out).to(boxes.device), torch.tensor(scores_out).to(scores.device)