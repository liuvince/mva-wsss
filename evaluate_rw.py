import numpy as np
import os
import torchvision
import six
from PIL import Image
from utils.utils import init_params, seed_reproducer, mkdir
from tqdm import tqdm
import pandas as pd

# https://chainercv.readthedocs.io/en/stable/reference/evaluations.html

def calc_semantic_segmentation_confusion(pred_labels, gt_labels):
    """Collect a confusion matrix.
    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.
    Args:
        pred_labels (iterable of numpy.ndarray): See the table in
            :func:`chainercv.evaluations.eval_semantic_segmentation`.
        gt_labels (iterable of numpy.ndarray): See the table in
            :func:`chainercv.evaluations.eval_semantic_segmentation`.
    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of pixels
        that are labeled as class :math:`i` by the ground truth and
        class :math:`j` by the prediction.
    """
    pred_labels = iter(pred_labels)
    gt_labels = iter(gt_labels)

    n_class = 0
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in six.moves.zip(pred_labels, gt_labels):
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class**2).reshape((n_class, n_class))

    for iter_ in (pred_labels, gt_labels):
        # This code assumes any iterator does not contain None as its items.
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same')
    return confusion

def calc_semantic_segmentation_iou(confusion):
    """Calculate Intersection over Union with a given confusion matrix.
    The definition of Intersection over Union (IoU) is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.
    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    Args:
        confusion (numpy.ndarray): A confusion matrix. Its shape is
            :math:`(n\_class, n\_class)`.
            The :math:`(i, j)` th element corresponds to the number of pixels
            that are labeled as class :math:`i` by the ground truth and
            class :math:`j` by the prediction.
    Returns:
        numpy.ndarray:
        An array of IoUs for the :math:`n\_class` classes. Its shape is
        :math:`(n\_class,)`.
    """
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) -
                       np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou



if __name__ == "__main__":
    # Make experiment reproducible
    seed_reproducer(2020)

    A = torchvision.datasets.VOCSegmentation('data',
                                         year='2012',
                                         image_set='train',
                                         download=False)

    hparams = init_params()

    labels = []
    df = pd.read_csv("experiment/data_train.csv")["filename"].values
    for _, label in A:
        label = np.array(label)
        labels.append(np.where((label == 255), -1, label))

    preds = []
    for id in tqdm(sorted(df)):
        id = os.path.splitext(os.path.basename(id))[0]
        cam = np.asarray(Image.open(os.path.join(hparams.cam_dir, id + ".png")))
        preds.append(cam.copy())
    confusion = calc_semantic_segmentation_confusion(preds, labels)

    iou = calc_semantic_segmentation_iou(confusion)

    print({'miou': np.nanmean(iou)})
